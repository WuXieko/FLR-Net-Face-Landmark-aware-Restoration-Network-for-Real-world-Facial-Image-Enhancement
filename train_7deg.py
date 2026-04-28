# train_7deg.py
# ============================================================
# 训练 NAFNet (7-deg)：7种退化 + 无 Landmark Loss
# ============================================================
# 用途：消融实验，分离退化改进和 Landmark Loss 的贡献
# Checkpoint: checkpoints_7deg/
# TensorBoard: runs/ffhq_7deg/
# ============================================================

import os, sys, time, argparse, torch, torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models.nafnet import build_model
from losses.losses import CombinedLoss
from datapipe.dataset import FaceDataset

DATA_DIR = "data/ffhq"
PATCH_SIZE = 256
NUM_WORKERS = 4
BATCH_SIZE = 8
LR = 2e-4
EPOCHS = 80
WIDTH = 32
SAVE_EVERY = 5
LOG_EVERY = 50
CKPT_DIR = "checkpoints_7deg"
LOG_DIR = "runs/ffhq_7deg"


def find_latest_checkpoint(ckpt_dir):
    if not os.path.exists(ckpt_dir): return None
    ckpts = [f for f in os.listdir(ckpt_dir) if f.startswith("epoch_") and f.endswith(".pth")]
    if not ckpts: return None
    ckpts.sort(key=lambda x: int(x.replace("epoch_","").replace(".pth","")))
    return os.path.join(ckpt_dir, ckpts[-1])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--ckpt", type=str, default=None)
    args = parser.parse_args()

    os.makedirs(CKPT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("NAFNet (7-deg) - 7 degradations, NO Landmark Loss")
    print("Purpose: Ablation study - isolate degradation contribution")
    print("=" * 60)
    print(f"Device: {device} ({torch.cuda.get_device_name(0)})")

    dataset = FaceDataset(DATA_DIR, patch_size=PATCH_SIZE)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True, drop_last=True,
        persistent_workers=True if NUM_WORKERS > 0 else False)
    print(f"Dataset: {len(dataset)} images | {len(dataloader)} batches/epoch")

    model = build_model(width=WIDTH).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    # 关键：use_landmark=False，不用 Landmark Loss
    criterion = CombinedLoss(use_landmark=False).to(device)
    print("Loss: L1 + Perceptual + FFT (NO Landmark)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
    scaler = torch.amp.GradScaler('cuda')

    start_epoch = 0; global_step = 0; best_loss = float('inf')

    if args.resume:
        ckpt_path = args.ckpt or find_latest_checkpoint(CKPT_DIR)
        if ckpt_path and os.path.exists(ckpt_path):
            print(f"\nResume: {ckpt_path}")
            checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)
            sd = checkpoint['model_state_dict']
            model.load_state_dict({k.replace("_orig_mod.",""):v for k,v in sd.items()})
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch']
            best_loss = checkpoint.get('loss', float('inf'))
            global_step = start_epoch * len(dataloader)
            print(f"   From epoch {start_epoch}")

    writer = SummaryWriter(LOG_DIR)
    remaining = EPOCHS - start_epoch
    print(f"\nRemaining: {remaining} epochs, ~{len(dataloader)*remaining*0.15/60:.0f} min")
    print("=" * 60)

    for epoch in range(start_epoch, EPOCHS):
        model.train(); epoch_loss = 0.0; t0 = time.time()

        for batch_idx, (lq, hq) in enumerate(dataloader):
            lq, hq = lq.to(device, non_blocking=True), hq.to(device, non_blocking=True)

            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                pred = model(lq)
                loss, loss_dict = criterion(pred, hq)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer); scaler.update()
            optimizer.zero_grad(set_to_none=True)

            epoch_loss += loss.item(); global_step += 1

            if global_step % LOG_EVERY == 0:
                writer.add_scalar("Loss/total", loss.item(), global_step)
                writer.add_scalar("Loss/l1", loss_dict["l1"], global_step)
                writer.add_scalar("Loss/perceptual", loss_dict["perc"], global_step)
                writer.add_scalar("Loss/fft", loss_dict["fft"], global_step)

            if global_step % 500 == 0:
                with torch.no_grad():
                    writer.add_images("Images/1_LQ", lq[:4].clamp(0,1), global_step)
                    writer.add_images("Images/2_Pred", pred[:4].float().clamp(0,1), global_step)
                    writer.add_images("Images/3_HQ", hq[:4].clamp(0,1), global_step)

            if (batch_idx+1)%100==0 or batch_idx==0:
                elapsed = time.time()-t0
                eta = (elapsed/(batch_idx+1))*(len(dataloader)-batch_idx-1)/60
                print(f"  Epoch [{epoch+1}/{EPOCHS}] [{batch_idx+1}/{len(dataloader)}] "
                      f"Loss: {loss.item():.4f} (l1={loss_dict['l1']:.4f} "
                      f"perc={loss_dict['perc']:.4f} fft={loss_dict['fft']:.4f}) "
                      f"ETA: {eta:.1f}min")

        scheduler.step()
        avg_loss = epoch_loss/len(dataloader); elapsed = time.time()-t0
        est_rem = elapsed*(EPOCHS-epoch-1)/60
        print(f"\n>>> Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f} | "
              f"LR: {optimizer.param_groups[0]['lr']:.2e} | "
              f"Time: {elapsed:.0f}s | Remaining: ~{est_rem:.0f}min\n")

        writer.add_scalar("Epoch/avg_loss", avg_loss, epoch)

        if (epoch+1)%SAVE_EVERY==0:
            p = os.path.join(CKPT_DIR, f"epoch_{epoch+1}.pth")
            torch.save({'epoch':epoch+1,'model_state_dict':model.state_dict(),
                        'optimizer_state_dict':optimizer.state_dict(),
                        'scheduler_state_dict':scheduler.state_dict(),'loss':avg_loss}, p)
            print(f"   Saved: {p}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            bp = os.path.join(CKPT_DIR, "best.pth")
            torch.save(model.state_dict(), bp)
            print(f"   Best: {bp} (loss={best_loss:.4f})")

    writer.close()
    print(f"\nDone! Best loss: {best_loss:.4f}")

if __name__ == "__main__":
    main()
    main()