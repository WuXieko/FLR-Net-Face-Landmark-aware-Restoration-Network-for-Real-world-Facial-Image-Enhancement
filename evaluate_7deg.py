# evaluate_7deg.py
# ============================================================
# FLR-Net vs NAFNet 对比（7种退化条件）
# ============================================================
# python evaluate_7deg.py
# ============================================================

import os, sys, csv, random, glob, numpy as np, cv2, torch
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models.nafnet import build_model
from datapipe.degradation import BlindDegradation

random.seed(42); np.random.seed(42); torch.manual_seed(42)
OUTPUT_DIR = "evaluation_7deg"


def calculate_psnr(i1, i2):
    a1, a2 = np.array(i1).astype(np.float64), np.array(i2).astype(np.float64)
    mse = np.mean((a1 - a2)**2)
    return float('inf') if mse == 0 else 10*np.log10(255**2/mse)

def calculate_ssim(i1, i2):
    a1, a2 = np.array(i1).astype(np.float64), np.array(i2).astype(np.float64)
    vals = []
    for c in range(a1.shape[2]):
        c1, c2 = a1[:,:,c], a2[:,:,c]
        mu1, mu2 = c1.mean(), c2.mean()
        s1, s2 = ((c1-mu1)**2).mean(), ((c2-mu2)**2).mean()
        s12 = ((c1-mu1)*(c2-mu2)).mean()
        C1, C2 = (0.01*255)**2, (0.03*255)**2
        vals.append(((2*mu1*mu2+C1)*(2*s12+C2))/((mu1**2+mu2**2+C1)*(s1+s2+C2)))
    return np.mean(vals)

def get_font(size):
    for f in ["C:/Windows/Fonts/arial.ttf","arial.ttf","/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]:
        try: return ImageFont.truetype(f, size)
        except: continue
    return ImageFont.load_default()

def load_model(path, device='cuda'):
    model = build_model(width=32).to(device)
    ckpt = torch.load(path, map_location=device, weights_only=True)
    sd = ckpt['model_state_dict'] if isinstance(ckpt, dict) and 'model_state_dict' in ckpt else ckpt
    model.load_state_dict({k.replace("_orig_mod.",""):v for k,v in sd.items()})
    model.eval()
    return model

def run_model(model, pil, device='cuda'):
    x = transforms.ToTensor()(pil).unsqueeze(0).to(device)
    with torch.no_grad():
        with torch.amp.autocast('cuda', dtype=torch.bfloat16): out = model(x)
    out = out.squeeze(0).float().clamp(0,1).cpu()
    r = transforms.ToPILImage()(out)
    return r.resize(pil.size, Image.BICUBIC) if r.size != pil.size else r

def apply_degradation(img_np, deg_type):
    if deg_type == 'blur':
        return cv2.GaussianBlur(img_np, (15,15), 5.0)
    elif deg_type == 'noise':
        return np.clip(img_np + np.random.normal(0, 30/255.0, img_np.shape).astype(np.float32), 0, 1)
    elif deg_type == 'downsample':
        h, w = img_np.shape[:2]
        return cv2.resize(cv2.resize(img_np,(w//4,h//4)),(w,h),interpolation=cv2.INTER_LINEAR)
    elif deg_type == 'jpeg':
        u8 = (img_np*255).astype(np.uint8)
        _,enc = cv2.imencode('.jpg',u8,[cv2.IMWRITE_JPEG_QUALITY,40])
        return cv2.imdecode(enc, cv2.IMREAD_COLOR).astype(np.float32)/255.0
    elif deg_type == 'color':
        d = img_np.copy(); d[:,:,0]*=1.2; d[:,:,2]*=0.8; return np.clip(d,0,1)
    elif deg_type == 'occlusion':
        d = img_np.copy(); h,w = d.shape[:2]; d[h//2:h//2+h//4, w//4:w//4+w//2]=0.15; return d
    elif deg_type == 'brightness':
        return np.clip(0.6*img_np - 0.1, 0, 1)
    elif deg_type == 'combined':
        d = cv2.GaussianBlur(img_np,(11,11),3.0)
        d = np.clip(d + np.random.normal(0,20/255.0,d.shape).astype(np.float32),0,1)
        d[:,:,0]*=1.1; d[:,:,2]*=0.9; d = np.clip(d,0,1)
        u8 = (d*255).astype(np.uint8)
        _,enc = cv2.imencode('.jpg',u8,[cv2.IMWRITE_JPEG_QUALITY,50])
        return cv2.imdecode(enc, cv2.IMREAD_COLOR).astype(np.float32)/255.0
    return img_np


# ============================================================
# Experiment 1: Overall comparison (7 degradations, random mix)
# ============================================================
def experiment_overall(nafnet, flrnet, paths, device):
    print("\n" + "="*60)
    print("Experiment 1: Overall PSNR/SSIM (7 Degradations)")
    print(f"Test images: {len(paths)}")
    print("="*60)

    # 7种退化全开
    deg = BlindDegradation(
        blur_prob=0.7, noise_prob=0.5, downsample_prob=0.4, jpeg_prob=0.6,
        color_prob=0.4, occlusion_prob=0.2, brightness_prob=0.4
    )

    results = []
    for i, path in enumerate(paths):
        try:
            img = Image.open(path).convert('RGB')
            degraded = deg(img)
            r_naf = run_model(nafnet, degraded, device)
            r_flr = run_model(flrnet, degraded, device)
            results.append({
                'image': os.path.basename(path),
                'psnr_degraded': calculate_psnr(img, degraded),
                'psnr_nafnet': calculate_psnr(img, r_naf),
                'psnr_flrnet': calculate_psnr(img, r_flr),
                'ssim_degraded': calculate_ssim(img, degraded),
                'ssim_nafnet': calculate_ssim(img, r_naf),
                'ssim_flrnet': calculate_ssim(img, r_flr),
            })
            if (i+1)%20==0: print(f"  [{i+1}/{len(paths)}] Done")
        except Exception as e: print(f"  Skipped: {e}")

    # Save CSV
    csv_path = os.path.join(OUTPUT_DIR, "overall_7deg.csv")
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=results[0].keys()); w.writeheader(); w.writerows(results)

    avg = lambda k: np.mean([r[k] for r in results if np.isfinite(r[k])])
    print(f"\n+-------------------+-----------+-----------+-----------+")
    print(f"|  Metric           | Degraded  | NAFNet    | FLR-Net   |")
    print(f"+-------------------+-----------+-----------+-----------+")
    print(f"| PSNR (dB)         |  {avg('psnr_degraded'):6.2f}   |  {avg('psnr_nafnet'):6.2f}   |  {avg('psnr_flrnet'):6.2f}   |")
    print(f"| SSIM              |  {avg('ssim_degraded'):6.4f}  |  {avg('ssim_nafnet'):6.4f}  |  {avg('ssim_flrnet'):6.4f}  |")
    print(f"+-------------------+-----------+-----------+-----------+")
    print(f"| FLR-Net vs NAFNet |           |           |  {avg('psnr_flrnet')-avg('psnr_nafnet'):+5.2f}   |")
    print(f"+-------------------+-----------+-----------+-----------+")

    with open(os.path.join(OUTPUT_DIR, "overall_7deg_summary.txt"), 'w') as f:
        f.write(f"Overall 7-Degradation Comparison ({len(results)} images)\n\n")
        f.write(f"PSNR: Degraded={avg('psnr_degraded'):.2f}, NAFNet={avg('psnr_nafnet'):.2f}, FLR-Net={avg('psnr_flrnet'):.2f}\n")
        f.write(f"SSIM: Degraded={avg('ssim_degraded'):.4f}, NAFNet={avg('ssim_nafnet'):.4f}, FLR-Net={avg('ssim_flrnet'):.4f}\n")
        f.write(f"FLR-Net vs NAFNet: {avg('psnr_flrnet')-avg('psnr_nafnet'):+.2f} dB\n")
    return results


# ============================================================
# Experiment 2: Per-degradation comparison (all 8 types)
# ============================================================
def experiment_per_degradation(nafnet, flrnet, paths, device):
    print("\n" + "="*60)
    print("Experiment 2: Per-Degradation Analysis")
    print("="*60)

    deg_types = {
        'blur':'Gaussian Blur', 'noise':'Gaussian Noise',
        'downsample':'Downsample 4x', 'jpeg':'JPEG Q=40',
        'color':'Color Jitter', 'occlusion':'Occlusion',
        'brightness':'Brightness', 'combined':'Combined'
    }

    all_results = {}
    for dk, dn in deg_types.items():
        psnr_deg, psnr_naf, psnr_flr = [], [], []
        print(f"\n  Testing: {dn}")
        for path in paths:
            try:
                img = Image.open(path).convert('RGB')
                img_np = np.array(img).astype(np.float32)/255.0
                deg_np = apply_degradation(img_np, dk)
                degraded = Image.fromarray((deg_np*255).astype(np.uint8))
                if degraded.size != img.size: degraded = degraded.resize(img.size, Image.BICUBIC)
                r_naf = run_model(nafnet, degraded, device)
                r_flr = run_model(flrnet, degraded, device)
                pd = calculate_psnr(img, degraded)
                pn = calculate_psnr(img, r_naf)
                pf = calculate_psnr(img, r_flr)
                if np.isfinite(pd) and np.isfinite(pn) and np.isfinite(pf):
                    psnr_deg.append(pd); psnr_naf.append(pn); psnr_flr.append(pf)
            except: continue
        if psnr_naf:
            ad, an, af = np.mean(psnr_deg), np.mean(psnr_naf), np.mean(psnr_flr)
            all_results[dk] = {'name':dn,'psnr_degraded':ad,'psnr_nafnet':an,'psnr_flrnet':af,
                               'gain_nafnet':an-ad,'gain_flrnet':af-ad,'flr_vs_naf':af-an}
            print(f"    Degraded: {ad:.2f} | NAFNet: {an:.2f} ({an-ad:+.2f}) | FLR-Net: {af:.2f} ({af-ad:+.2f}) | Diff: {af-an:+.2f}")

    csv_path = os.path.join(OUTPUT_DIR, "per_degradation_7deg.csv")
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['Degradation','PSNR Degraded','PSNR NAFNet','PSNR FLR-Net','Gain NAFNet','Gain FLR-Net','FLR vs NAF'])
        for v in all_results.values():
            w.writerow([v['name'],f"{v['psnr_degraded']:.2f}",f"{v['psnr_nafnet']:.2f}",f"{v['psnr_flrnet']:.2f}",
                        f"{v['gain_nafnet']:+.2f}",f"{v['gain_flrnet']:+.2f}",f"{v['flr_vs_naf']:+.2f}"])

    # Bar chart
    generate_chart(all_results)
    return all_results


def generate_chart(results):
    valid = {k:v for k,v in results.items() if np.isfinite(v['psnr_nafnet']) and np.isfinite(v['psnr_flrnet'])}
    if not valid: return

    W, H = 900, 520; ml,mb,mt,mr = 80,80,50,40; cw,ch = W-ml-mr, H-mt-mb
    canvas = Image.new('RGB',(W,H),(255,255,255)); draw = ImageDraw.Draw(canvas)
    ft,fl,fv = get_font(16),get_font(12),get_font(10)

    title = "PSNR: NAFNet vs FLR-Net (7 Degradation Types)"
    bb = draw.textbbox((0,0),title,font=ft)
    draw.text(((W-bb[2]+bb[0])//2,12),title,fill=(30,30,30),font=ft)

    names = [v['name'] for v in valid.values()]
    nv = [v['psnr_nafnet'] for v in valid.values()]
    fv_ = [v['psnr_flrnet'] for v in valid.values()]
    n = len(names); mx = max(max(nv),max(fv_))*1.12; gw = cw/n; bw = gw*0.3

    for i in range(6):
        val = mx*i/5; y = mt+ch-val/mx*ch
        draw.line([(ml,y),(W-mr,y)],fill=(230,230,230))
        draw.text((5,y-7),f"{val:.1f}",fill=(120,120,120),font=fv)

    draw.text((8,mt-18),"PSNR (dB)",fill=(60,60,60),font=fl)

    for i in range(n):
        cx = ml+gw*i+gw/2
        # NAFNet bar (blue)
        h1 = nv[i]/mx*ch; x1 = cx-bw-2; y1 = mt+ch-h1
        draw.rectangle([x1,y1,x1+bw,mt+ch],fill=(100,150,220))
        draw.text((x1,y1-13),f"{nv[i]:.1f}",fill=(60,100,180),font=fv)
        # FLR-Net bar (green)
        h2 = fv_[i]/mx*ch; x2 = cx+2; y2 = mt+ch-h2
        draw.rectangle([x2,y2,x2+bw,mt+ch],fill=(100,200,120))
        draw.text((x2,y2-13),f"{fv_[i]:.1f}",fill=(40,160,60),font=fv)
        # Label
        short = names[i].replace("Gaussian ","G.").replace("Downsample 4x","Down4x").replace("JPEG Q=40","JPEG")
        bb = draw.textbbox((0,0),short,font=fl)
        draw.text((cx-(bb[2]-bb[0])//2,H-mb+8),short,fill=(60,60,60),font=fl)
        # Diff
        diff = fv_[i]-nv[i]; dc = (40,160,60) if diff>=0 else (200,60,60)
        draw.text((cx-15,H-mb+28),f"{diff:+.2f}",fill=dc,font=fv)

    # Legend
    lx = W-mr-180; ly = mt+5
    draw.rectangle([lx,ly,lx+14,ly+14],fill=(100,150,220)); draw.text((lx+20,ly),"NAFNet",fill=(60,60,60),font=fl)
    draw.rectangle([lx,ly+22,lx+14,ly+36],fill=(100,200,120)); draw.text((lx+20,ly+22),"FLR-Net (Ours)",fill=(60,60,60),font=fl)
    draw.text((ml,H-18),"Numbers below: FLR-Net - NAFNet PSNR difference (dB)",fill=(120,120,120),font=fv)

    canvas.save(os.path.join(OUTPUT_DIR,"nafnet_vs_flrnet_7deg_chart.png"),dpi=(300,300))
    print(f"\n  Chart saved: {OUTPUT_DIR}/nafnet_vs_flrnet_7deg_chart.png")


# ============================================================
# Experiment 3: Visual comparison (GT | Degraded | NAFNet | FLR-Net)
# ============================================================
def experiment_visual(nafnet, flrnet, paths, device):
    print("\n" + "="*60)
    print("Experiment 3: Visual Comparison")
    print("="*60)

    deg = BlindDegradation(
        blur_prob=0.7, noise_prob=0.5, downsample_prob=0.4, jpeg_prob=0.6,
        color_prob=0.4, occlusion_prob=0.2, brightness_prob=0.4
    )
    vis_dir = os.path.join(OUTPUT_DIR, "visual_comparison"); os.makedirs(vis_dir, exist_ok=True)
    font_l, font_i = get_font(14), get_font(12)

    for idx, path in enumerate(paths[:10]):
        try:
            img = Image.open(path).convert('RGB'); degraded = deg(img)
            r_naf = run_model(nafnet, degraded, device)
            r_flr = run_model(flrnet, degraded, device)
            pn, pf = calculate_psnr(img,r_naf), calculate_psnr(img,r_flr)
            sn, sf = calculate_ssim(img,r_naf), calculate_ssim(img,r_flr)

            w, h = img.size; gap = 4; lh = 28; ih = 50
            canvas = Image.new('RGB',(w*4+gap*3, h+lh+ih),(30,30,30))
            draw = ImageDraw.Draw(canvas)
            for i,(label,im) in enumerate([("Original (GT)",img),("Degraded",degraded),("NAFNet",r_naf),("FLR-Net (Ours)",r_flr)]):
                x = i*(w+gap); canvas.paste(im,(x,lh))
                bb = draw.textbbox((0,0),label,font=font_l)
                draw.text((x+(w-bb[2]+bb[0])//2,5),label,fill=(220,220,220),font=font_l)
            iy = lh+h+6
            draw.text((12,iy),f"NAFNet: PSNR={pn:.2f} dB, SSIM={sn:.4f}",fill=(100,150,220),font=font_i)
            draw.text((12,iy+18),f"FLR-Net: PSNR={pf:.2f} dB, SSIM={sf:.4f}",fill=(100,220,120),font=font_i)
            dt = f"FLR-Net vs NAFNet: PSNR {pf-pn:+.2f} dB"
            bb = draw.textbbox((0,0),dt,font=font_i)
            draw.text((w*4+gap*3-bb[2]+bb[0]-12,iy+10),dt,fill=(255,220,80),font=font_i)
            canvas.save(os.path.join(vis_dir,f"compare_{idx+1:02d}.png"))
            print(f"  Sample {idx+1}: NAFNet={pn:.2f} FLR-Net={pf:.2f} ({pf-pn:+.2f})")
        except Exception as e: print(f"  Skipped: {e}")
    print(f"\n  Saved in: {vis_dir}/")


# ============================================================
# Main
# ============================================================
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--nafnet", default="checkpoints_7deg/best.pth")
    parser.add_argument("--flrnet", default="checkpoints_v2/best.pth")
    parser.add_argument("--data", default="data/ffhq")
    parser.add_argument("--num_test", type=int, default=200)
    parser.add_argument("--num_deg_test", type=int, default=50)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("="*60)
    print("NAFNet vs FLR-Net (7 Degradation Conditions)")
    print("="*60)

    print(f"\nLoading NAFNet: {args.nafnet}")
    nafnet = load_model(args.nafnet, device)
    print(f"Loading FLR-Net: {args.flrnet}")
    flrnet = load_model(args.flrnet, device)

    all_paths = []
    for ext in ('*.jpg','*.jpeg','*.png','*.bmp'):
        all_paths.extend(glob.glob(os.path.join(args.data,'**',ext),recursive=True))
    all_paths.sort()
    test_pool = all_paths[int(len(all_paths)*0.9):]
    test_paths = random.sample(test_pool, min(args.num_test, len(test_pool)))
    deg_paths = random.sample(test_pool, min(args.num_deg_test, len(test_pool)))

    print(f"Test images: {len(test_paths)}\n")

    experiment_overall(nafnet, flrnet, test_paths, device)
    experiment_per_degradation(nafnet, flrnet, deg_paths, device)
    experiment_visual(nafnet, flrnet, test_paths[:10], device)

    print("\n"+"="*60)
    print("All experiments completed!")
    print(f"Output: {OUTPUT_DIR}/")
    print(f"  - overall_7deg.csv              -> Per-image results")
    print(f"  - overall_7deg_summary.txt      -> Summary")
    print(f"  - per_degradation_7deg.csv      -> 8 degradation types")
    print(f"  - nafnet_vs_flrnet_7deg_chart.png -> Bar chart")
    print(f"  - visual_comparison/            -> Visual comparison images")
    print("="*60)

if __name__ == "__main__":
    main()