# losses/losses.py
# ============================================================
# 组合损失函数 (v2)
# ============================================================
# L1 Loss         → 像素级准确性
# Perceptual Loss → 语义感知相似度
# FFT Loss        → 频域高频细节
# Landmark Loss   → 人脸结构约束（新增，基于 MediaPipe FaceMesh）
# ============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("[警告] mediapipe 未安装，Landmark Loss 不可用")


# ============================================================
# Perceptual Loss
# ============================================================
class PerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features
        self.slice = nn.Sequential(*list(vgg.children())[:16]).eval()
        for p in self.slice.parameters():
            p.requires_grad = False
        self.register_buffer('mean',
                             torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std',
                             torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, pred, target):
        pred_n = (pred - self.mean) / self.std
        target_n = (target - self.mean) / self.std
        return F.l1_loss(self.slice(pred_n), self.slice(target_n))


# ============================================================
# FFT Loss
# ============================================================
class FFTLoss(nn.Module):
    def forward(self, pred, target):
        pred_f = torch.fft.fft2(pred.float(), norm='ortho')
        target_f = torch.fft.fft2(target.float(), norm='ortho')
        return F.l1_loss(torch.abs(pred_f), torch.abs(target_f))


# ============================================================
# Facial Landmark Loss (NEW)
# ============================================================
class LandmarkLoss(nn.Module):
    """
    人脸关键点损失。
    用 MediaPipe FaceMesh 提取预测图和目标图的 468 个人脸关键点，
    计算关键点坐标的 L2 距离作为损失。

    这强制模型精确恢复人脸几何结构（眼睛、嘴巴、鼻子的位置和形状）。

    注意：
    - MediaPipe 在 CPU 上运行，每个 batch 需要先转到 CPU 处理
    - 如果某张图检测不到人脸，这张图的 landmark loss 为 0
    """

    def __init__(self):
        super().__init__()
        if not MEDIAPIPE_AVAILABLE:
            self.face_mesh = None
            return

        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,      # 每张图独立检测
            max_num_faces=1,             # 只检测一个人脸
            refine_landmarks=False,      # 不做细化，加速
            min_detection_confidence=0.3  # 低一点，尽量都能检测到
        )

    def extract_landmarks(self, img_tensor):
        """
        从一个 batch 的 tensor 提取 landmarks
        Args:
            img_tensor: [B, 3, H, W], 范围 [0, 1]
        Returns:
            landmarks: [B, 468, 2]，归一化坐标，未检测到的样本为 None
            mask: [B]，True 表示检测到人脸
        """
        B, C, H, W = img_tensor.shape
        # 转到 CPU numpy uint8
        imgs_np = (img_tensor.detach().cpu().float().clamp(0, 1)
                   .permute(0, 2, 3, 1).numpy() * 255).astype(np.uint8)

        landmarks_batch = []
        mask = []

        for i in range(B):
            results = self.face_mesh.process(imgs_np[i])
            if results.multi_face_landmarks:
                lms = results.multi_face_landmarks[0].landmark
                coords = np.array([[lm.x, lm.y] for lm in lms],
                                  dtype=np.float32)  # [468, 2]
                landmarks_batch.append(coords)
                mask.append(True)
            else:
                landmarks_batch.append(np.zeros((468, 2), dtype=np.float32))
                mask.append(False)

        landmarks_tensor = torch.from_numpy(np.stack(landmarks_batch))
        mask_tensor = torch.tensor(mask, dtype=torch.bool)
        return landmarks_tensor, mask_tensor

    def forward(self, pred, target):
        if self.face_mesh is None:
            return torch.tensor(0.0, device=pred.device)

        pred_lm, pred_mask = self.extract_landmarks(pred)
        target_lm, target_mask = self.extract_landmarks(target)

        # 只在两张图都检测到人脸时计算损失
        valid = pred_mask & target_mask

        if valid.sum() == 0:
            return torch.tensor(0.0, device=pred.device)

        pred_lm = pred_lm.to(pred.device)
        target_lm = target_lm.to(pred.device)
        valid = valid.to(pred.device)

        # 只保留检测成功的样本
        diff = (pred_lm - target_lm) ** 2  # [B, 468, 2]
        loss_per_sample = diff.mean(dim=(1, 2))  # [B]
        loss = loss_per_sample[valid].mean()

        return loss


# ============================================================
# Combined Loss (v2)
# ============================================================
class CombinedLoss(nn.Module):
    """
    组合损失 = w1*L1 + w2*Perc + w3*FFT + w4*Landmark

    权重：
        L1       = 1.0
        Perc     = 0.1
        FFT      = 0.05
        Landmark = 0.01  （新增，较小因为关键点检测本身有误差）
    """

    def __init__(self, w_l1=1.0, w_perc=0.1, w_fft=0.05, w_lm=0.01,
                 use_landmark=True):
        super().__init__()
        self.l1 = nn.L1Loss()
        self.perc = PerceptualLoss()
        self.fft = FFTLoss()
        self.landmark = LandmarkLoss() if use_landmark else None

        self.w_l1 = w_l1
        self.w_perc = w_perc
        self.w_fft = w_fft
        self.w_lm = w_lm
        self.use_landmark = use_landmark

    def forward(self, pred, target):
        loss_l1 = self.l1(pred, target)
        loss_perc = self.perc(pred, target)
        loss_fft = self.fft(pred, target)

        total = (self.w_l1 * loss_l1 +
                 self.w_perc * loss_perc +
                 self.w_fft * loss_fft)

        loss_dict = {
            "l1": loss_l1.item(),
            "perc": loss_perc.item(),
            "fft": loss_fft.item(),
        }

        # Landmark loss（每 5 个 step 算一次，因为 mediapipe 较慢）
        if self.use_landmark and self.landmark is not None:
            loss_lm = self.landmark(pred, target)
            total = total + self.w_lm * loss_lm
            loss_dict["landmark"] = loss_lm.item() if torch.is_tensor(loss_lm) else loss_lm
        else:
            loss_dict["landmark"] = 0.0

        return total, loss_dict


if __name__ == "__main__":
    criterion = CombinedLoss().cuda()

    pred = torch.rand(2, 3, 256, 256).cuda().requires_grad_(True)
    target = torch.rand(2, 3, 256, 256).cuda()

    loss, loss_dict = criterion(pred, target)
    print(f"Total Loss: {loss.item():.4f}")
    print(f"  L1:        {loss_dict['l1']:.4f}")
    print(f"  Perc:      {loss_dict['perc']:.4f}")
    print(f"  FFT:       {loss_dict['fft']:.4f}")
    print(f"  Landmark:  {loss_dict['landmark']:.4f}")

    loss.backward()
    print("\n✅ Combined Loss v2 测试通过")