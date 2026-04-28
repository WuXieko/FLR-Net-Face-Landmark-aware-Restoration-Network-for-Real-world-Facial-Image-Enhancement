# datapipe/degradation.py
# ============================================================
# 人脸感知盲退化 Pipeline (v2)
# ============================================================
# 在原 4 种通用退化基础上，新增 3 种人脸场景特有退化：
#   5. 色彩偏移（Color Jitter）       - 模拟冷/暖/荧光光源
#   6. 局部遮挡（Random Occlusion）    - 模拟口罩、墨镜、刘海
#   7. 亮度对比度扰动（Brightness/Contrast）- 模拟逆光、闪光灯
# ============================================================

import cv2
import random
import numpy as np
from PIL import Image


class BlindDegradation:
    """
    人脸感知盲退化生成器 (v2)。

    通用退化（原有）：
        1. 高斯模糊   → 失焦、运动模糊
        2. 高斯噪声   → 高 ISO、老照片颗粒
        3. 下采样     → 低分辨率
        4. JPEG 压缩  → 社交平台压缩

    人脸特有退化（新增）：
        5. 色彩偏移   → 不同光源色温变化
        6. 局部遮挡   → 口罩、墨镜、刘海遮挡
        7. 亮度对比度 → 逆光、闪光灯、暗光
    """

    def __init__(self,
                 blur_prob=0.7,
                 noise_prob=0.5,
                 downsample_prob=0.4,
                 jpeg_prob=0.6,
                 color_prob=0.4,
                 occlusion_prob=0.2,
                 brightness_prob=0.4):
        self.blur_prob = blur_prob
        self.noise_prob = noise_prob
        self.downsample_prob = downsample_prob
        self.jpeg_prob = jpeg_prob
        self.color_prob = color_prob
        self.occlusion_prob = occlusion_prob
        self.brightness_prob = brightness_prob

    def __call__(self, img_pil):
        img = np.array(img_pil).astype(np.float32) / 255.0
        original = img.copy()

        # ---------- 1. 高斯模糊 ----------
        if random.random() < self.blur_prob:
            k = random.choice([7, 9, 11, 13, 15, 17, 19, 21])
            sigma = random.uniform(0.5, 8.0)
            img = cv2.GaussianBlur(img, (k, k), sigma)

        # ---------- 2. 高斯噪声 ----------
        if random.random() < self.noise_prob:
            noise_level = random.uniform(1, 50) / 255.0
            noise = np.random.normal(0, noise_level, img.shape).astype(np.float32)
            img = np.clip(img + noise, 0, 1)

        # ---------- 3. 下采样+上采样 ----------
        if random.random() < self.downsample_prob:
            h, w = img.shape[:2]
            scale = random.uniform(0.3, 0.9)
            small = cv2.resize(img, (int(w * scale), int(h * scale)),
                               interpolation=cv2.INTER_AREA)
            img = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)

        # ---------- 4. JPEG 压缩 ----------
        if random.random() < self.jpeg_prob:
            quality = random.randint(30, 90)
            img_uint8 = (img * 255).astype(np.uint8)
            _, enc = cv2.imencode('.jpg', img_uint8,
                                  [cv2.IMWRITE_JPEG_QUALITY, quality])
            img = cv2.imdecode(enc, cv2.IMREAD_COLOR).astype(np.float32) / 255.0

        # ---------- 5. 色彩偏移（新增）----------
        if random.random() < self.color_prob:
            r_gain = random.uniform(0.75, 1.25)
            g_gain = random.uniform(0.75, 1.25)
            b_gain = random.uniform(0.75, 1.25)
            img[:, :, 0] *= r_gain
            img[:, :, 1] *= g_gain
            img[:, :, 2] *= b_gain
            img = np.clip(img, 0, 1)

        # ---------- 6. 局部遮挡（新增）----------
        if random.random() < self.occlusion_prob:
            h, w = img.shape[:2]
            bh = random.randint(max(h // 8, 4), max(h // 3, 5))
            bw = random.randint(max(w // 4, 4), max(w // 2, 5))
            if random.random() < 0.6:
                y = random.randint(h // 2, max(h - bh, h // 2 + 1))
            else:
                y = random.randint(0, max(h - bh, 1))
            x = random.randint(0, max(w - bw, 1))
            color = random.uniform(0.05, 0.9)
            img[y:y+bh, x:x+bw] = color

        # ---------- 7. 亮度/对比度（新增）----------
        if random.random() < self.brightness_prob:
            alpha = random.uniform(0.6, 1.4)
            beta = random.uniform(-0.2, 0.2)
            img = np.clip(alpha * img + beta, 0, 1)

        # ---------- 兜底 ----------
        if np.array_equal(img, original):
            img = cv2.GaussianBlur(img, (7, 7), 1.5)

        return Image.fromarray((img * 255).astype(np.uint8))


if __name__ == "__main__":
    import os

    test_img_path = r"C:\Users\76161\Desktop\my_facerest\data\ffhq\00000.png"
    for d in ["data/ffhq", "WIDER_train"]:
        if os.path.exists(d):
            for root, _, files in os.walk(d):
                for f in files:
                    if f.endswith(('.png', '.jpg')):
                        test_img_path = os.path.join(root, f)
                        break
                if test_img_path:
                    break
            if test_img_path:
                break

    if test_img_path is None:
        print("未找到测试图片")
        exit()

    print(f"测试图片: {test_img_path}")
    img = Image.open(test_img_path).convert('RGB').resize((256, 256))

    deg = BlindDegradation()
    os.makedirs("degradation_test", exist_ok=True)
    img.save("degradation_test/original.png")
    for i in range(8):
        degraded = deg(img)
        degraded.save(f"degradation_test/sample_{i}.png")
    print("✅ 生成 8 张退化示例到 degradation_test/ 目录")