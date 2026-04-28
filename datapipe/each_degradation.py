# test_each_degradation.py
import sys, os, cv2, numpy as np
sys.path.insert(0, '.')
from PIL import Image

img = Image.open(r"C:\Users\76161\Desktop\my_facerest\data\ffhq\00000.png").convert('RGB').resize((256,256))
img_np = np.array(img).astype(np.float32) / 255.0
os.makedirs("degradation_test", exist_ok=True)
img.save("degradation_test/0_original.png")

# 1. 高斯模糊
d = cv2.GaussianBlur(img_np, (15,15), 5.0)
Image.fromarray((d*255).astype(np.uint8)).save("degradation_test/1_blur.png")

# 2. 高斯噪声
d = np.clip(img_np + np.random.normal(0, 35/255, img_np.shape).astype(np.float32), 0, 1)
Image.fromarray((d*255).astype(np.uint8)).save("degradation_test/2_noise.png")

# 3. 下采样
s = cv2.resize(img_np, (64,64))
d = cv2.resize(s, (256,256), interpolation=cv2.INTER_LINEAR)
Image.fromarray((d*255).astype(np.uint8)).save("degradation_test/3_downsample.png")

# 4. JPEG压缩
u8 = (img_np*255).astype(np.uint8)
_, enc = cv2.imencode('.jpg', u8, [cv2.IMWRITE_JPEG_QUALITY, 10])
d = cv2.imdecode(enc, cv2.IMREAD_COLOR).astype(np.float32)/255
Image.fromarray((d*255).astype(np.uint8)).save("degradation_test/4_jpeg.png")

# 5. 色彩偏移（新增）
d = img_np.copy()
d[:,:,0] *= 1.25; d[:,:,2] *= 0.75
d = np.clip(d, 0, 1)
Image.fromarray((d*255).astype(np.uint8)).save("degradation_test/5_color.png")

# 6. 局部遮挡（新增）
d = img_np.copy()
d[140:220, 40:210] = 0.15
Image.fromarray((d*255).astype(np.uint8)).save("degradation_test/6_occlusion.png")

# 7. 亮度对比度（新增）
d = np.clip(0.6 * img_np - 0.1, 0, 1)
Image.fromarray((d*255).astype(np.uint8)).save("degradation_test/7_brightness.png")

print("✅ 7 种退化各生成 1 张，保存在 degradation_test/")