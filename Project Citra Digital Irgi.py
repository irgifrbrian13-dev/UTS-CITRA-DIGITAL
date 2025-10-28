import matplotlib.pyplot as plt
import numpy as np
from skimage import img_as_float, img_as_ubyte
from skimage.util import random_noise
from skimage.filters import rank
from skimage.morphology import disk
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
import imageio
from skimage.color import rgb2gray
import os

# ====== GANTI SESUAI NAMA FILE GAMBARMU ======
image_path = r'D:\Tugas Citra Digital\Skyline R34.jpg'

# Cek apakah file ada
if not os.path.exists(image_path):
    raise FileNotFoundError(f"File tidak ditemukan: {image_path}")

# Membaca dan ubah ke grayscale float
image = img_as_float(rgb2gray(imageio.imread(image_path)))

# Tambahkan noise salt & pepper
noisy = random_noise(image, mode='s&p', amount=0.1)

# Rank filters butuh 8-bit
noisy_ubyte = img_as_ubyte(noisy)

# Terapkan berbagai filter
mean_filtered = rank.mean(noisy_ubyte, footprint=disk(3))
min_filtered = rank.minimum(noisy_ubyte, footprint=disk(3))
median_filtered = rank.median(noisy_ubyte, footprint=disk(3))
max_filtered = rank.maximum(noisy_ubyte, footprint=disk(3))

# Ubah kembali ke float [0,1]
mean_filtered_float = img_as_float(mean_filtered)
min_filtered_float = img_as_float(min_filtered)
median_filtered_float = img_as_float(median_filtered)
max_filtered_float = img_as_float(max_filtered)

# Hitung metrik
metrics = {
   "Noisy": {
       "PSNR": psnr(image, noisy),
       "SSIM": ssim(image, noisy, data_range=1.0)
   },
   "Mean Filtered": {
       "PSNR": psnr(image, mean_filtered_float),
       "SSIM": ssim(image, mean_filtered_float, data_range=1.0)
   },
   "Min Filtered": {
       "PSNR": psnr(image, min_filtered_float),
       "SSIM": ssim(image, min_filtered_float, data_range=1.0)
   },
   "Median Filtered": {
       "PSNR": psnr(image, median_filtered_float),
       "SSIM": ssim(image, median_filtered_float, data_range=1.0)
   },
   "Max Filtered": {
       "PSNR": psnr(image, max_filtered_float),
       "SSIM": ssim(image, max_filtered_float, data_range=1.0)
   }
}

# Tampilkan hasil
fig, axes = plt.subplots(1, 6, figsize=(15, 5))
ax = axes.ravel()

titles = ["Original", "Noisy", "Mean Filter", "Min Filter", "Median Filter", "Max Filter"]
images = [image, noisy, mean_filtered, min_filtered, median_filtered, max_filtered]

for i in range(6):
    ax[i].imshow(images[i], cmap='gray')
    ax[i].set_title(titles[i])
    ax[i].axis('off')

plt.tight_layout()
plt.show()

# Cetak nilai PSNR dan SSIM
for name, vals in metrics.items():
    print(f"{name} -> PSNR: {vals['PSNR']:.2f}, SSIM: {vals['SSIM']:.4f}")
