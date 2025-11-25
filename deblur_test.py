from richardsonlucy import richardson_lucy
from wiener import wiener_deconv
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from pillow_heif import register_heif_opener
import torch

register_heif_opener()

image_files = [
    "data/blur/blur1.HEIC",
    "data/blur/blur2.HEIC",
    "data/blur/blur3.HEIC"
]

psf_np = np.ones((5,5), np.float32) / 25
psf_torch = torch.tensor(psf_np).unsqueeze(0).unsqueeze(0) # (1,1,5,5)

fig, axes = plt.subplots(len(image_files), 3, figsize=(15, 15))

cols = ["Original", "Richardson-Lucy", "Wiener"]
for ax, col in zip(axes[0], cols):
    ax.set_title(col, fontsize=14, fontweight='bold')

for i, filename in enumerate(image_files):
    print(f"Processing {filename}...")
    
    try:
        img_pil = Image.open(filename)
        img = np.array(img_pil.convert('RGB'))
        img = cv2.resize(img, (299, 299))
        img_f = img.astype(np.float32) / 255.0
    except Exception as e:
        print(f"Could not load {filename}: {e}")
        continue

    obs = torch.tensor(img_f).permute(2,0,1).unsqueeze(0)
    x0 = torch.ones_like(obs) * img_f.mean()

    out_channels = []
    for c in range(3):
        obs_c = obs[:, c:c+1, :, :]  
        x0_c = x0[:, c:c+1, :, :]  

        out_c = richardson_lucy(
            observation=obs_c,
            x_0=x0_c,
            k=psf_torch,
            steps=10
        )
        out_channels.append(out_c)

    out_rl = torch.cat(out_channels, dim=1)
    out_rl = out_rl.squeeze(0).permute(1,2,0).numpy()

    out_wiener = wiener_deconv(img_f)

    axes[i, 0].imshow(img_f)
    axes[i, 0].axis('off')
    if i == 0: axes[i, 0].set_title("Original")
    
    axes[i, 1].imshow(out_rl)
    axes[i, 1].axis('off')
    if i == 0: axes[i, 1].set_title("Richardson-Lucy")

    axes[i, 2].imshow(out_wiener)
    axes[i, 2].axis('off')
    if i == 0: axes[i, 2].set_title("Wiener")

plt.tight_layout()
plt.show()