#!/usr/bin/env python3
import os
import re
import random
from PIL import Image
import matplotlib.pyplot as plt

def crop(img, crop_size=256):
    left   = crop_size * 3
    top    = crop_size * 2
    right  = left + crop_size
    bottom = top + crop_size
    return img.crop((left, top, right, bottom))

# === USER CONFIGURATION ===
root_dir      = '/scratch/ll5484/lillian/GM/dataset/FHDMi/test'   # ← change to your root folder
# filter_regex  = r'FHDMi_Base_exp2_epoch50'   #DIV2K_unknown_Large_exp2_epoch500, DIV2K_unknown_Base_exp2_epoch650, DIV2K_bicubic_Large_exp2_epoch600           # ← change to your directory‐name regex
filter_regex  = r'FHDMi_Base_exp2_epoch90'
#filter_regex  = r'FHDMi_Large_exp2_epoch50'
#filter_regex  = r'FHDMi_Large_exp2_epoch90'
num_images    = 4                                                # ← how many images to pick per folder
output_dir    = 'step_figures'                                         # ← output directory
os.makedirs(output_dir, exist_ok=True)                                      # ← create directory if it doesn't exist
output_pdf    = os.path.join(output_dir, f"{filter_regex}_step_figure.pdf")  # ← output filename
# ===========================

# compile the regex
pattern = re.compile(filter_regex)

# 1) find & sort matching subdirectories
subdirs = sorted(
    (d for d in os.listdir(root_dir)
    if os.path.isdir(os.path.join(root_dir, d)) and pattern.search(d)),
    key=lambda x: int(x.rsplit('_', 1)[-1]) if x.rsplit('_', 1)[-1].isdigit() else 0
)
if not subdirs:
    raise SystemExit("No subdirectories match your filter.")

# 2) gather & sort image filenames in each folder
exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.gif')
images = {}
for d in subdirs:
    files = [
        f for f in os.listdir(os.path.join(root_dir, d))
        if f.lower().endswith(exts)
    ]
    files.sort()
    images[d] = files

# 3) keep only folders with at least num_images
valid = [d for d in subdirs if len(images[d]) >= num_images]
if not valid:
    raise SystemExit(f"No folder has ≥ {num_images} images.")

# 4) choose random indices from the common range
min_count = min(len(images[d]) for d in valid)
indices   = sorted(random.sample(range(min_count), num_images))

# 5) build & save the grid
nrows, ncols = num_images, len(valid)
fig, axes   = plt.subplots(nrows, ncols,
                            figsize=(3 * ncols, 3 * nrows),
                            squeeze=False)

# tighten vertical space
fig.subplots_adjust(hspace=0.05, wspace=0.05)  # ↓ reduce this value for even tighter stacking

for col, d in enumerate(valid):
    # extract only "step" and everything after from the folder name
    print(f'label={d}')
    idx = d.find("step")
    if idx != -1:
        label = d[idx:]
    else:
        label = d

    for row, img_idx in enumerate(indices):
        path = os.path.join(root_dir, d, images[d][img_idx])
        img  = Image.open(path)
        img = crop(img)
        ax   = axes[row][col]
        ax.imshow(img)

        # hide ticks and spines but keep labels enabled
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

        # only on the bottom row, put the label underneath
        if row == num_images - 1:
            ax.xaxis.set_label_position('bottom')
            ax.set_xlabel(label, fontsize=25, labelpad=15)

fig.savefig(output_pdf, format='pdf', dpi=300)
print(f"Saved {num_images} images from {len(valid)} folders to {output_pdf}")