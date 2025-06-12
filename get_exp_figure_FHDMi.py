#!/usr/bin/env python3
import os
import re
import random
from PIL import Image
import matplotlib.pyplot as plt

def crop(img, crop_size=256): # not exactly center crop
    left   = crop_size * 3 # 4 is okay too
    top    = crop_size * 2 # 3 is okay too  ; also edit the code for steps_figure
    right  = left + crop_size
    bottom = top + crop_size
    return img.crop((left, top, right, bottom))

#starting crop has to be multiples of 256

# === USER CONFIGURATION ===
root_dir      = '/scratch/ll5484/lillian/GM/dataset/FHDMi/test'   # ← change to your root folder
filter_regex  = r'FHDMi_Base'     #DIV2K_unknown_Large DIV2K_unknown_Base     # ← change to your directory‐name regex
#filter_regex  = r'FHDMi_Large'

epoch_filter = 'epoch90' 
#epoch_filter = 'epoch50' 

num_images    = 4                                                # ← how many images to pick per folder
output_dir    = 'exp_figures'                                         # ← output directory
os.makedirs(output_dir, exist_ok=True)                                      # ← create directory if it doesn't exist
output_pdf    = os.path.join(output_dir, f"{filter_regex}_{epoch_filter}_exp_figure_diff_plugin.pdf") # ← output filename
# ===========================

# compile the regex
pattern = re.compile(filter_regex)

# 1) find & sort matching subdirectories
if 'unknown' in filter_regex:
    degradation = 'unknown'
else:
    degradation = 'bicubic'
if 'Large' in filter_regex:
    model = 'Large'
else:
    model = 'Base'

# subdirs = sorted(
#     (d for d in os.listdir(root_dir)
#     if os.path.isdir(os.path.join(root_dir, d)) and ((pattern.search(d) and 'step_20' in d) or 'HR' in d or ('LR' in d and degradation in d) or (model in d and degradation in d and 'output' in d)))
# )
subdirs = sorted(
    (d for d in os.listdir(root_dir)
     if os.path.isdir(os.path.join(root_dir, d)) and (
        ('target' in d) or 
        ('source' in d ) or 
        (model in d and 'output' in d) or 
        (pattern.search(d) and 'step_20' in d and (epoch_filter is None or epoch_filter in d))
     ))
)
if not subdirs:
    raise SystemExit("No subdirectories match your filter.")

def sort_key(subdir):
    if 'target' in subdir:
        return (0, subdir)
    elif 'source' in subdir:
        return (1, subdir)
    elif 'output' in subdir:
        return (2, subdir)
    elif 'exp' in subdir:
        match = re.search(r'exp(\d)', subdir)
        if match:
            return (3, int(match.group(1)))
    return (4, subdir)

subdirs.sort(key=sort_key)

# 2) gather & sort image filenames in each folder
min_dir = subdirs[0]
for dir in subdirs:
    if len(os.listdir(os.path.join(root_dir,dir))) < len(os.listdir(os.path.join(root_dir,min_dir))):
        min_dir = dir
    
image_number_list = [file_name.split('_')[1].split('.')[0] for file_name in os.listdir(os.path.join(root_dir,min_dir))]

images = {}
for d in subdirs:
    files = sorted(os.listdir(os.path.join(root_dir, d)))
    images[d] = [f for f in files if f.split('_')[1].split('.')[0] in image_number_list]

# 3) keep only one folder per target label if it has ≥ num_images
label_map = {
    "target": "X",
    "source": "Y",
    "output": "Z",
    "exp1": "Z2X|Y",
    "exp2": "Z2X|NC\n(Our)",
    "exp3": "N2X|Y",
    "exp4": "Y2X|Z",
    "exp5": "N2X|Z",
}

label_to_dir = {}
for d in subdirs:
    for key, label in label_map.items():
        if key in d and len(images[d]) >= num_images and label not in label_to_dir:
            label_to_dir[label] = d

ordered_labels = ["X", "Y", "Z", "Z2X|Y", "Z2X|NC\n(Our)", "N2X|Y", "Y2X|Z", "N2X|Z"]
valid = [label_to_dir[label] for label in ordered_labels if label in label_to_dir]

if len(valid) < len(ordered_labels):
    missing = set(ordered_labels) - set(label_to_dir.keys())
    raise SystemExit(f"Missing folders for labels: {missing}")

# 4) choose random indices from the common range
min_count = min(len(images[d]) for d in valid)
indices   = sorted(random.sample(range(min_count), num_images))

# number_list = [
#     469, 2482, 2967, 3438, 3706, 4165, 4266, 4490,
#     5020, 5049, 5256, 6167, 6523, 7672, 8489, 9534,
#     9625, 10465, 10566, 10712, 11102, 11129, 11153,
#     11421, 11604, 12197, 12228, 12235
# ]
# vals = [1, 18, 20, 21, 22, 27, 28, 32,
#     37, 38, 39, 44, 46, 61, 65,
#     67, 74, 75, 76, 83, 84, 85,
#     87, 90, 96, 98, 99] 
# vals = [
#     0, 1, 2, 3, 4, 5, 6, 7, 8, 
#     16, 17, 18, 19,
#     20, 21, 22, 23, 24, 25, 26, 28, 29,
#     30, 31, 33, 34, 35, 36, 38, 
#     40, 41, 46, 47, 49,
#     50, 51, 53, 55, 56, 58, 59,
#     66, 67, 69,
#     70,  72,  74, 75, 77, 78, 79,
#     81, 83, 84, 85, 86, 87, 88, 89,
#     90, 91, 92, 93, 94, 95, 96, 97, 98, 99
# ]
# indices   = sorted(random.sample(vals, num_images))

# indices[0]= 39 #56
# indices[1] = 37 #67
# indices[2] = 83
# indices[3] = 71 #84
# print(indices)

# 5) build & save the grid
nrows, ncols = num_images, len(valid)
fig, axes   = plt.subplots(nrows, ncols,
                            figsize=(3 * ncols, 3 * nrows),
                            squeeze=False)

fig.subplots_adjust(hspace=0.05, wspace=0.05)

for col, label in enumerate(ordered_labels):
    d = label_to_dir[label]
    print(f'label={d}')
    new_label = label  # already formatted in label_map

    for row, img_idx in enumerate(indices):
        fname = images[d][img_idx]
        print(f"name: {fname}")
        path = os.path.join(root_dir, d, images[d][img_idx])
        img  = Image.open(path)
        img = crop(img)
        ax   = axes[row][col]
        ax.imshow(img)

        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

        if row == num_images - 1:
            ax.xaxis.set_label_position('bottom')
            ax.set_xlabel(new_label, fontsize=25, labelpad=15)

fig.savefig(output_pdf, format='pdf', dpi=300)
print(f"Saved {num_images} images from {len(valid)} folders to {output_pdf}")
