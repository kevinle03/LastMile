#!/usr/bin/env python3
import os
import re
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import ConnectionPatch
import numpy as np

# === USER CONFIGURATION ===
root_dir      = '/scratch/ll5484/lillian/GM/dataset/FHDMi/test'   # ← change to your root folder
filter_regex  = r'FHDMi_Base'      # or r'FHDMi_Large'
epoch_filter  = 'epoch90'          # ← change to your epoch filter
# src_04165.png,src_11102.png,src_09610.png,src_08151.png,src_07672.png,src_07309.png,src_07116.png
# '07672', '07309', '07116', '08151'
# '06167', '00881', '04165', '09610'
selected_image_numbers = ['06167', '00881', '04165', '09610']  # <<<< Specify image numbers here as strings
output_dir    = 'comparison_figures'      # ← output directory
os.makedirs(output_dir, exist_ok=True)
output_pdf    = os.path.join(output_dir, f"LastMile_vs_DiffPlugin_figure_{filter_regex}_{epoch_filter}.pdf")
# ===========================
pattern = re.compile(filter_regex)

if 'unknown' in filter_regex:
    degradation = 'unknown'
else:
    degradation = 'bicubic'
if 'Large' in filter_regex:
    model = 'Large'
else:
    model = 'Base'

subdirs = [
    d for d in os.listdir(root_dir)
    if os.path.isdir(os.path.join(root_dir, d)) and (
        (pattern.search(d) and 'step_20' in d and 'exp2' in d
         and (epoch_filter is None or epoch_filter in d)) or
        ('diff-plugin' in d) or ('source' in d) or ('target' in d)
    )
]
if not subdirs:
    raise SystemExit("No subdirectories match your filter. Please check root_dir and filter_regex.")

exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.gif')

# Build mapping: {directory: {image_number: filename}}
images = {}
for d in subdirs:
    dir_path = os.path.join(root_dir, d)
    files = [f for f in os.listdir(dir_path) if f.endswith(exts)]
    num_to_file = {}
    for fname in files:
        match = re.search(r'(\d{5})', fname)
        if match:
            num = match.group(1)
            num_to_file[num] = fname
    images[d] = num_to_file

label_map = {
    "exp2":   "Last Mile",
    "diff-plugin": "Diff-Plugin",
    "source": "Y",
    "target": "X",
}

label_to_dir = {}
for d in subdirs:
    for key, label in label_map.items():
        if key in d and all(num in images[d] for num in selected_image_numbers) and label not in label_to_dir:
            label_to_dir[label] = d

ordered_labels = ["X", "Y", "Last Mile", "Diff-Plugin"]
valid_dirs = [label_to_dir[lbl] for lbl in ordered_labels if lbl in label_to_dir]
if len(valid_dirs) < len(ordered_labels):
    missing = set(ordered_labels) - set(label_to_dir.keys())
    raise SystemExit(f"Missing folders for labels: {missing}")

# === ZOOM CONFIGURATION ===
zoom_size = 256
zoom_interpolation = Image.NEAREST
zoom_positions_by_number = {
    selected_image_numbers[0]: (3*zoom_size, 1*zoom_size),
    selected_image_numbers[1]: (2*zoom_size, 1.5*zoom_size),
    selected_image_numbers[2]: (2*zoom_size, 2*zoom_size),
    selected_image_numbers[3]: (4*zoom_size, 0.5*zoom_size),
}

# Build the figure:
gap_count = len(ordered_labels) - 1
ncols = len(ordered_labels) * 2 + gap_count
nrows = len(selected_image_numbers)

ratio = 1920 / 1080
gap_ratio = 0.05
width_ratios = []
for method_i in range(len(ordered_labels)):
    width_ratios.append(ratio)  # full‐image column
    width_ratios.append(1)      # zoom column
    if method_i < len(ordered_labels) - 1:
        width_ratios.append(gap_ratio)

fig, axes = plt.subplots(
    nrows=nrows,
    ncols=ncols,
    figsize=(2.82 * ncols, 3 * nrows),
    squeeze=False,
    gridspec_kw={'width_ratios': width_ratios}
)
fig.subplots_adjust(hspace=0, wspace=0)

for row_i, img_num in enumerate(selected_image_numbers):
    for method_i, label in enumerate(ordered_labels):
        d = label_to_dir[label]
        img_dict = images[d]
        if img_num not in img_dict:
            raise ValueError(f"Image number {img_num} not found in directory {d}")
        fname = img_dict[img_num]
        print(f"[{label}] using file: {fname}")
        path = os.path.join(root_dir, d, fname)
        img = Image.open(path).convert("RGB")
        img_np = np.array(img)

        if img_num not in zoom_positions_by_number:
            raise ValueError(f"No zoom coordinates defined for image number {img_num}")
        zoom_x, zoom_y = zoom_positions_by_number[img_num]

        base_col = method_i * 3

        # 1) full‐image sub‐column
        col_full = base_col
        ax_full = axes[row_i, col_full]
        ax_full.imshow(img_np)
        ax_full.set_xticks([])
        ax_full.set_yticks([])
        for spine in ax_full.spines.values():
            spine.set_visible(False)

        # 2) draw rectangle on full image
        rect = patches.Rectangle((zoom_x, zoom_y),
                                 zoom_size, zoom_size,
                                 linewidth=1,
                                 edgecolor='yellow',
                                 facecolor='none')
        ax_full.add_patch(rect)

        # 3) zoomed‐in sub‐column
        col_zoom = base_col + 1
        ax_zoom = axes[row_i, col_zoom]
        zoom_crop = img.crop((zoom_x, zoom_y,
                              zoom_x + zoom_size, zoom_y + zoom_size))
        ax_zoom.imshow(zoom_crop, interpolation='none')
        ax_zoom.set_xticks([])
        ax_zoom.set_yticks([])
        for spine in ax_zoom.spines.values():
            spine.set_visible(False)

        # 4) draw connection lines
        con_top = ConnectionPatch(
            xyA=(zoom_x + zoom_size, zoom_y),
            xyB=(0, 0),
            coordsA="data",
            coordsB="data",
            axesA=ax_full,
            axesB=ax_zoom,
            color="yellow",
            linewidth=1,
        )
        con_bottom = ConnectionPatch(
            xyA=(zoom_x + zoom_size, zoom_y + zoom_size),
            xyB=(0, zoom_size),
            coordsA="data",
            coordsB="data",
            axesA=ax_full,
            axesB=ax_zoom,
            color="yellow",
            linewidth=1,
        )
        ax_full.add_artist(con_top)
        ax_full.add_artist(con_bottom)

        # 5) (no title on top)

    # 6) turn off gap columns
    for gap_i in range(len(ordered_labels) - 1):
        gap_col_index = gap_i * 3 + 2
        ax_gap = axes[row_i, gap_col_index]
        ax_gap.axis('off')

# ─── Add labels at the very bottom ───
for method_i, label in enumerate(ordered_labels):
    base_col = method_i * 3
    ax_bot = axes[nrows - 1, base_col]
    ax_bot.set_xlabel(label, fontsize=35, labelpad=12)

# 7) Save and show
plt.tight_layout(
    pad=0,
    h_pad=0,
    w_pad=0
)
fig.savefig(output_pdf, dpi=400, bbox_inches='tight')
print(f"Saved figure to {output_pdf}")
plt.show()
