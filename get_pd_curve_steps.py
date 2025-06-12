import os
import re
import pandas as pd
import matplotlib.pyplot as plt
from adjustText import adjust_text
import numpy as np
from scipy.interpolate import make_interp_spline
from scipy.interpolate import PchipInterpolator
# ───────── USER CONFIGURATION ─────────
csv_file      = './FHDMi_Large_metrics/mean_metrics.csv'
filter_string = 'FHDMi_Large'  # 'FHDMi_Base', 'FHDMi_Large', 'DIV2K_bicubic_Large', 'DIV2K_bicubic_Base'
epoch         = 90
x_column      = 'FID'
y_column      = 'SSIM'  # we'll use 1 - SSIM
y_transform   = lambda ssim: 1 - ssim
output_dir    = 'step_pd_figures'
# ────────────────────────────────────────────────────

# Load CSV
df = pd.read_csv(csv_file)

# Filter Folder names
if filter_string is not None:
    def keep_folder(d):
        return (filter_string in d and f'epoch{epoch}' in d and 'exp2' in d)

    df = df[df['Folder'].apply(keep_folder)].sort_values('Folder')

# Prepare data
x = df[x_column]  # FID
y = y_transform(df[y_column])  # 1 - SSIM
labels = df['Folder']
xlabel = f"{x_column} (↓ better)"
ylabel = f"1 - {y_column} (↓ better)"

# Start plot
plt.figure(figsize=(10, 8))
plt.scatter(x, y, color='red', marker='o', s=100)

# sorted_indices = np.argsort(x)
# x_sorted = np.array(x)[sorted_indices]
# y_sorted = np.array(y)[sorted_indices]


# x_smooth = np.linspace(x_sorted.min(), x_sorted.max(), 300)
# spline = make_interp_spline(x_sorted, y_sorted, k=3)  # Cubic spline
# y_smooth = spline(x_smooth)


# plt.plot(x_smooth, y_smooth, color='red', linestyle='-', linewidth=2, zorder=1)
# -----------------------------------------------------------
# sorted_indices = np.argsort(x)
# x_sorted = np.array(x)[sorted_indices]
# y_sorted = np.array(y)[sorted_indices]

# x_smooth = np.linspace(x_sorted.min(), x_sorted.max(), 300)
# pchip = PchipInterpolator(x_sorted, y_sorted)
# y_smooth = pchip(x_smooth)

# plt.plot(x_smooth, y_smooth, color='red', linestyle='-', linewidth=2, zorder=1)
# ----------------------------------------------------------
sorted_indices = np.argsort(x)
x_sorted = np.array(x)[sorted_indices]
y_sorted = np.array(y)[sorted_indices]
coeffs = np.polyfit(x_sorted, y_sorted, deg=2)
poly = np.poly1d(coeffs)

x_smooth = np.linspace(x_sorted.min(), x_sorted.max(), 300)
y_smooth = poly(x_smooth)

plt.plot(x_smooth, y_smooth, color='red', linestyle='-', linewidth=2, zorder=1)
# Label map and coordinate tracker
texts = []
coords = {}
label_map = {}
for label in labels:
    print(f'label={label}')
    name = label.split('_step_')[1]
    label_map[label] = name

    xi = df.loc[df['Folder'] == label, x_column].values[0]
    yi = y_transform(df.loc[df['Folder'] == label, y_column].values[0])
    coords[name] = (xi, yi)

    txt = plt.text(xi, yi, name, fontsize=12, weight='bold' if "Our" in name else 'normal',
                   color='red' if "Our" in name else 'black')
    texts.append(txt)

# Adjust label overlaps
adjust_text(texts, only_move={'points': 'xy', 'texts': 'xy'}, expand_text=(1.05, 1.05), force_text=1)

# Style
# plt.title("Perception–Distortion (FID vs 1 - SSIM)", fontsize=20)
plt.xlabel(xlabel, fontsize=16)
plt.ylabel(ylabel, fontsize=16)
plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.tight_layout()

# Save
os.makedirs(output_dir, exist_ok=True)
save_path = os.path.join(output_dir, f"{filter_string}_epoch{epoch}_{x_column}_vs_{y_column}_steps.pdf")
plt.savefig(save_path, format='pdf', dpi=300)
plt.close()

print(f"Plot saved to {save_path}")