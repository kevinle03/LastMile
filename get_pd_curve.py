import os
import re
import pandas as pd
import matplotlib.pyplot as plt
from adjustText import adjust_text
import numpy as np

# ───────── USER CONFIGURATION ─────────
csv_file      = './DIV2K_BB_metrics/mean_metrics.csv'
filter_string = 'DIV2K_bicubic_Base'  # 'FHDMi_Base', 'FHDMi_Large', 'DIV2K_bicubic_Large', 'DIV2K_bicubic_Base'
epoch         = 600
x_column      = 'FID'
y_column      = 'SSIM'  # we'll use 1 - SSIM
y_transform   = lambda ssim: 1 - ssim
output_dir    = 'pd_figures'
highlight_key = 'Z2X|NC'
# ────────────────────────────────────────────────────

# Load CSV
df = pd.read_csv(csv_file)

# Filter Folder names
if filter_string is not None:
    pattern = re.compile(filter_string)
    if 'DIV2K' in filter_string:
        dataset = 'DIV2K'
    elif 'FHDMi' in filter_string:
        dataset = 'FHDMi'
    else:
        raise ValueError("Unknown dataset in filter_string")

    degradation = 'unknown' if 'unknown' in filter_string else 'bicubic'
    model = 'Large' if 'Large' in filter_string else 'Base'

    def keep_folder(d):
        return (
            (pattern.search(d) and f'epoch{epoch}' in d and 'step_20' in d)
            or ('HR' in d or 'target' in d)
            or ('LR' in d and degradation in d or 'source' in d)
            or (model in d and 'output' in d and (degradation in d or dataset == 'FHDMi'))
        )

    df = df[df['Folder'].apply(keep_folder)].sort_values('Folder')

# Prepare data
x = df[x_column]  # FID
y = y_transform(df[y_column])  # 1 - SSIM
labels = df['Folder']
xlabel = f"{x_column} (↓ better)"
ylabel = f"1 - {y_column} (↓ better)"

# Start plot
plt.figure(figsize=(10, 8))
plt.scatter(x, y, color='gray', marker='x', s=100, label='All Methods')

# Label map and coordinate tracker
texts = []
highlighted = False
coords = {}
label_map = {}
for label in labels:
    if "LR" in label or 'source' in label:
        name = "Y"
    elif "output" in label:
        name = "Z"
    elif "exp1" in label:
        name = "Z2X|Y"
    elif "exp2" in label:
        name = "Z2X|NC\n(Our)"
    elif "exp3" in label:
        name = "N2X|Y"
    elif "exp4" in label:
        name = "Y2X|Z"
    elif "exp5" in label:
        name = "N2X|Z"
    else:
        continue
    print(f'label={label}')
    label_map[label] = name

    xi = df.loc[df['Folder'] == label, x_column].values[0]
    yi = y_transform(df.loc[df['Folder'] == label, y_column].values[0])
    coords[name] = (xi, yi)

    # highlight "Our" method
    if "Our" in name and not highlighted:
        plt.scatter([xi], [yi], color='red', s=180, edgecolors='black', linewidths=1.5, label='Ours')
        highlighted = True

    txt = plt.text(xi, yi, name, fontsize=12, weight='bold' if "Our" in name else 'normal',
                   color='red' if "Our" in name else 'black')
    texts.append(txt)

# Adjust label overlaps
adjust_text(texts, only_move={'points': 'xy', 'texts': 'xy'}, expand_text=(1.05, 1.05), force_text=1)

# Draw colored arrows from Z to specific targets
arrow_paths = {
    "Z2X|Y": "blue",
    "N2X|Y": "green",
    "Z2X|NC\n(Our)": "red",
    "Y2X|Z": "orange",
    "N2X|Z": "purple"
}

for to_label, color in arrow_paths.items():
    if "Z" in coords and to_label in coords:
        x_start, y_start = coords["Z"]
        x_end, y_end = coords[to_label]
        plt.annotate("",
            xy=(x_end, y_end), xytext=(x_start, y_start),
            arrowprops=dict(arrowstyle='->', linewidth=2, color=color),
            zorder=2
        )
        plt.plot([], [], color=color, lw=2, label=f"Z → {to_label}")

# Style
plt.title("Perception–Distortion (FID vs 1 - SSIM)", fontsize=20)
plt.xlabel(xlabel, fontsize=16)
plt.ylabel(ylabel, fontsize=16)
plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(fontsize=13, loc='best')
plt.tight_layout()

# Save
os.makedirs(output_dir, exist_ok=True)
save_path = os.path.join(output_dir, f"{filter_string}_{x_column}_vs_1m{y_column}_colored_arrows.pdf")
plt.savefig(save_path, format='pdf', dpi=300)
plt.close()

print(f"Arrow plot saved to {save_path}")