import os
import torch
from torchvision.io import read_image
from torchmetrics.functional import peak_signal_noise_ratio, structural_similarity_index_measure
import lpips
import csv
from torchmetrics.image.fid import FrechetInceptionDistance as FID
from torchmetrics.image.kid import KernelInceptionDistance as KID
from pyiqa import create_metric
from statistics import mean
import random
import glob

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

hr_test_dir = '/scratch/ll5484/lillian/GM/dataset/FHDMi/test/target'
test_root_dir = '/scratch/ll5484/lillian/GM/dataset/FHDMi/test'
test_dirs = [os.path.join(test_root_dir, folder) for folder in os.listdir(test_root_dir) 
             if os.path.isdir(os.path.join(test_root_dir, folder)) and folder != os.path.basename(hr_test_dir)]

filter_for = None
if filter_for:
    test_dirs = [d for d in test_dirs if filter_for in os.path.basename(d)]
    
filter_out = None
if filter_out:
    test_dirs = [d for d in test_dirs if filter_out not in os.path.basename(d)]

# out_test_dir = './FHDMi_Base_metrics'
out_test_dir = './testing_metrics_code'

def save_metrics_to_csv(out_dir, folder_name, psnr, ssim, lpips, fid, kid_mean, kid_std, niqe):
    mean_csv_path = os.path.join(out_dir, "mean_metrics.csv")

    # Save mean metrics
    mean_file_exists = os.path.exists(mean_csv_path)
    with open(mean_csv_path, mode='a', newline='') as mean_file:
        writer = csv.writer(mean_file)
        if not mean_file_exists:
            writer.writerow(["Folder", "PSNR", "SSIM", "LPIPS", "FID", "KID_mean", "KID_std", "NIQE"])
        writer.writerow([folder_name, mean(psnr), mean(ssim), mean(lpips), fid, kid_mean, kid_std, mean(niqe)])

def calculate_metrics(lr_dirs, hr_dir, out_dir):
    calculate_lpips = lpips.LPIPS(net='alex').to(device)
    niqe = create_metric('niqe', device=device, as_loss=False)
    fid = FID(normalize=True).to(device)
    kid = KID(normalize=True, subset_size=50).to(device)

    hr_files = sorted(os.listdir(hr_dir))
    
    min_lr_dir = lr_dirs[0]
    for lr_dir in lr_dirs:
        if len(os.listdir(lr_dir)) < len(os.listdir(min_lr_dir)):
            min_lr_dir = lr_dir
        
    image_number_list = sorted([file_name.split('_')[1].split('.')[0] for file_name in os.listdir(min_lr_dir)])

    for lr_dir in sorted(lr_dirs):
        print(f"Calculating metrics for {lr_dir}...")
        psnr_results, ssim_results, lpips_results, niqe_results = [], [], [], []
        for idx, image_number in enumerate(image_number_list):
            lr_file_path = glob.glob(f"{lr_dir}/*{image_number}*")[0]
            hr_file_path = glob.glob(f"{hr_dir}/*{image_number}*")[0]
            if not lr_file_path or not hr_file_path:
                print(f"Skipping {image_number} due to missing files in {lr_dir} or {hr_dir}.")
                continue
            print(f"Processing {idx+1}/{len(image_number_list)}: {lr_file_path}")
            lr_tensor_whole = read_image(lr_file_path).unsqueeze(0).to(device)/255.0  # [B, C, H, W]; range is [0,1]; dtype=torch.float32
            hr_tensor_whole = read_image(hr_file_path).unsqueeze(0).to(device)/255.0
            if lr_tensor_whole.shape != hr_tensor_whole.shape:
                lr_tensor_whole = torch.nn.functional.interpolate(lr_tensor_whole, size=hr_tensor_whole.shape[2:], mode='bicubic', align_corners=False).clamp(0.0, 1.0)

            x_tiles = lr_tensor_whole.shape[3] // 256
            y_tiles = lr_tensor_whole.shape[2] // 256

            for j in range(y_tiles):
                for i in range(x_tiles): # variable i used twice; may caused issue but i tested and seems like no issue
                    lr_tensor = lr_tensor_whole[:, :, j*256:(j+1)*256, i*256:(i+1)*256]
                    hr_tensor = hr_tensor_whole[:, :, j*256:(j+1)*256, i*256:(i+1)*256]
            
                    kid.update(lr_tensor, real=False)
                    fid.update(lr_tensor, real=False)
                    kid.update(hr_tensor, real=True)
                    fid.update(hr_tensor, real=True)
            
            lr_tensor = lr_tensor_whole
            hr_tensor = hr_tensor_whole       
            psnr_results.append(peak_signal_noise_ratio(lr_tensor, hr_tensor, data_range=1.0).item())
            ssim_results.append(structural_similarity_index_measure(lr_tensor, hr_tensor, data_range=1.0).item())
            lpips_results.append(calculate_lpips(lr_tensor*2-1, hr_tensor*2-1).item())
            niqe_results.append(niqe(lr_tensor).item())
                
        fid_results = fid.compute().item()
        kid_mean_tensor, kid_std_tensor = kid.compute()
        kid_mean_results = kid_mean_tensor.item()
        kid_std_results  = kid_std_tensor.item()
        
        fid.reset()
        kid.reset()
        
        # Save results to text files
        folder_name = os.path.basename(lr_dir)
        save_metrics_to_csv(out_dir, folder_name, psnr_results, ssim_results, lpips_results, fid_results, kid_mean_results, kid_std_results, niqe_results)
        print(f"Metrics for {folder_name} saved to {out_dir}")

if __name__ == "__main__":
    # Calculate metrics for test set
    os.makedirs(out_test_dir, exist_ok=True)

    if not test_dirs:
        raise ValueError(f"No test directories found in {test_root_dir}.")
    if not os.listdir(hr_test_dir):
        raise ValueError(f"No files found in HR test directory {hr_test_dir}.")

    calculate_metrics(test_dirs, hr_test_dir, out_test_dir)
    print("Test metrics calculation completed.")