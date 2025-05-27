# rectified_flow_experiments.py

import pandas as pd
import torchvision
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
# from pytorch_fid import fid_score
# from cleanfid import fid
import os
import json
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from tqdm import tqdm
from PIL import Image
import shutil
from diffusers import AutoencoderKL, SD3Transformer2DModel
from diffusers.models.controlnets.controlnet_sd3 import SD3ControlNetModel
from accelerate import Accelerator
import lpips
from geomloss import SamplesLoss
import glob
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import argparse

import tensorflow as tf
import tensorflow_gan as tfgan
import numpy as np

def get_parser():
    parser = argparse.ArgumentParser(description='VIDEO DEMOIREING')
    parser.add_argument('--pretrained_teacher_model', type=str, default='stable-diffusion-v1-5',
                    help='Path or name of the pretrained teacher model')


    # Top-level fields from JSON
    parser.add_argument('--data_root', type=str, default='/scratch/ll5484/lillian/GM/dataset')
    parser.add_argument('--dataset', type=str, default='DIV2K_bicubic_Base') #ex: FHDMi_Base, FHDMi_Large, DIV2K_bicubic_Base, DIV2K_bicubic_Large, DIV2K_unknown_Base, DIV2K_unknown_Large
    parser.add_argument('--train_mode', type=str, default='train')
    parser.add_argument('--val_mode', type=str, default='valid')
    parser.add_argument('--num_epochs', type=int, default=1)
    # args.start_epoch
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--train_batchsize', type=int, default=2)
    parser.add_argument('--test_batchsize', type=int, default=1)#fix to 1
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--mode', type=str, default='train') # 'train' or 'eval'
    
    parser.add_argument('--crop_size', type=int, nargs=2, default=[256, 256])  # e.g., --crop_size 256 256
    parser.add_argument('--eval_interval', type=int, default=1)
    parser.add_argument('--eval_ckp', type=int,nargs='+', default=[240, 250])# number of checkpoints to evaluate
    parser.add_argument('--sample_steps', type=int, nargs='+', default=[50],
                        help='List of sampling steps for evaluation')# sample steps for evaluation
    parser.add_argument('--save_path', type=str, default='evaltb1000_debug')

    parser.add_argument('--experiments', type=json.loads, default=json.dumps([
        { "tag": "exp1_z2x_condY", "source": "z", "condition": "lr" },
        { "tag": "exp2_z2x_condNone", "source": "z", "condition": "null" },
        { "tag": "exp3_noise2x_condY", "source": "noise", "condition": "lr" },
        { "tag": "exp4_y2x_condNone", "source": "y", "condition": "null" },
        { "tag": "exp5_noise2x_condZ", "source": "noise", "condition": "mse" }
    ]), help='List of experiment configurations in JSON format')

    return parser.parse_args()

def load_images_from_folder(folder, image_size=(299, 299)):
    image_paths = sorted(glob.glob(os.path.join(folder, '*')))
    images = []
    for img_path in image_paths:
        img = Image.open(img_path).convert("RGB").resize(image_size)
        img = np.array(img).astype(np.float32) / 255.0
        images.append(img)
    return np.stack(images, axis=0)

def compute_fid_tf(real_dir, gen_dir):
    real_images = load_images_from_folder(real_dir)
    gen_images = load_images_from_folder(gen_dir)

    # real_tensor = tf.convert_to_tensor(real_images)
    # gen_tensor = tf.convert_to_tensor(gen_images)

    fid_score = tfgan.eval.frechet_inception_distance(
        real_images, gen_images,
        num_batches=1
    )
    return fid_score.numpy()

# -------------------------------
# 2. Custom Dataset Loader
# -------------------------------
class PairedImageDataset(Dataset):
    def __init__(self, crop_size, root, dataset, mode, center_crop=False):
        model = ''
        if 'Base' in dataset:
            model = '*Base*'
        elif 'Large' in dataset:
            model = '*Large*'
        if 'DIV2K' in dataset:
            downsample_mode = ''
            if 'bicubic' in dataset:
                downsample_mode = '*bicubic*'
            elif 'unknown' in dataset:
                downsample_mode = '*unknown*'
            dataset = 'DIV2K'
            if mode == 'test':
                mode = 'valid'
            self.y_dir = sorted(glob.glob(os.path.join(root, dataset, mode, "*LR*"+downsample_mode)))[0]
            self.z_dir = sorted(glob.glob(os.path.join(root, dataset, mode, downsample_mode+model+"*output*")))[0]
            self.x_dir = sorted(glob.glob(os.path.join(root, dataset, mode, "*HR*")))[0]
        elif 'FHDMi' in dataset:
            dataset = 'FHDMi'
            if mode == 'valid':
                mode = 'test'
            self.y_dir = glob.glob(os.path.join(root, dataset, mode, "*source*"))[0]
            self.z_dir = glob.glob(os.path.join(root, dataset, mode, model+"*output*"))[0]
            self.x_dir = glob.glob(os.path.join(root, dataset, mode, "*target*"))[0]
        else:
            # manually change to path of your dataset
            self.y_dir = os.path.join(root, dataset, mode, "lr")
            self.z_dir = os.path.join(root, dataset, mode, "mse_res")
            self.x_dir = os.path.join(root, dataset, mode, "gt")
            
        self.filenames_y = sorted(os.listdir(self.y_dir))
        self.filenames_z = sorted(os.listdir(self.z_dir))
        self.filenames_x = sorted(os.listdir(self.x_dir))
        if not (len(self.filenames_y) == len(self.filenames_z) == len(self.filenames_x)):
            raise ValueError(f"Mismatch in the number of images: "
                f"y_dir ({len(self.filenames_y)}), "
                f"z_dir ({len(self.filenames_z)}), "
                f"x_dir ({len(self.filenames_x)})")

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        self.crop_size = crop_size
        self.center_crop = center_crop

    def __len__(self):
        return len(self.filenames_x)

    def __getitem__(self, idx):
        y_img = Image.open(os.path.join(self.y_dir, self.filenames_y[idx])).convert("RGB")
        z_img = Image.open(os.path.join(self.z_dir, self.filenames_z[idx])).convert("RGB")
        x_img = Image.open(os.path.join(self.x_dir, self.filenames_x[idx])).convert("RGB")

        if y_img.size != z_img.size:
            y_img = y_img.resize(z_img.size, resample=Image.BILINEAR)
        if x_img.size != z_img.size:
            x_img = x_img.resize(z_img.size, resample=Image.BILINEAR)

        if self.center_crop:
            y_img = transforms.functional.center_crop(y_img, self.crop_size)
            z_img = transforms.functional.center_crop(z_img, self.crop_size)
            x_img = transforms.functional.center_crop(x_img, self.crop_size)
        else:
            i, j, h, w = transforms.RandomCrop.get_params(z_img, output_size=self.crop_size)
            y_img = transforms.functional.crop(y_img, i, j, h, w)
            z_img = transforms.functional.crop(z_img, i, j, h, w)
            x_img = transforms.functional.crop(x_img, i, j, h, w)

        return {
            "lr": self.transform(y_img),
            "mse": self.transform(z_img),
            "clean": self.transform(x_img),
            "name": self.filenames_x[idx]
        }

# -------------------------------
# 3. Evaluation Function
# -------------------------------
def evaluate_model(args, tag, source_type, condition_type, controlnet=None, transformer=None, checkpoint_dir=None, epoch_tag=None, val_loader=None, vae=None, device=None, writer=None):
    import numpy as np
    from kneed import KneeLocator
    import matplotlib.pyplot as plt
    import time

    accelerator = Accelerator()
    device = device or accelerator.device

    if vae is None:
        vae = AutoencoderKL.from_pretrained("stabilityai/stable-diffusion-3-medium-diffusers", subfolder="vae").to(device)

    close_writer = False
    if writer is None:
        log_dir = os.path.join(args.save_path, "tensorboard_eval", tag)
        os.makedirs(log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=log_dir)
        close_writer = True

    if args.mode == "eval":
        assert checkpoint_dir is not None and args.eval_ckp is not None
        for epoch in args.eval_ckp:
            epoch_tag = f"epoch{epoch}"
            transformer_path = os.path.join(checkpoint_dir, f"transformer_{epoch_tag}")
            controlnet_path = os.path.join(checkpoint_dir, f"controlnet_{epoch_tag}")

            if not os.path.exists(transformer_path):
                raise FileNotFoundError(f"Pretrained transformer model not found at {transformer_path}.")
            if not os.path.exists(controlnet_path):
                raise FileNotFoundError(f"Pretrained controlnet model not found at {controlnet_path}.")

            transformer = SD3Transformer2DModel.from_pretrained(transformer_path).to(device)
            controlnet = SD3ControlNetModel.from_pretrained(controlnet_path, use_safetensors=True).to(device)

            args.mode = "train"
            return evaluate_model(args, tag, source_type, condition_type, controlnet, transformer, checkpoint_dir, epoch_tag, val_loader, vae, device)

    if val_loader is None:
        val_dataset = PairedImageDataset(args.crop_size, args.data_root, dataset=args.dataset, mode=args.val_mode, center_crop=True)
        val_loader = DataLoader(val_dataset, batch_size=args.test_batchsize, shuffle=False)

    vae, controlnet, transformer, val_loader = accelerator.prepare(vae, controlnet, transformer, val_loader)

    lpips_model = lpips.LPIPS(net="alex").to(device)
    sinkhorn = SamplesLoss("sinkhorn", p=2, blur=0.01)

    pooled_dim = controlnet.config.pooled_projection_dim
    joint_dim = controlnet.config.joint_attention_dim
    seq_len = controlnet.config.pos_embed_max_size
    extra_channels = controlnet.config.extra_conditioning_channels

    controlnet.eval()
    transformer.eval()

    steps_list, psnr_list, ssim_list, lpips_list, fid_list, time_list = [], [], [], [], [], []
    sample_steps = [50]#list(range(5, 56, 5))

    for sample_step in sample_steps:
        lpips_scores, psnr_scores, ssim_scores = [], [], []
        x_real_all, x_fake_all = [], []
        # if not getattr(args, "skip_fid", False):
        #     save_dir = os.path.join(args.save_path, tag, f"{epoch_tag}_step{sample_step}")
        #     gen_dir = os.path.join(save_dir, "gen")
        #     gt_dir = os.path.join(save_dir, "gt")
        #     os.makedirs(gen_dir, exist_ok=True)
        #     os.makedirs(gt_dir, exist_ok=True)

        def denormalize(x):
            return ((x + 1) / 2.0).clamp(0, 1)

        start_time = time.time()
        for batch_idx, batch in enumerate(tqdm(val_loader, desc=f"Evaluating {tag} - step {sample_step}")):
            y = batch["lr"].to(device)
            z = batch["mse"].to(device)
            x_gt = batch["clean"].to(device)
            if y.shape != z.shape:
                y = F.interpolate(y, size=z.shape[-2:], mode="bilinear")
            name = batch["name"][0] if isinstance(batch["name"], list) else batch["name"]

            with torch.no_grad():
                y_latent = vae.encode(y).latent_dist.sample() * vae.config.scaling_factor
                z_latent = vae.encode(z).latent_dist.sample() * vae.config.scaling_factor
                x_latent = vae.encode(x_gt).latent_dist.sample() * vae.config.scaling_factor

                x0 = {"z": z_latent, "y": y_latent, "noise": torch.randn_like(x_latent)}[source_type]

                xt = x0
                encoder_hidden = torch.zeros((xt.size(0), seq_len, joint_dim), device=device)
                pooled_proj = torch.zeros((xt.size(0), pooled_dim), device=device)

                control_cond = {"lr": y_latent, "mse": z_latent, "null": torch.zeros_like(x0)}[condition_type]
                if control_cond.size(1) != xt.size(1) + extra_channels:
                    control_cond = F.pad(control_cond, (0, 0, 0, 0, 0, extra_channels))

                for t in range(sample_step):
                    timestep = torch.ones(xt.size(0), device=device).long() * int((t + 1) / sample_step * 1000)
                    controlnet_out = controlnet(controlnet_cond=control_cond, hidden_states=xt,
                                                timestep=timestep, encoder_hidden_states=encoder_hidden,
                                                pooled_projections=pooled_proj)[0]

                    pred_velocity = transformer(hidden_states=xt, timestep=timestep,
                                                encoder_hidden_states=encoder_hidden,
                                                pooled_projections=pooled_proj,
                                                block_controlnet_hidden_states=controlnet_out)[0]

                    xt = xt + pred_velocity / sample_step

                x_pred = vae.decode(xt / vae.config.scaling_factor).sample

            # if not getattr(args, "skip_fid", False):
            #     save_image(denormalize(x_pred), os.path.join(gen_dir, name))
            #     save_image(denormalize(x_gt), os.path.join(gt_dir, name))

            if batch_idx == 0:
                writer.add_images(f"{tag}/step{sample_step}/gen", denormalize(x_pred), global_step=int(epoch_tag.strip("epoch")))
                writer.add_images(f"{tag}/step{sample_step}/gt", denormalize(x_gt), global_step=int(epoch_tag.strip("epoch")))

            batch_lpips = lpips_model(x_gt, x_pred).squeeze().detach().cpu()
            lpips_scores.append(batch_lpips.item() if batch_lpips.ndim == 0 else batch_lpips.mean().item())
            x_real_all.append(x_gt)
            x_fake_all.append(x_pred)

        elapsed = time.time() - start_time
        x_real_all = torch.cat(x_real_all, dim=0)
        x_fake_all = torch.cat(x_fake_all, dim=0)

        fid_value = 0.0
        # if not getattr(args, "skip_fid", False):
        #     fid_value = compute_fid_tf(gt_dir, gen_dir)
        #     shutil.rmtree(gt_dir)
        #     shutil.rmtree(gen_dir)
        #     shutil.rmtree(save_dir)

        for i in range(len(x_real_all)):
            x1 = x_real_all[i].permute(1, 2, 0).cpu().numpy()
            x2 = x_fake_all[i].permute(1, 2, 0).cpu().numpy()
            psnr_scores.append(psnr(x1, x2, data_range=1))
            ssim_scores.append(ssim(x1, x2, channel_axis=2, data_range=1))

        avg_lpips = sum(lpips_scores) / len(lpips_scores)
        avg_psnr = np.mean(psnr_scores)
        avg_ssim = np.mean(ssim_scores)

        writer.add_scalar(f"{tag}/step{sample_step}/lpips", avg_lpips, int(epoch_tag.strip("epoch")))
        writer.add_scalar(f"{tag}/step{sample_step}/psnr", avg_psnr, int(epoch_tag.strip("epoch")))
        writer.add_scalar(f"{tag}/step{sample_step}/ssim", avg_ssim, int(epoch_tag.strip("epoch")))
        if not getattr(args, "skip_fid", False):
            writer.add_scalar(f"{tag}/step{sample_step}/fid", fid_value, int(epoch_tag.strip("epoch")))

        steps_list.append(sample_step)
        psnr_list.append(avg_psnr)
        ssim_list.append(avg_ssim)
        lpips_list.append(avg_lpips)
        fid_list.append(fid_value)
        time_list.append(elapsed)

    best_metrics = {
        "LPIPS": lpips_list[np.argmin(lpips_list)],
        "FID": fid_list[np.argmin(fid_list)] if fid_list else None,
        "PSNR": psnr_list[np.argmax(psnr_list)],
        "SSIM": ssim_list[np.argmax(ssim_list)],
        "Step_LPIPS": steps_list[np.argmin(lpips_list)],
        "Step_FID": steps_list[np.argmin(fid_list)] if fid_list else None,
        "Step_PSNR": steps_list[np.argmax(psnr_list)],
        "Step_SSIM": steps_list[np.argmax(ssim_list)]
    }
    if close_writer:
        writer.close()
    return best_metrics


# -------------------------------
# 4. Training Function
# -------------------------------

def print_memory(stage=""):
    """Helper to print current GPU memory usage."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024**3)
        reserved = torch.cuda.memory_reserved() / (1024**3)
        print(f"[{stage}] Allocated: {allocated:.2f} GiB | Reserved: {reserved:.2f} GiB")

def train_model(args, tag, source_type, condition_type):
    accelerator = Accelerator(
        mixed_precision="fp16", 
        gradient_accumulation_steps=5
    )
    device = accelerator.device
    # Load models
    checkpoint_dir = os.path.join(args.save_path, tag, "checkpoints")
    if os.path.exists(checkpoint_dir):
        latest_controlnet_ckpt = sorted(glob.glob(os.path.join(checkpoint_dir, "controlnet_epoch*")), key=lambda x: int(x.split("_epoch")[-1]), reverse=True)
        latest_transformer_ckpt = sorted(glob.glob(os.path.join(checkpoint_dir, "transformer_epoch*")), key=lambda x: int(x.split("_epoch")[-1]), reverse=True)

        if latest_controlnet_ckpt and latest_transformer_ckpt:
            controlnet_epoch = int(latest_controlnet_ckpt[0].split("_epoch")[-1])
            transformer_epoch = int(latest_transformer_ckpt[0].split("_epoch")[-1])

            if controlnet_epoch == transformer_epoch:
                print(f"Loading checkpoint from epoch {controlnet_epoch}")
                args.start_epoch=controlnet_epoch
                controlnet = SD3ControlNetModel.from_pretrained(latest_controlnet_ckpt[0], use_safetensors=True).to(device)
                transformer = SD3Transformer2DModel.from_pretrained(latest_transformer_ckpt[0]).to(device)
            else:
                print("Mismatch in checkpoint epochs. Initializing models from scratch.")
                controlnet = None
                transformer = None
        else:
            print("No valid checkpoint found. Initializing models from scratch.")
            controlnet = None
            transformer = None
    else:
        print("Checkpoint directory does not exist. Initializing models from scratch.")
        controlnet = None
        transformer = None
    if controlnet is None or transformer is None:
        transformer = SD3Transformer2DModel.from_pretrained("stabilityai/stable-diffusion-3-medium-diffusers", subfolder="transformer").to(device)
        controlnet = SD3ControlNetModel.from_pretrained("alimama-creative/SD3-Controlnet-Inpainting", use_safetensors=True).to(device)
    vae = AutoencoderKL.from_pretrained("stabilityai/stable-diffusion-3-medium-diffusers", subfolder="vae").to(device)
    
    # Prepare datasets and optimizer
    train_dataset = PairedImageDataset(args.crop_size, args.data_root, dataset=args.dataset, mode=args.train_mode)
    val_dataset = PairedImageDataset(args.crop_size, args.data_root, dataset=args.dataset, mode=args.val_mode, center_crop=True)
    train_loader = DataLoader(train_dataset, batch_size=args.train_batchsize, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.test_batchsize, shuffle=False)
    
    params = list(controlnet.parameters()) + list(transformer.parameters())
    optimizer = torch.optim.AdamW(params, lr=args.lr)
    vae, controlnet, transformer, optimizer, train_loader, val_loader = accelerator.prepare(
        vae, controlnet, transformer, optimizer, train_loader, val_loader)
    
    pooled_dim = controlnet.config.pooled_projection_dim
    joint_dim = controlnet.config.joint_attention_dim
    seq_len = controlnet.config.pos_embed_max_size
    extra_channels = controlnet.config.extra_conditioning_channels
    
    controlnet.train()
    transformer.train()

    log_dir = os.path.join(args.save_path, tag, "tensorboard")
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)
    checkpoint_dir = os.path.join(args.save_path, tag, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    eval_interval = args.eval_interval
    best_scores = {"LPIPS": float("inf"), "FID": float("inf"), "PSNR": float("-inf"), "SSIM": float("-inf")}
    best_ckpts = {k: None for k in best_scores}

    for epoch in range(args.start_epoch, args.num_epochs):
        epoch_losses = []
        for batch in tqdm(train_loader, desc=f"Training {tag} - Epoch {epoch + 1}"):
#########################################################################
# this part below is the orginal code from train_controlnet_sd3tb.py
            # y = batch["lr"].to(device)
            # z = batch["mse"].to(device)
            # x_gt = batch["clean"].to(device)
            # if y.shape != z.shape:
            #     y = F.interpolate(y, size=z.shape[-2:], mode="bilinear")
            # if x_gt.shape != z.shape:
            #     x_gt = F.interpolate(x_gt, size=z.shape[-2:], mode="bilinear")
                
            # y_latent = vae.encode(y).latent_dist.sample() * vae.config.scaling_factor
            # z_latent = vae.encode(z).latent_dist.sample() * vae.config.scaling_factor
            # if y_latent.shape != z_latent.shape:
            #     y_latent = F.interpolate(y_latent, size=z_latent.shape[-2:], mode="bilinear")
            # x_latent = vae.encode(x_gt).latent_dist.sample() * vae.config.scaling_factor
            
            # x0 = {"z": z_latent, "y": y_latent, "noise": torch.randn_like(x_latent)}[source_type]
            # x1 = x_latent
            # t = torch.rand(x0.size(0), device=device)
            # xt = (1 - t.view(-1, 1, 1, 1)) * x0 + t.view(-1, 1, 1, 1) * x1
            # target_velocity = x1 - x0
            
            # encoder_hidden = torch.zeros((xt.size(0), seq_len, joint_dim), device=device)
            # pooled_proj = torch.zeros((xt.size(0), pooled_dim), device=device)
            
            # control_cond = y_latent if condition_type == "lr" else torch.zeros_like(x0)
            # if control_cond.size(1) != xt.size(1) + extra_channels:
            #     control_cond = F.pad(control_cond, (0, 0, 0, 0, 0, extra_channels))
#########################################################################
# part below came from train_controlnet_sd3.py and modified.
            x_gt = batch["clean"].to(device)
            x_latent = vae.encode(x_gt).latent_dist.sample() * vae.config.scaling_factor

            # Determine x0 based on source_type
            if source_type == "z":
                z = batch["mse"].to(device)
                z_latent = vae.encode(z).latent_dist.sample() * vae.config.scaling_factor
                x0 = z_latent
            elif source_type == "y":
                y = batch["lr"].to(device)
                if y.shape != batch["mse"].shape:
                    y = F.interpolate(y, size=batch["mse"].shape[-2:], mode="bilinear")
                y_latent = vae.encode(y).latent_dist.sample() * vae.config.scaling_factor
                x0 = y_latent
            elif source_type == "noise":
                x0 = torch.randn_like(x_latent)
            else:
                raise ValueError(f"Unknown source_type {source_type}")

            # Create interpolation for flow
            x1 = x_latent
            t = torch.rand(x0.size(0), device=device)
            xt = (1 - t.view(-1, 1, 1, 1)) * x0 + t.view(-1, 1, 1, 1) * x1
            target_velocity = x1 - x0

            # Prepare ControlNet conditioning
            if condition_type == "lr":
                y = batch["lr"].to(device)
                if y.shape != batch["mse"].shape:
                    y = F.interpolate(y, size=batch["mse"].shape[-2:], mode="bilinear")
                y_latent = vae.encode(y).latent_dist.sample() * vae.config.scaling_factor
                control_cond = y_latent
            elif condition_type == "mse":
                z = batch["mse"].to(device)
                z_latent = vae.encode(z).latent_dist.sample() * vae.config.scaling_factor
                control_cond = z_latent
            else:
                y_latent = torch.zeros_like(x0)
                control_cond = y_latent

            if control_cond.size(1) != xt.size(1) + extra_channels:
                control_cond = F.pad(control_cond, (0, 0, 0, 0, 0, extra_channels))

            encoder_hidden = torch.zeros((xt.size(0), seq_len, joint_dim), device=device)
            pooled_proj = torch.zeros((xt.size(0), pooled_dim), device=device)                
#########################################################################
            with accelerator.accumulate([controlnet, transformer]):
                controlnet_out = controlnet(
                    controlnet_cond=control_cond, hidden_states=xt, 
                    timestep=(t * 1000).long().view(-1),
                    encoder_hidden_states=encoder_hidden, pooled_projections=pooled_proj
                )[0]
                transformer_out = transformer(
                    hidden_states=xt, timestep=(t * 1000).long().view(-1),
                    encoder_hidden_states=encoder_hidden, pooled_projections=pooled_proj,
                    block_controlnet_hidden_states=controlnet_out
                )[0]
                
                if transformer_out.shape != target_velocity.shape:
                    transformer_out = F.interpolate(transformer_out, size=target_velocity.shape[-2:], mode="bilinear")
                    
                loss = F.mse_loss(transformer_out, target_velocity)
                accelerator.backward(loss)
                epoch_losses.append(loss.item())
                optimizer.step()
                optimizer.zero_grad()

        avg_loss = sum(epoch_losses) / len(epoch_losses)
        writer.add_scalar(f"{tag}/train/loss", avg_loss, epoch + 1)

        if (epoch + 1) % eval_interval == 0:
            epoch_tag = f"epoch{epoch + 1}"
            args.skip_fid = True

            metrics = evaluate_model(
                args=args, tag=tag, source_type=source_type, condition_type=condition_type,
                checkpoint_dir=checkpoint_dir, epoch_tag=epoch_tag,
                val_loader=val_loader, vae=vae, device=device,
                controlnet=controlnet, transformer=transformer, writer=writer
            )

            # improved = False
            # for key in best_scores:
            #     metric = metrics.get(key, None)
            #     if metric is not None:
            #         if (key in ["LPIPS"] and metric < best_scores[key]) or (key in ["PSNR", "SSIM"] and metric > best_scores[key]):
            #             best_scores[key] = metric
            #             if best_ckpts[key] is not None and os.path.exists(best_ckpts[key]):
            #                 shutil.rmtree(best_ckpts[key])
            #             ckpt_path = os.path.join(checkpoint_dir, f"best_{key}_{tag}")
            #             controlnet.save_pretrained(os.path.join(ckpt_path, "controlnet"))
            #             transformer.save_pretrained(os.path.join(ckpt_path, "transformer"))
            #             best_ckpts[key] = ckpt_path
            #             improved = True

            # if improved:
            #     print(f"Saved improved model at epoch {epoch + 1}: {best_scores}")

            # Always save latest checkpoint
            controlnet.save_pretrained(os.path.join(checkpoint_dir, f"controlnet_{epoch_tag}"))
            transformer.save_pretrained(os.path.join(checkpoint_dir, f"transformer_{epoch_tag}"))

            controlnet.train()
            transformer.train()

    writer.close()

# -------------------------------
# 5. Run Configured Experiments
# -------------------------------
if __name__ == '__main__':
    args = get_parser()
    best_scores = {}
    for exp in args.experiments:
        tag = exp["tag"]
        if args.mode == 'train':
            train_model(args, tag, exp["source"], exp["condition"])
        else:
            metrics = evaluate_model(
                args=args,
                tag=tag,
                source_type=exp["source"],
                condition_type=exp["condition"],
                checkpoint_dir=os.path.join(args.save_path, tag, "checkpoints"),
                epoch_tag=f"epoch{args.eval_ckp[0]}",  # Default to first epoch in eval_ckp
                val_loader=None,  # Should be passed from training context for full reuse
                vae=None,         # Load in training context if needed
                device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
            )
            best_scores[tag] = metrics
    print("\nBest evaluation scores:")
    for tag, scores in best_scores.items():
        print(f"{tag}: {scores}")
# if __name__ == '__main__':
#     args = get_parser()
#     # args.mode = 'train'
#     for exp in args.experiments:
#         if args.mode == 'train':
#             train_model(args, exp["tag"], exp["source"], exp["condition"])
#         else:
#             evaluate_model(args, exp["tag"], exp["source"], exp["condition"])