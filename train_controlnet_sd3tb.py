import os
import json
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
from PIL import Image
from diffusers import AutoencoderKL, SD3Transformer2DModel
from diffusers.models.controlnets.controlnet_sd3 import SD3ControlNetModel
from accelerate import Accelerator
import glob
from torch.utils.tensorboard import SummaryWriter
import argparse

def get_parser():
    parser = argparse.ArgumentParser(description="Train SD3 with ControlNet")
    parser.add_argument('--data_root', type=str, default='/scratch/ll5484/lillian/GM/dataset')
    parser.add_argument('--dataset', type=str, default='DIV2K_bicubic_Base') #ex: FHDMi_Base, FHDMi_Large, DIV2K_bicubic_Base, DIV2K_bicubic_Large, DIV2K_unknown_Base, DIV2K_unknown_Large
    parser.add_argument('--num_epochs', type=int, default=1000)
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--train_batchsize', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--crop_size', type=int, nargs=2, default=[256, 256])  # e.g., --crop_size 256 256
    parser.add_argument('--checkpoint_interval', type=int, default=50)
    parser.add_argument('--save_path', type=str, default='evaltb1000_debug')

    parser.add_argument('--experiments', type=json.loads, default=json.dumps([
        { "tag": "exp1_z2x_condY", "source": "z", "condition": "lr" },
        { "tag": "exp2_z2x_condNone", "source": "z", "condition": "null" },
        { "tag": "exp3_noise2x_condY", "source": "noise", "condition": "lr" },
        { "tag": "exp4_y2x_condNone", "source": "y", "condition": "null" },
        { "tag": "exp5_noise2x_condZ", "source": "noise", "condition": "mse" }
    ]), help='List of experiment configurations in JSON format')

    return parser.parse_args()

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
    train_dataset = PairedImageDataset(args.crop_size, args.data_root, args.dataset, 'train')
    train_loader = DataLoader(train_dataset, batch_size=args.train_batchsize, shuffle=True)
    
    params = list(controlnet.parameters()) + list(transformer.parameters())
    optimizer = torch.optim.AdamW(params, lr=args.lr)
    vae, controlnet, transformer, optimizer, train_loader = accelerator.prepare(
        vae, controlnet, transformer, optimizer, train_loader)
    
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
    
    checkpoint_interval = args.checkpoint_interval

    for epoch in range(args.start_epoch, args.num_epochs):
        epoch_losses = []
        for batch in tqdm(train_loader, desc=f"Training {tag} - Epoch {epoch + 1}"):
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

        if (epoch + 1) % checkpoint_interval == 0:
            epoch_tag = f"epoch{epoch + 1}"
            args.skip_fid = True

            controlnet.save_pretrained(os.path.join(checkpoint_dir, f"controlnet_{epoch_tag}"))
            transformer.save_pretrained(os.path.join(checkpoint_dir, f"transformer_{epoch_tag}"))

    writer.close()

if __name__ == '__main__':
    args = get_parser()
    for exp in args.experiments:
        tag = exp["tag"]
        train_model(args, tag, exp["source"], exp["condition"])