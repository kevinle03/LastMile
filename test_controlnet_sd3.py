import os
import json
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from tqdm import tqdm
from PIL import Image
from diffusers import AutoencoderKL, SD3Transformer2DModel
from diffusers.models.controlnets.controlnet_sd3 import SD3ControlNetModel
from accelerate import Accelerator
import glob
import argparse
import math

def get_parser():
    parser = argparse.ArgumentParser(description='IMAGE SUPER-RESOLUTION')
    parser.add_argument('--pretrained_teacher_model', type=str, default='stable-diffusion-v1-5',
                    help='Path or name of the pretrained teacher model')
    parser.add_argument('--data_root', type=str, default='/scratch/ll5484/lillian/GM/dataset')
    parser.add_argument('--dataset', type=str, default='DIV2K_bicubic_Base') #ex: FHDMi_Base, FHDMi_Large, DIV2K_bicubic_Base, DIV2K_bicubic_Large, DIV2K_unknown_Base, DIV2K_unknown_Large
    parser.add_argument('--save_path', type=str, default='evaltb1000BB_run4')
    parser.add_argument('--crop_size', type=int, nargs=2, default=[256, 256])  # e.g., --crop_size 256 256
    parser.add_argument('--sample_steps', type=int, nargs='+', default=[1],
                        help='List of sampling steps for evaluation')# sample steps for evaluation
    parser.add_argument('--output_folder', type=str, default='/scratch/ll5484/lillian/GM/dataset/DIV2K/valid')
    parser.add_argument('--experiments', type=json.loads, default=json.dumps([
        { "tag": "exp1_z2x_condY", "source": "z", "condition": "lr", "eval_ckp": [] },
        { "tag": "exp2_z2x_condNone", "source": "z", "condition": "null", "eval_ckp": [1000] },
        { "tag": "exp3_noise2x_condY", "source": "noise", "condition": "lr", "eval_ckp": [] },
        { "tag": "exp4_y2x_condNone", "source": "y", "condition": "null", "eval_ckp": [] },
        { "tag": "exp5_noise2x_condZ", "source": "noise", "condition": "mse", "eval_ckp": [] },
    ]), help='List of experiment configurations in JSON format')

    return parser.parse_args()

# -------------------------------
# 2. Custom Dataset Loader
# -------------------------------
class PairedImageDataset(Dataset):
    def __init__(self, root, dataset, mode):
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

    def __len__(self):
        return len(self.filenames_x)

    def __getitem__(self, idx):
        y_img = Image.open(os.path.join(self.y_dir, self.filenames_y[idx])).convert("RGB")
        z_img = Image.open(os.path.join(self.z_dir, self.filenames_z[idx])).convert("RGB")
        x_img = Image.open(os.path.join(self.x_dir, self.filenames_x[idx])).convert("RGB")

        if y_img.size != z_img.size:
            y_img = y_img.resize(z_img.size, resample=Image.BILINEAR)

        return {
            "lr": self.transform(y_img),
            "mse": self.transform(z_img),
            "clean": self.transform(x_img),
            "name": self.filenames_x[idx]
        }

def tile_process(y_tensor, z_tensor, source_type, condition_type, sample_step, vae, transformer, controlnet, pooled_dim, joint_dim, seq_len, extra_channels, tile_size, tile_pad):
    if tile_pad >= tile_size:
        raise ValueError(f"tile_pad ({tile_pad}) must be smaller than tile_size ({tile_size}).")
    _, _, height, width = z_tensor.shape
    output_height = height
    output_width = width

    output = torch.zeros((1, 3, output_height, output_width), dtype=torch.float32, device=z_tensor.device)

    # Loop through all tiles
    count = 0
    start_y = 0
    while start_y < height:
        start_x = 0
        while start_x < width:
            count += 1
            # Define the input tile area within the original image
            if start_x + tile_size <= width:
                input_start_x = start_x
                input_end_x = start_x + tile_size
            else:
                input_start_x = width - tile_size
                input_end_x = width

            if start_y + tile_size <= height:
                input_start_y = start_y
                input_end_y = start_y + tile_size
            else:
                input_start_y = height - tile_size
                input_end_y = height

            # Extract the input tile with padding
            y_tensor_tile = y_tensor[:, :, input_start_y:input_end_y, input_start_x:input_end_x]
            z_tensor_tile = z_tensor[:, :, input_start_y:input_end_y, input_start_x:input_end_x]

            try:
                with torch.no_grad():
                    y_latent = vae.encode(y_tensor_tile).latent_dist.sample() * vae.config.scaling_factor
                    z_latent = vae.encode(z_tensor_tile).latent_dist.sample() * vae.config.scaling_factor

                    x0 = {"z": z_latent, "y": y_latent, "noise": torch.randn_like(z_latent)}[source_type]

                    xt = x0
                    encoder_hidden = torch.zeros((xt.size(0), seq_len, joint_dim), device=z_tensor.device)
                    pooled_proj = torch.zeros((xt.size(0), pooled_dim), device=z_tensor.device)

                    control_cond = {"lr": y_latent, "mse": z_latent, "null": torch.zeros_like(x0)}[condition_type]
                    if control_cond.size(1) != xt.size(1) + extra_channels:
                        control_cond = F.pad(control_cond, (0, 0, 0, 0, 0, extra_channels))

                    for t in range(sample_step):
                        timestep = torch.ones(xt.size(0), device=z_tensor.device).long() * int((t + 1) / sample_step * 1000)
                        controlnet_out = controlnet(controlnet_cond=control_cond, hidden_states=xt,
                                                    timestep=timestep, encoder_hidden_states=encoder_hidden,
                                                    pooled_projections=pooled_proj)[0]

                        pred_velocity = transformer(hidden_states=xt, timestep=timestep,
                                                    encoder_hidden_states=encoder_hidden,
                                                    pooled_projections=pooled_proj,
                                                    block_controlnet_hidden_states=controlnet_out)[0]

                        xt = xt + pred_velocity / sample_step

                    output_tile = vae.decode(xt / vae.config.scaling_factor).sample
                    
                    torch.cuda.empty_cache()

            except RuntimeError as error:
                print(f'Error processing tile: {error}')
                continue  # Skip this tile if an error occurs
            
            print(f'Processing Tile {count}')

            output[:, :, start_y:min(start_y + tile_size, height), start_x:min(start_x + tile_size, width)] = \
                output_tile[:, :, 0 if start_y+tile_size<=height else tile_size-(height-start_y):, 0 if start_x+tile_size<=width else tile_size-(width-start_x):]
            
            start_x += (tile_size - tile_pad)   # increment start_x leaving an overlap of size tile_pad betweeen tile 

        start_y += (tile_size - tile_pad)   # increment start_y

    return output  # Returns tensor in (B, C, H, W), range [-1, 1] (or same as input)

def denormalize(x):
    return ((x + 1) / 2.0).clamp(0, 1)
# -------------------------------
# 3. Evaluation Function
# -------------------------------
def test_model(args, tag, source_type, condition_type, checkpoint_dir, epoch, device):
    accelerator = Accelerator()
    device = device or accelerator.device

    vae = AutoencoderKL.from_pretrained("stabilityai/stable-diffusion-3-medium-diffusers", subfolder="vae").to(device)

    epoch_tag = f"epoch{epoch}"
    transformer_path = os.path.join(checkpoint_dir, f"transformer_{epoch_tag}")
    controlnet_path = os.path.join(checkpoint_dir, f"controlnet_{epoch_tag}")

    if not os.path.exists(transformer_path):
        raise FileNotFoundError(f"Pretrained transformer model not found at {transformer_path}.")
    if not os.path.exists(controlnet_path):
        raise FileNotFoundError(f"Pretrained controlnet model not found at {controlnet_path}.")

    transformer = SD3Transformer2DModel.from_pretrained(transformer_path).to(device)
    controlnet = SD3ControlNetModel.from_pretrained(controlnet_path, use_safetensors=True).to(device)

    val_dataset = PairedImageDataset(args.data_root, dataset=args.dataset, mode="valid")
    val_loader = DataLoader(val_dataset, shuffle=False)

    vae, controlnet, transformer, val_loader = accelerator.prepare(vae, controlnet, transformer, val_loader)

    pooled_dim = controlnet.config.pooled_projection_dim
    joint_dim = controlnet.config.joint_attention_dim
    seq_len = controlnet.config.pos_embed_max_size
    extra_channels = controlnet.config.extra_conditioning_channels

    controlnet.eval()
    transformer.eval()

    for sample_step in args.sample_steps:
        for batch_idx, batch in enumerate(tqdm(val_loader, desc=f"Evaluating {tag} - epoch {epoch} - step {sample_step}")):
            # if batch_idx<=30:
            #     continue
            y = batch["lr"].to(device)
            z = batch["mse"].to(device)
            if y.shape != z.shape:
                y = F.interpolate(y, size=z.shape[-2:], mode="bilinear")
            name = batch["name"][0] if isinstance(batch["name"], list) else batch["name"]

            tile_pad = 0
            tile_size = 256
            output_img = tile_process(y, z, source_type, condition_type, sample_step, vae, transformer, controlnet, pooled_dim, joint_dim, seq_len, extra_channels, tile_size, tile_pad)
            normalized_out_img = denormalize(output_img.squeeze(0)) # normalize from [-1, 1] to [0, 1]
            # Save the processed image
            output_dir = os.path.join(args.output_folder, args.dataset + "_" + tag.split("_")[0] + "_epoch" + str(epoch) + f"_step_{sample_step}")
            os.makedirs(output_dir, exist_ok=True)
            
            save_path = os.path.join(output_dir, name)
            print(save_path)
            save_image(normalized_out_img, save_path)


# -------------------------------
# 5. Run Configured Experiments
# -------------------------------
if __name__ == '__main__':
    args = get_parser()
    for exp in args.experiments:
        tag = exp["tag"]
        for epoch in exp["eval_ckp"]:
            try:
                test_model(
                    args=args,
                    tag=tag,
                    source_type=exp["source"],
                    condition_type=exp["condition"],
                    checkpoint_dir=os.path.join(args.save_path, tag, "checkpoints"),
                    epoch=epoch,
                    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
                )
            except FileNotFoundError as e:
                print(f"Skipping {tag} epoch {epoch}: {e}")
                continue