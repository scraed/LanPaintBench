import os
import time  # 导入time模块用于计时
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from typing import Dict, Tuple
from tqdm import tqdm
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np
from pipeline_lib_imagenet import same_seeds
import lpips
import torchmetrics
from torchmetrics.image.fid import FrechetInceptionDistance
import torch
import torch.nn as nn
from PIL import Image
import io
from torch.utils.data import Dataset
from diffusers import DDPMScheduler, DDIMScheduler, DPMSolverMultistepScheduler, EulerDiscreteScheduler, LMSDiscreteScheduler
from pipeline_lib_imagenet import *
from diffusers import UNet2DModel, DDIMScheduler
from diffusers.models.unets.unet_2d import UNet2DOutput
from diffusers.models.unets.unet_2d_condition import UNet2DConditionOutput
import numpy as np
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset

from guided_diffusion.unet import SuperResModel, UNetModel, EncoderUNetModel
import argparse
import types
same_seeds(0)

def inference_batch(func_name,func,vqvae,y,mask,n_steps):
    samples = func(
        y=y,
        mask=mask,
        generator=torch.Generator(device='cpu').manual_seed(0), # Use a separate
        num_inference_steps=n_steps,
        output_type = "numpy",
    ).images
    samples_replace = torch.tensor(samples).to(torch_device)
    if vqvae is not None:
        with torch.no_grad():
            image = vqvae.decode(samples_replace)
    else:
        image = samples_replace
    return image 

class CustomImageDataset(Dataset):
    def __init__(self, dataset,mask_type='half',mask_size=64):
        self.dataset = dataset
        self.mask_type = mask_type
        self.mask_size = mask_size
        if mask_type == 'checkerboard':
            # print('Creating checkerboard mask')
            self.mask  = create_checkerboard_mask(mask_size,mask_size,int(mask_size/16))
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image = self.dataset[idx]['image']
        # get image file name
        image_path = self.dataset[idx]['label']
        # print(image_path)
        image = np.array(image)
        # image = Image.open(image_path)
        image = (image/127.5) - 1.0
        image = transforms.ToTensor()(image)
        # print("*"*20)
        # print('mask type',self.mask_type)
        if self.mask_type == 'half':
            # create a mask size with 3,mask_size,mask_size
            mask = torch.ones(3,self.mask_size,self.mask_size)
            # mask = torch.ones_like(image)
            mask[:, :, self.mask_size//2:] = 0
        elif self.mask_type == 'box':
            mask = torch.ones(3,self.mask_size,self.mask_size)
            mask[:, self.mask_size//4:3*self.mask_size//4, self.mask_size//4:3*self.mask_size//4] = 0
        elif self.mask_type == 'outpaint':
            mask = torch.zeros(3,self.mask_size,self.mask_size)
            mask[:, self.mask_size//4:3*self.mask_size//4, self.mask_size//4:3*self.mask_size//4] = 1
        elif self.mask_type == 'checkerboard':
            # print('I am creating mask')
            mask = torch.tensor(self.mask).unsqueeze(0)
        # convert image to float type
        image = image.float()
        # convert mask to int type
        # print('mask',mask)
        mask = mask.int()
        return image,mask,image_path

def psnr(img1, img2):
    # Ensure the images are in the range [0, 1]
    img1 = (img1 + 1) / 2  # Rescale from [-1, 1] to [0, 1]
    img2 = (img2 + 1) / 2  # Rescale from [-1, 1] to [0, 1]

    # Calculate the Mean Squared Error (MSE)
    mse = F.mse_loss(img1, img2)

    # Calculate the PSNR
    max_pixel = 1.0  # Since the images are in the range [0, 1]
    psnr_value = 20 * torch.log10(max_pixel / torch.sqrt(mse))

    return psnr_value

def create_checkerboard_mask(rows, cols, grid_size):
    # Initialize an empty mask
    mask = np.zeros((rows, cols), dtype=int)
    # Loop through the rows
    for i in range(rows):
        # Determine if the row is odd or even
        if i // grid_size % 2 == 0:
            # Odd row pattern: start at n=0,2,4,...
            for n in range(0, cols // grid_size, 2):
                mask[i, n*grid_size:(n+1)*grid_size] = 1
        else:
            # Even row pattern: start at n=1,3,5,...
            for n in range(1, cols // grid_size, 2):
                mask[i, n*grid_size:(n+1)*grid_size] = 1
                
    return mask

parser = argparse.ArgumentParser(description="Select a pipeline method to run.")
parser.add_argument(
    '--method',
    type=str,
    nargs='+',  # Allow one or more values
    required=True,
    help="Name of the pipeline method to run."
)
parser.add_argument(
    '--suffix',
    type=str,
    default='',
    help="rename save folder."
)
parser.add_argument(
    '--batch_size',
    type=int,
    default=16,
    help="Batch size for processing images."
)
parser.add_argument(
    '--num_imgs',
    type=int,
    default=1000,
    help="number of images."
)
parser.add_argument(
    '--record_time',
    action='store_true',  
    help="whether to record processing time."
)
parser.add_argument(
    '--metrics',
    action='store_true',  
    help="whether to calculate metrics."
)
parser.add_argument(
    '--mask_type',
    type=str,
    nargs='+',  # Allow one or more values
    default='sr',
    help="Type of mask to use."
)
parser.add_argument(
    '--mask_grid_size',
    type=int,
    default=16,
    help="Size of the grid for checkerboard mask."
)
parser.add_argument(
    '--chara_lamb',
    type=int,
    default=8,
    help="Lambda value for Chara."
)
parser.add_argument(
    '--chara_beta',
    type=float,
    default=1.,
    help="Lambda value for Chara."
)
parser.add_argument(
    '--ite_step_size',
    type=float,
    default=0.03,
    help="Step size for iterative method."
)
parser.add_argument(
    '--uld_alpha',
    type=float,
    default=0.25,
    help="Step size for iterative method."
)
parser.add_argument(
    '--param_m',
    type=float,
    default=0.0,
    help="power of (1-abt)"
)
parser.add_argument(
    '--uld_friction',
    type=float,
    default=8,
    help="Step size for iterative method."
)
parser.add_argument(
    '--sample_steps',
    type=int,
    default=20,
    help="Number of sample steps."
)
parser.add_argument(
    '--validate',
    action='store_true',  
    help="whether to record processing time."
)
args = parser.parse_args()

# set to cuda
torch_device = "cuda" if torch.cuda.is_available() else "cpu"
seed = 3

beta_start=0.0001
beta_end = 0.02
num_train_timesteps = 1000
noise_scheduler = DDPMScheduler(beta_start=beta_start,beta_end = beta_end, num_train_timesteps=num_train_timesteps, beta_schedule='linear')
dpm_scheduler = DPMSolverMultistepScheduler.from_config(noise_scheduler.config)
ddim_scheduler = DDIMScheduler.from_config(noise_scheduler.config)
euler_scheduler = EulerDiscreteScheduler.from_config(noise_scheduler.config)
lms_scheduler = LMSDiscreteScheduler.from_config(noise_scheduler.config)

batch_size = args.batch_size
mask_type = args.mask_type

n_steps = args.sample_steps
uld_friction = args.uld_friction
uld_alpha = args.uld_alpha
chara_lamb = args.chara_lamb 
ite_step_size = args.ite_step_size 
chara_beta = args.chara_beta
param_m = args.param_m

image_size = 256
mask_grid_size = args.mask_grid_size
noise_schedule='linear' 
num_channels=256 
num_head_channels=64 
num_res_blocks=2 
resblock_updown=True

learn_sigma=True
class_cond=False
use_checkpoint=False
attention_resolutions="32,16,8"
use_scale_shift_norm = True
use_fp16 = False

def create_model(
    image_size,
    num_channels,
    num_res_blocks,
    channel_mult="",
    learn_sigma=False,
    class_cond=False,
    use_checkpoint=False,
    attention_resolutions="16",
    num_heads=1,
    num_head_channels=-1,
    num_heads_upsample=-1,
    use_scale_shift_norm=False,
    dropout=0,
    resblock_updown=False,
    use_fp16=False,
    use_new_attention_order=False,
):
    if channel_mult == "":
        if image_size == 512:
            channel_mult = (0.5, 1, 1, 2, 2, 4, 4)
        elif image_size == 256:
            channel_mult = (1, 1, 2, 2, 4, 4)
        elif image_size == 128:
            channel_mult = (1, 1, 2, 3, 4)
        elif image_size == 64:
            channel_mult = (1, 2, 3, 4)
        else:
            raise ValueError(f"unsupported image size: {image_size}")
    else:
        channel_mult = tuple(int(ch_mult) for ch_mult in channel_mult.split(","))

    attention_ds = []
    for res in attention_resolutions.split(","):
        attention_ds.append(image_size // int(res))

    return UNetModel(
        image_size=image_size,
        in_channels=3,
        model_channels=num_channels,
        out_channels=(3 if not learn_sigma else 6),
        num_res_blocks=num_res_blocks,
        attention_resolutions=tuple(attention_ds),
        dropout=dropout,
        channel_mult=channel_mult,
        num_classes=(NUM_CLASSES if class_cond else None),
        use_checkpoint=use_checkpoint,
        use_fp16=use_fp16,
        num_heads=num_heads,
        num_head_channels=num_head_channels,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
        resblock_updown=resblock_updown,
        use_new_attention_order=use_new_attention_order,
    )

model = create_model(
        image_size,
        num_channels,
        num_res_blocks,
        # channel_mult=channel_mult,
        learn_sigma=learn_sigma,
        class_cond=class_cond,
        use_checkpoint=use_checkpoint,
        attention_resolutions=attention_resolutions,
        # num_heads=num_heads,
        num_head_channels=num_head_channels,
        # num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
        # dropout=dropout,
        resblock_updown=resblock_updown,
        use_fp16=use_fp16,
        # use_new_attention_order=use_new_attention_order,
    )

model.load_state_dict(torch.load('./checkpoints/256x256_diffusion_uncond.pt'))
if use_fp16:
    model.convert_to_fp16()
model.eval()

class score_net(UNet2DModel):
    def __init__(self, model, scheduler, *args, **kwargs):
        super(score_net, self).__init__(*args, **kwargs)
        self.scheduler = scheduler
        self.unet_model = model

    def forward(
            self,
            x: torch.Tensor,
            timestep: Union[torch.Tensor, float, int],
            encoder_hidden_states: Optional[torch.Tensor] = None,
            return_dict: bool = True,
        ) -> Union[UNet2DConditionOutput, UNet2DOutput, Tuple]:

        eps = self.unet_model(x, timestep)
        eps = eps[:,:3,:,:]
        
        if not return_dict:
            return (eps,)
        return UNet2DOutput(sample=eps)
    
unet = score_net(model, dpm_scheduler,
    sample_size=(256),  # the number of dimensions of the input samples
    in_channels=3,
    out_channels=3,  # the number of output channels
    layers_per_block=1,  # how many ResNet layers to use per UNet block
    norm_num_groups= 1,
    block_out_channels=(1,),  # the number of output channels for each UNet block,
    attention_head_dim=1,  # the number of dimensions per head in the attention block
    down_block_types=("DownBlock2D",),  # the types of downsampling blocks to use
    up_block_types=( "AttnUpBlock2D",),)  # the types of upsampling blocks to use)

def cal_eps(self, image_rescale, t):
    return self.__call__(image_rescale, t.expand(image_rescale.shape[0]).to(image_rescale.device)).sample  # Access instance attributes

# Bind as a method to the specific instance
unet.cal_eps = types.MethodType(cal_eps, unet)

uld_chara_pipeline_10_sam = ULDCharaPipeline_Sampler(
    unet=unet,
    scheduler=euler_scheduler,
    step_size = ite_step_size,
    n_steps = 10,
    friction = uld_friction,
    m = param_m,
    alpha = uld_alpha,
    chara_lamb = chara_lamb,
    chara_beta = chara_beta
).to(torch_device)

uld_chara_pipeline_5_sam = ULDCharaPipeline_Sampler(
    unet=unet,
    scheduler=euler_scheduler,
    step_size = ite_step_size,
    n_steps = 5,
    friction = uld_friction,
    m = param_m,
    alpha = uld_alpha,
    chara_lamb = chara_lamb,
    chara_beta = chara_beta
).to(torch_device)

uld_pipeline_10_sam = ULDPipeline_Sampler(
    unet=unet,
    scheduler=euler_scheduler,
    step_size = ite_step_size,
    n_steps = 10,
    friction = uld_friction,
    m = param_m,
    alpha = uld_alpha
).to(torch_device)

uld_pipeline_5_sam = ULDPipeline_Sampler(
    unet=unet,
    scheduler=euler_scheduler,
    step_size = ite_step_size,
    n_steps = 5,
    friction = uld_friction,
    m = param_m,
    alpha = uld_alpha
).to(torch_device)

save_folders = "./results/"
os.makedirs(save_folders, exist_ok=True)


pipeline_func_lib  = {

    'LanPaint-10': uld_chara_pipeline_10_sam,
    'LanPaint-5': uld_chara_pipeline_5_sam,
}


all_results = {}

data_files = {'test': 'data/test-*.parquet'}
print('Loading the dataset')
ds = load_dataset("benjamin-paine/imagenet-1k-256x256",data_files=data_files,verification_mode="no_checks")['test'].select(range(args.num_imgs))

for mask_type in args.mask_type:
    print(f"Processing mask type: {mask_type}")
    custom_dataset = CustomImageDataset(ds,mask_type=mask_type,mask_size=256)

    
    lpips_func = lpips.LPIPS(net='alex').to(torch_device)
    ssim_metric = torchmetrics.functional.structural_similarity_index_measure

    for method_name in args.method:     

        assert method_name in pipeline_func_lib, f"Method {method_name} not found in pipeline_func_lib"

        dataloader = DataLoader(custom_dataset, batch_size=args.batch_size, shuffle=False)
        
        fid = FrechetInceptionDistance(feature=2048).cuda()
        lpips_result_method = 0
        psnr_result_method = 0
        ssim_result_method = 0
        total_batches = 0
        total_samples = 0
        total_inference_time = 0.0  
        peak_memory = 0  

        save_folders_pipeline = os.path.join(save_folders, f'{method_name}_{args.suffix}', 
                                            f"lamb{chara_lamb}_itestepsize_{ite_step_size}_alpha_{uld_alpha}_friction_{uld_friction}_{param_m}",
                                            f"{n_steps}steps", str(mask_type))
        os.makedirs(save_folders_pipeline, exist_ok=True)
        torch.cuda.empty_cache()
        print(f"Processing images using {method_name} mask type {mask_type}")
        

        for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Processing {method_name}")):
            torch.cuda.reset_peak_memory_stats(torch_device)
            images, masks, _ = batch
            images = images.to(torch_device)
            masks = masks.to(torch_device)
            
            with torch.no_grad():

                torch.cuda.synchronize()  
                start_time = time.time()
                current_peak_before = torch.cuda.max_memory_allocated(torch_device) 

                decode_img = inference_batch(method_name, pipeline_func_lib[method_name], None, images, masks, n_steps)
                current_peak_after = torch.cuda.max_memory_allocated(torch_device)

                torch.cuda.synchronize()  
                batch_time = time.time() - start_time
                batch_peak = current_peak_after - current_peak_before
                total_inference_time += batch_time

                if batch_peak > peak_memory:
                    peak_memory = batch_peak
                

                lpips_result = torch.mean(lpips_func(decode_img, images))
                psnr_result = psnr(decode_img, images)
                ssim_result = torch.mean(ssim_metric(decode_img, images))
                
                lpips_result_method += lpips_result.item()
                psnr_result_method += psnr_result.item()
                ssim_result_method += ssim_result.item()
                total_batches += 1
                total_samples += images.size(0)
                
 
                real_images = (images + 1) * 127.5
                real_images = real_images.clamp(0, 255).type(torch.uint8)
                fake_images = (decode_img + 1) * 127.5
                fake_images = fake_images.clamp(0, 255).type(torch.uint8)
                fid.update(real_images, real=True)
                fid.update(fake_images, real=False)
                

                for img_idx in range(decode_img.shape[0]):
                    image_processed = decode_img[img_idx].cpu().permute(1, 2, 0)
                    image_processed = (image_processed + 1.0) * 127.5
                    image_processed = image_processed.clamp(0, 255).numpy().astype(np.uint8)
                    image_pil = Image.fromarray(image_processed)
                    image_pil.save(os.path.join(save_folders_pipeline, f"image_{batch_idx * args.batch_size + img_idx}.png"))
                
                del images, masks, decode_img
        

        avg_lpips = lpips_result_method / total_batches
        avg_psnr = psnr_result_method / total_batches
        avg_ssim = ssim_result_method / total_batches
        fid_score = fid.compute().item()
        

        avg_time_per_sample = total_inference_time / total_samples
        peak_memory_per_image = peak_memory / (1024 ** 2) / args.batch_size

        all_results.setdefault(mask_type, {})[method_name] = {
            'LPIPS': avg_lpips,
            'PSNR': avg_psnr,
            'SSIM': avg_ssim,
            'FID': fid_score,
            'Time': avg_time_per_sample,
            'Peak_Memory': peak_memory_per_image  
        }
        
        print(f"Method: {method_name}, Mask: {mask_type}")
        print(f"LPIPS: {avg_lpips:.4f}, PSNR: {avg_psnr:.4f}, SSIM: {avg_ssim:.4f}, FID: {fid_score:.4f}, Time: {avg_time_per_sample:.4f} sec/sample, Peak Memory: {peak_memory_per_image:.2f} MB/image")
        torch.cuda.empty_cache()

        import csv
        csv_filename = os.path.join(save_folders, f"imagenet_metrics_{args.mask_type}_{n_steps}steps_{chara_lamb}lamb_{ite_step_size}stepsize_{uld_friction}fric_{uld_alpha}alpha_{param_m}m.csv")

        with open(csv_filename, 'w', newline='') as csvfile:
            fieldnames = ['Mask Type', 'Method', 'LPIPS', 'PSNR', 'SSIM', 'FID', 'Time (s/sample)', 'Peak Memory (MB/image)']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for mask_type, methods in all_results.items():
                for method_name, metrics in methods.items():
                    writer.writerow({
                        'Mask Type': mask_type,
                        'Method': method_name,
                        'LPIPS': metrics['LPIPS'],
                        'PSNR': metrics['PSNR'],
                        'SSIM': metrics['SSIM'],
                        'FID': metrics['FID'],
                        'Time (s/sample)': metrics['Time'],
                        'Peak Memory (MB/image)': metrics['Peak_Memory']
                    })

        print(f"All metrics saved to {csv_filename}")