import math
import time
from typing import List, Tuple
import einops
import numpy as np
import pytorch_lightning as pl
from fastapi import FastAPI
from omegaconf import OmegaConf
from pydantic import BaseModel
from model.ccsr_stage1 import ControlLDM
from model.q_sampler import SpacedSampler
from sr_service import *
from utils.common import instantiate_from_config, load_state_dict
from utils.generate_uuid import generate_uuid
from utils.image import auto_resize


class SRItem(BaseModel):
    sr_input: list
    sr_xn: int
    input_size: list


def process(
        model: ControlLDM,
        control_imgs: List[np.ndarray],
        steps: int,
        t_max: float,
        t_min: float,
        strength: float,
        color_fix_type: str,
        tile_diffusion: bool,
        tile_diffusion_size: int,
        tile_diffusion_stride: int,
        tile_vae: bool,
        vae_decoder_tile_size: int,
        vae_encoder_tile_size: int
) -> Tuple[List[np.ndarray]]:
    """
    Apply CCSR model on a list of low-quality images.

    Args:
        model (ControlLDM): Model.
        control_imgs (List[np.ndarray]): A list of low-quality images (HWC, RGB, range in [0, 255]).
        steps (int): Sampling steps.
        t_max (float): The starting point of uniform sampling strategy.
        t_min (float): The ending point of uniform sampling strategy.
        strength (float): Control strength. Set to 1.0 during training.
        color_fix_type (str): Type of color correction for samples.
        tile_diffusion (bool): If specified, a patch-based sampling strategy for diffusion peocess will be used for sampling.
        tile_diffusion_size (int): Size of patch for diffusion peocess.
        tile_diffusion_stride (int): Stride of sliding patch for diffusion peocess.
        tile_vae (bool): If specified, a patch-based sampling strategy for the encoder and decoder in VAE will be used.
        vae_decoder_tile_size (int): Size of patch for VAE decoder.
        vae_encoder_tile_size (int): Size of patch for VAE encoder.

    Returns:
        preds (List[np.ndarray]): Restoration results (HWC, RGB, range in [0, 255]).
    """

    n_samples = len(control_imgs)
    sampler = SpacedSampler(model, var_type="fixed_small")
    control = torch.tensor(np.stack(control_imgs) / 255.0, dtype=torch.float32, device=model.device).clamp_(0, 1)
    control = einops.rearrange(control, "n h w c -> n c h w").contiguous()

    model.control_scales = [strength] * 13

    height, width = control.size(-2), control.size(-1)
    shape = (n_samples, 4, height // 8, width // 8)
    x_T = torch.randn(shape, device=model.device, dtype=torch.float32)

    if not tile_diffusion and not tile_vae:
        # samples = sampler.sample_ccsr_stage1(
        #     steps=steps, t_max=t_max, shape=shape, cond_img=control,
        #     positive_prompt="", negative_prompt="", x_T=x_T,
        #     cfg_scale=1.0,
        #     color_fix_type=color_fix_type
        # )
        samples = sampler.sample_ccsr(
            steps=steps, t_max=t_max, t_min=t_min, shape=shape, cond_img=control,
            positive_prompt="", negative_prompt="", x_T=x_T,
            cfg_scale=1.0,
            color_fix_type=color_fix_type
        )
    else:
        if tile_vae:
            model._init_tiled_vae(encoder_tile_size=vae_encoder_tile_size, decoder_tile_size=vae_decoder_tile_size)
        if tile_diffusion:
            samples = sampler.sample_with_tile_ccsr(
                tile_size=tile_diffusion_size, tile_stride=tile_diffusion_stride,
                steps=steps, t_max=t_max, t_min=t_min, shape=shape, cond_img=control,
                positive_prompt="", negative_prompt="", x_T=x_T,
                cfg_scale=1.0,
                color_fix_type=color_fix_type
            )
        else:
            samples = sampler.sample_ccsr(
                steps=steps, t_max=t_max, t_min=t_min, shape=shape, cond_img=control,
                positive_prompt="", negative_prompt="", x_T=x_T,
                cfg_scale=1.0,
                color_fix_type=color_fix_type
            )

    x_samples = samples.clamp(0, 1)
    x_samples = (einops.rearrange(x_samples, "b c h w -> b h w c") * 255).cpu().numpy().clip(0, 255).astype(np.uint8)

    preds = [x_samples[i] for i in range(n_samples)]

    return preds


pl.seed_everything(233)
model: ControlLDM = instantiate_from_config(OmegaConf.load(config))
load_state_dict(model, torch.load(ckpt, map_location="cpu"), strict=True)
model.freeze()
model.to(check_device(device=device))

app = FastAPI()


def main(image_url, sr_xn):
    # 判断sr倍数
    _, sr_image = pre_process(image_url.image_url, sr_xn)
    if _:
        preds = process(
            model, [sr_image], steps=steps,
            t_max=t_max, t_min=t_min,
            strength=1,
            color_fix_type=color_fix_type,
            tile_diffusion=tile_diffusion, tile_diffusion_size=tile_diffusion_size,
            tile_diffusion_stride=tile_diffusion_stride,
            tile_vae=tile_vae, vae_decoder_tile_size=vae_decoder_tile_size,
            vae_encoder_tile_size=vae_encoder_tile_size
        )
        sr_image_url = put_image("test", time.time(), preds[0])
        return sr_image_url
    else:
        return "image out size"


@app.post("/")
async def run(item: SRItem):
    sr_image_url_list = []
    for sr_input in item.sr_input:
        image_obj = read_image(sr_input)
        lq = Image.open(image_obj).convert("RGB")
        if item.sr_xn == 2:
            original_width, original_height = lq.size
            new_width = original_width // 2
            new_height = original_height // 2
            lq = lq.resize((new_width, new_height), Image.Resampling.LANCZOS)
        if sr_scale != 1:
            lq = lq.resize(tuple(math.ceil(x * sr_scale) for x in lq.size), Image.BICUBIC)
        if not tiled:
            lq_resized = auto_resize(lq, 512)
        else:
            lq_resized = auto_resize(lq, tile_size)
        x = lq_resized.resize(tuple(s // 64 * 64 for s in lq_resized.size), Image.LANCZOS)
        start_time = time.time()
        x = np.array(x)
        preds = process(
            model, [x], steps=steps,
            t_max=t_max, t_min=t_min,
            strength=1,
            color_fix_type=color_fix_type,
            tile_diffusion=tile_diffusion, tile_diffusion_size=tile_diffusion_size,
            tile_diffusion_stride=tile_diffusion_stride,
            tile_vae=tile_vae, vae_decoder_tile_size=vae_decoder_tile_size,
            vae_encoder_tile_size=vae_encoder_tile_size
        )
        sr_image_url = put_image(bucket=sr_bucket, file_name=generate_uuid(), obj=preds[0])
        sr_image_url_list.append(sr_image_url)
    print(f"sr runtime is {time.time() - start_time}")
    return {"output_url_list": sr_image_url_list}


if __name__ == '__main__':
    # uvicorn.run(app, host="0.0.0.0", port=4562)
    image_url = "test/1024_image/1709706120.1407204.png"
    # image_url = "test/256_image/21.png"
    sr_xn = 4
    stat_time = time.time()
    sr_result_url = main(image_url, sr_xn)
    print(time.time() - stat_time)
    print(sr_result_url)
    # image = read_image(sr_result_url)
    # Image.open(image).convert("RGB").show()
