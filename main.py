import io
import math
import time
from io import BytesIO
from typing import Union, List, Tuple

import cv2
import einops
import numpy as np
import pytorch_lightning as pl
import uvicorn
from PIL import Image
from fastapi import FastAPI
from minio import Minio

from omegaconf import OmegaConf
import torch
from pydantic import BaseModel

from env.config import *
from ldm.xformers_state import disable_xformers
from model.ccsr_stage1 import ControlLDM
from model.q_sampler import SpacedSampler
from utils.common import instantiate_from_config, load_state_dict
from utils.generate_uuid import generate_uuid
from utils.image import auto_resize


class SRItem(BaseModel):
    sr_input: list
    sr_xn: int
    input_size: list


def check_device(device):
    if device == "cuda":
        # check if CUDA is available
        if not torch.cuda.is_available():
            print("CUDA not available because the current PyTorch install was not "
                  "built with CUDA enabled.")
            device = "cpu"
    else:
        # xformers only support CUDA. Disable xformers when using cpu or mps.
        disable_xformers()
        if device == "mps":
            # check if MPS is available
            if not torch.backends.mps.is_available():
                if not torch.backends.mps.is_built():
                    print("MPS not available because the current PyTorch install was not "
                          "built with MPS enabled.")
                    device = "cpu"
                else:
                    print("MPS not available because the current MacOS version is not 12.3+ "
                          "and/or you do not have an MPS-enabled device on this machine.")
                    device = "cpu"
    print(f'using device {device}')
    return device


def process(
        model: ControlLDM,
        control_imgs: List[np.ndarray],
        steps: int,
        t_max: float,
        t_min: float,
        strength: float,
        color_fix_type: str,
        tiled: bool,
        tile_size: int,
        tile_stride: int
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
        tiled (bool): If specified, a patch-based sampling strategy will be used for sampling.
        tile_size (int): Size of patch.
        tile_stride (int): Stride of sliding patch.

    Returns:
        preds (List[np.ndarray]): Restoration results (HWC, RGB, range in [0, 255]).
    """

    print("Start SR Task ")
    n_samples = len(control_imgs)
    sampler = SpacedSampler(model, var_type="fixed_small")
    control = torch.tensor(np.stack(control_imgs) / 255.0, dtype=torch.float32, device=model.device).clamp_(0, 1)
    control = einops.rearrange(control, "n h w c -> n c h w").contiguous()

    model.control_scales = [strength] * 13

    height, width = control.size(-2), control.size(-1)
    shape = (n_samples, 4, height // 8, width // 8)
    x_T = torch.randn(shape, device=model.device, dtype=torch.float32)
    if not tiled:
        # samples = sampler.sample_ccsr_stage1(
        #     steps=steps, t_max=t_max, shape=shape, cond_img=control,
        #     positive_prompt="", negative_prompt="", x_T=x_T,
        #     cfg_scale=1.0, color_fix_type=color_fix_type
        # )
        samples = sampler.sample_ccsr(
            steps=steps, t_max=t_max, t_min=t_min, shape=shape, cond_img=control,
            positive_prompt="", negative_prompt="", x_T=x_T,
            cfg_scale=1.0, color_fix_type=color_fix_type
        )
    else:
        samples = sampler.sample_with_mixdiff_ccsr(
            tile_size=tile_size, tile_stride=tile_stride,
            steps=steps, t_max=t_max, t_min=t_min, shape=shape, cond_img=control,
            positive_prompt="", negative_prompt="", x_T=x_T,
            cfg_scale=1.0, color_fix_type=color_fix_type
        )

    x_samples = samples.clamp(0, 1)
    x_samples = (einops.rearrange(x_samples, "b c h w -> b h w c") * 255).cpu().numpy().clip(0, 255).astype(np.uint8)
    control = (einops.rearrange(control, "b c h w -> b h w c") * 255).cpu().numpy().clip(0, 255).astype(np.uint8)

    preds = [x_samples[i] for i in range(n_samples)]
    print("Finished SR Tasks")
    return preds


pl.seed_everything(233)
model: ControlLDM = instantiate_from_config(OmegaConf.load(config))
load_state_dict(model, torch.load(ckpt, map_location="cpu"), strict=True)
model.freeze()
model.to(check_device(device=device))

app = FastAPI()
MINIO_IP = "www.minio.aida.com.hk"
MINIO_PORT = 9000
MINIO_ACCESS = 'vXKFLSJkYeEq2DrSZvkB'
MINIO_SECRET = 'uKTZT3x7C43WvPN9QTc99DiRkwddWZrG9Uh3JVlR'
MINIO_SECURE = True
minio_client = Minio(
    f"{MINIO_IP}:{MINIO_PORT}",
    access_key=MINIO_ACCESS,
    secret_key=MINIO_SECRET,
    secure=MINIO_SECURE)


def read_image(image_url):
    image_data = minio_client.get_object(image_url.split("/", 1)[0], image_url.split("/", 1)[1])
    image_bytes = image_data.read()
    return BytesIO(image_bytes)


def put_image(bucket, file_name, obj):
    img = Image.fromarray(np.array(obj))
    image_data = io.BytesIO()
    img.save(image_data, format='PNG')
    image_data.seek(0)
    image_bytes = image_data.read()
    url = f"{bucket}/{minio_client.put_object(bucket_name=bucket, data=io.BytesIO(image_bytes), object_name=f'{file_name}.png', length=len(image_bytes), content_type='image/png').object_name}"
    return url


def main(image_url, sr_xn):
    image_obj = read_image(image_url)
    lq = Image.open(image_obj).convert("RGB")

    # 判断sr倍数
    if sr_xn == 2:
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
    # x = np.array(x)[:, :, ::-1]  # 防止颜色反转
    x = np.array(x)
    preds = process(
        model,
        [x],
        steps=steps,
        t_max=0.6667, t_min=0.3333,
        strength=1,
        color_fix_type="adain",
        tiled=False,
        tile_size=512, tile_stride=256
    )
    sr_image_url = put_image("test", time.time(), preds[0])
    return sr_image_url


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
        preds = process(model, [x], steps=steps, t_max=t_max, t_min=t_min, strength=1, color_fix_type=color_fix_type, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        sr_image_url = put_image(bucket=sr_bucket, file_name=generate_uuid(), obj=preds[0])
        sr_image_url_list.append(sr_image_url)
    print(f"sr runtime is {time.time() - start_time}")
    return {"output_url_list": sr_image_url_list}


if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=4562)
    # image_url = "test/1705570348_0.png"
    # sr_xn = 2
    # sr_result_url = main(image_url, sr_xn)
    # print(sr_result_url)
    # image = read_image(sr_result_url)
    # Image.open(image).convert("RGB").show()
