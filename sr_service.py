import io
import math
from io import BytesIO
from typing import List

import einops
from omegaconf import OmegaConf
import pytorch_lightning as pl
from model.q_sampler import SpacedSampler
import numpy as np
import torch
from PIL import Image
from minio import Minio
from model.ccsr_stage1 import ControlLDM
from utils.common import instantiate_from_config, load_state_dict
from env.config import *
from ldm.xformers_state import disable_xformers
from functools import partial
from utils.generate_uuid import generate_uuid


def auto_resize(img: Image.Image, size: int) -> Image.Image:
    short_edge = min(img.size)
    if short_edge < size:
        r = size / short_edge
        img = img.resize(
            tuple(math.ceil(x * r) for x in img.size), Image.BICUBIC
        )
    else:
        # make a deep copy of this image for safety
        img = img.copy()
    return img


class SuperResolution:
    def __init__(self):
        self.minio_client = Minio(
            f"{MINIO_IP}:{MINIO_PORT}",
            access_key=MINIO_ACCESS,
            secret_key=MINIO_SECRET,
            secure=MINIO_SECURE)
        pl.seed_everything(233)
        self.model: ControlLDM = instantiate_from_config(OmegaConf.load(config))
        load_state_dict(self.model, torch.load(ckpt, map_location="cpu"), strict=True)
        self.model.freeze()
        self.model.to(self.check_device(device=device))

    @staticmethod
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

    def read_image(self, image_url):
        image_data = self.minio_client.get_object(image_url.split("/", 1)[0], image_url.split("/", 1)[1])
        image_bytes = image_data.read()
        return BytesIO(image_bytes)

    def put_image(self, bucket, file_name, obj):
        if isinstance(obj, Image.Image):
            image_data = io.BytesIO()
            obj.save(image_data, format='PNG')
            image_data.seek(0)
            image_bytes = image_data.read()
            url = f"{bucket}/{self.minio_client.put_object(bucket_name=bucket, data=io.BytesIO(image_bytes), object_name=f'{file_name}.png', length=len(image_bytes), content_type='image/png').object_name}"
            return url
        else:
            img = Image.fromarray(np.array(obj))
            image_data = io.BytesIO()
            img.save(image_data, format='PNG')
            image_data.seek(0)
            image_bytes = image_data.read()
            url = f"{bucket}/{self.minio_client.put_object(bucket_name=bucket, data=io.BytesIO(image_bytes), object_name=f'{file_name}.png', length=len(image_bytes), content_type='image/png').object_name}"
            return url

    def pre_process(self, image_url, sr_xn):
        image_obj = self.read_image(image_url)
        lq = Image.open(image_obj).convert("RGB")
        original_width, original_height = lq.size
        if original_height * sr_xn > 2048 and original_width * sr_xn > 2048:
            return True, None

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
        return False, np.array(x)

    def process(self, control_imgs: List[np.ndarray]):
        n_samples = len(control_imgs)
        sampler = SpacedSampler(self.model, var_type="fixed_small")
        control = torch.tensor(np.stack(control_imgs) / 255.0, dtype=torch.float32, device=self.model.device).clamp_(0, 1)
        control = einops.rearrange(control, "n h w c -> n c h w").contiguous()

        self.model.control_scales = [strength] * 13

        height, width = control.size(-2), control.size(-1)
        shape = (n_samples, 4, height // 8, width // 8)
        x_T = torch.randn(shape, device=self.model.device, dtype=torch.float32)

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
                self.model._init_tiled_vae(encoder_tile_size=vae_encoder_tile_size,
                                           decoder_tile_size=vae_decoder_tile_size)
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
        x_samples = (einops.rearrange(x_samples, "b c h w -> b h w c") * 255).cpu().numpy().clip(0, 255).astype(
            np.uint8)

        preds = [x_samples[i] for i in range(n_samples)]
        url = self.put_image(sr_bucket, generate_uuid(), preds[0])
        return url


if __name__ == '__main__':
    image_url = "test/128_image/11.png"
    sr_xn = 4
    service = SuperResolution()


    def callback(data, result, error):
        if error:
            print(error)
            data.append(error)
        else:
            print(result)
            data.append(result)


    user_data = []
    _, sr_image = service.pre_process(image_url, sr_xn)
    if _:
        print("image size error")
    else:
        print(service.process([sr_image]))
