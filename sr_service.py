import io
import math
from io import BytesIO

import numpy as np
import torch
from PIL import Image
from minio import Minio

from env.config import *
from ldm.xformers_state import disable_xformers

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
    if isinstance(obj, Image.Image):
        image_data = io.BytesIO()
        obj.save(image_data, format='PNG')
        image_data.seek(0)
        image_bytes = image_data.read()
        url = f"{bucket}/{minio_client.put_object(bucket_name=bucket, data=io.BytesIO(image_bytes), object_name=f'{file_name}.png', length=len(image_bytes), content_type='image/png').object_name}"
        return url
    else:
        img = Image.fromarray(np.array(obj))
        image_data = io.BytesIO()
        img.save(image_data, format='PNG')
        image_data.seek(0)
        image_bytes = image_data.read()
        url = f"{bucket}/{minio_client.put_object(bucket_name=bucket, data=io.BytesIO(image_bytes), object_name=f'{file_name}.png', length=len(image_bytes), content_type='image/png').object_name}"
        return url


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


def pre_process(image_url, sr_xn):
    image_obj = read_image(image_url)
    lq = Image.open(image_obj).convert("RGB")
    original_width, original_height = lq.size
    if original_height * sr_xn and original_width * sr_xn:
        return False, None

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
    return True, np.array(x)
