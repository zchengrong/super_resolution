from io import BytesIO

import cv2
from PIL import Image
from minio import Minio

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
image_url = "test/1705570348_0.png"
image_data = minio_client.get_object(image_url.split("/", 1)[0], image_url.split("/", 1)[1])
image_bytes = image_data.read()
lq = Image.open(BytesIO(image_bytes))
lq.show()
print("ok")
print(64 * 64
      )
