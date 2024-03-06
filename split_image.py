import os

from PIL import Image


def split_image_4(image):
    width, height = image.size

    # 计算每块区域的大小
    block_width = int(width / 2)
    block_height = int(height / 2)

    # 创建四个空白图片对象
    top_left = Image.new('RGB', (block_width, block_height))
    top_right = Image.new('RGB', (block_width, block_height))
    bottom_left = Image.new('RGB', (block_width, block_height))
    bottom_right = Image.new('RGB', (block_width, block_height))

    # 复制原始图片到相应位置
    top_left.paste(image.crop((0, 0, block_width, block_height)))
    top_right.paste(image.crop((block_width, 0, width, block_height)))
    bottom_left.paste(image.crop((0, block_height, block_width, height)))
    bottom_right.paste(image.crop((block_width, block_height, width, height)))

    return [top_left, top_right, bottom_left, bottom_right]


def split_image_1(image, scale):
    width, height = image.size

    # 计算每块区域的大小
    block_width = int(width * scale)
    block_height = int(height * scale)

    # 创建四个空白图片对象
    new_image = Image.new('RGB', (block_width, block_height))

    # 复制原始图片到相应位置
    new_image.paste(image.crop((0, 0, block_width, block_height)))

    return new_image


# 读取图片并调用函数进行分割
# image = Image.open("83db1552-dabe-11ee-b52d-b48351119060_1.png")
# result = split_image_1(image, 0.75)

# 输出结果
# for i in range(len(result)):
#     result[i].save(f"new_image.jpg", "JPEG")

# result = split_image_4(image)
#
# # 输出结果
# for i in range(len(result)):
#     result[i].save(f"split_{i}.jpg", "JPEG")
if __name__ == '__main__':
    image_list = os.listdir('512_image')
    for image in image_list:
        image_obj = Image.open('512_image/' + image)
        result = split_image_1(image_obj, 0.25)
        result.save('128_image/' + image)
