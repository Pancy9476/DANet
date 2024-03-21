from PIL import Image
import os

def crop_images_in_folder(input_folder, output_folder):
    # 遍历输入文件夹中的所有图片文件
    for filename in os.listdir(input_folder):
        if filename.endswith((".jpg", ".png", ".jpeg")):
            # 构建输入图片路径和输出图片路径
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            # 打开图片
            img = Image.open(input_path)

            # 获取原始图片的宽和高
            width, height = img.size

            # 计算裁剪后的宽度和高度，使其为16的倍数
            new_width = width - (width % 16)
            new_height = height - (height % 16)

            # 裁剪图片
            cropped_img = img.crop((0, 0, new_width, new_height))

            # 确保输出文件夹存在
            # os.makedirs(output_folder, exist_ok=True)

            # 保存裁剪后的图片
            cropped_img.save(output_path)

            print(f"Image {filename} cropped and saved to {output_path}")

# 用法
input_folder = "need_crop_image/TNO/vi"
output_folder = "test_img/TNO/vi"
crop_images_in_folder(input_folder, output_folder)
input_folder = "need_crop_image/TNO/ir"
output_folder = "test_img/TNO/ir"
crop_images_in_folder(input_folder, output_folder)