from PIL import Image
import os

def convert_jpg_to_png(input_folder, output_folder):
    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 遍历输入文件夹中的所有图片文件
    for filename in os.listdir(input_folder):
        if filename.endswith(".jpg"):
            # 构建输入图片路径和输出图片路径
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, os.path.splitext(filename)[0] + ".png")

            # 打开 JPEG 图片
            img = Image.open(input_path)

            # 保存为 PNG 图片
            img.save(output_path, format="PNG")

            print(f"Image {filename} converted and saved to {output_path}")

# 用法示例
input_folder = "./test_imgs/RoadScene/ir/"
output_folder = "./test_imgs/RoadScene_png/ir/"
convert_jpg_to_png(input_folder, output_folder)
input_folder = "./test_imgs/RoadScene/vi/"
output_folder = "./test_imgs/RoadScene_png/vi/"
convert_jpg_to_png(input_folder, output_folder)