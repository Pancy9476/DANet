import os
from PIL import Image

def cp_image_sizes(folder):
    # 获取文件夹中的所有文件名
    files = os.listdir(folder)

    # 遍历共有的文件名，并比较图片尺寸
    for filename in files:
        path = os.path.join(folder, filename)

        # 打开图片并获取尺寸
        try:
            img = Image.open(path)

            size = img.size

            # img_resized = img.crop((0, 0, 960, 768))
            img_resized = img.crop((64, 64, 896, 704))

            # 构建新的文件名
            new_filename = filename.split('.')[0] + ".png"  # 可以根据需要修改文件格式
            new_file_path = os.path.join(folder, new_filename)

            img_resized.save(path)
            os.rename(path, new_file_path)
            print(f"Resized and cropped image '{filename}' from {size} to 832,640")

        except Exception as e:
            print(f"Error processing image '{filename}': {str(e)}")

if __name__ == "__main__":
    folder1 = "test_img/LLVIP/ir"
    folder2 = "test_img/LLVIP/vi"

    cp_image_sizes(folder1)
    cp_image_sizes(folder2)