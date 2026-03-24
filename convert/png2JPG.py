import os
from PIL import Image

def convert_png_to_jpg(folder_path):
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(".png"):
            png_path = os.path.join(folder_path, filename)
            jpg_filename = os.path.splitext(filename)[0] + ".JPG"
            jpg_path = os.path.join(folder_path+'_', jpg_filename)

            with Image.open(png_path) as img:
                rgb_img = img.convert("RGB")  # 去掉透明通道
                rgb_img.save(jpg_path, "JPEG")

            print(f"Converted: {png_path} -> {jpg_path}")

if __name__ == "__main__":
    folder = "/home/ubuntu/datasets_optuna/FSGS/Ujikintoki/train"
    os.makedirs(folder+'_')
    convert_png_to_jpg(folder)