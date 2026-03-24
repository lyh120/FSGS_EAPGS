import os
import glob
import numpy as np
from PIL import Image
import torch

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images

def save_depth_png16(depth_seq, out_dir):
    # depth_seq: [S, H, W] float32
    dmin = float(depth_seq.min())
    dmax = float(depth_seq.max())
    # 统一全序列归一化，保证帧间一致可视化
    depth_norm = (depth_seq - dmin) / (dmax - dmin + 1e-8)
    depth_u16 = (depth_norm * 65535.0).astype(np.uint16)
    for i in range(depth_u16.shape[0]):
        Image.fromarray(depth_u16[i]).save(os.path.join(out_dir, f"depth_{i+1:04d}.png"))

def main(image_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    # 1) 收集图片（按文件名排序）
    exts = ["*.png", "*.jpg","*.JPG", "*.jpeg", "*.bmp", "*.tif", "*.tiff"]
    image_paths = []
    for e in exts:
        image_paths += glob.glob(os.path.join(image_dir, e))
    image_paths = sorted(image_paths)
    if not image_paths:
        raise RuntimeError("未找到图片，请检查路径/后缀。")

    # 2) 设备与精度
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    else:
        dtype = torch.float32

    # 3) 加载模型与图片
    # 第一次会自动从 Hugging Face 下载权重
    model = VGGT.from_pretrained("facebook/VGGT-1B").to(device).eval()
    images = load_and_preprocess_images(image_paths).to(device)

    # 4) 推理（一次性多帧，保证帧间一致）
    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=(device=="cuda"), dtype=dtype):
            preds = model(images)

    # 5) 取出深度并保存
    # depth: [B, S, H, W, 1]  -> [S, H, W]
    depth = preds["depth"].squeeze(0).squeeze(-1).float().cpu().numpy()

    # 保存原始深度（npy）
    for i in range(depth.shape[0]):
        np.save(os.path.join(out_dir, f"{i+1:04d}.npy"), depth[i])

    # 保存统一尺度的16位PNG（可视化一致）
    save_depth_png16(depth, out_dir)

    print(f"Done. Saved {depth.shape[0]} frames to: {out_dir}")

if __name__ == "__main__":
    # 修改这两个路径
    image_dir = "/home/liuyuhao/FSGS/dataset/Cupcake/train"
    out_dir = "/home/liuyuhao/FSGS/dataset/Cupcake/depth_maps"
    main(image_dir, out_dir)