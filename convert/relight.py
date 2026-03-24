import argparse
from pathlib import Path

import numpy as np
from PIL import Image
import torch


def render_enhance(
        image,
        enable=True,
        brightness=0.08,
        gamma=0.85,
        contrast=1.25,
        saturation=1.35
):

    if not enable:
        return image

    image = torch.clamp(image, 0, 1)

    # (3, H, W) -> (H, W, 3)
    image = image.permute(1, 2, 0)

    # 1 brightness shift
    image = image + brightness

    # 2 gamma correction
    image = image.pow(gamma)

    # 3 contrast
    mean = image.mean(dim=(0, 1), keepdim=True)
    image = (image - mean) * contrast + mean

    # 4 saturation
    gray = image.mean(dim=2, keepdim=True)
    image = gray + (image - gray) * saturation

    image = torch.clamp(image, 0, 1)

    # (H, W, 3) -> (3, H, W)
    image = image.permute(2, 0, 1)

    return image


def _pil_to_tensor(img: Image.Image) -> torch.Tensor:
    arr = np.array(img).astype(np.float32) / 255.0
    # (H, W, 3) -> (3, H, W)
    return torch.from_numpy(arr).permute(2, 0, 1)


def _tensor_to_pil(t: torch.Tensor) -> Image.Image:
    t = torch.clamp(t, 0, 1)
    # (3, H, W) -> (H, W, 3)
    arr = (t.permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


def process_folder(input_dir: Path, output_dir: Path) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    exts = ("*.png", "*.jpg", "*.jpeg", "*.JPG", "*.JPEG")
    paths = []
    for ext in exts:
        paths.extend(sorted(input_dir.glob(ext)))

    count = 0
    for p in paths:
        img = Image.open(p).convert("RGB")
        t = _pil_to_tensor(img)
        out_t = render_enhance(
            t,
            enable=True,
            brightness=0.12,
            gamma=0.82,
            contrast=1.4,
            saturation=1.40,
        )
        out_img = _tensor_to_pil(out_t)
        out_path = output_dir / p.name
        out_img.save(out_path)
        count += 1
    return count


def main():
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    in_dir = Path("/home/ubuntu/mar24_")
    if not in_dir.exists():
        raise SystemExit(f"Input folder not found: {in_dir}")

    out_dir = in_dir.parent / f"relight_{in_dir.name}"
    count = process_folder(in_dir, out_dir)
    print(f"Done. Output in: {out_dir} ({count} files)")


if __name__ == "__main__":
    main()
