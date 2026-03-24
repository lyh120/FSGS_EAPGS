import os
import shutil

import cv2
import numpy as np
from tqdm import tqdm


VALID_EXTENSIONS = (".png", ".jpg", ".jpeg")

# 1.0 表示完全套用目标亮度统计，越小越保守，通常更自然。
MATCH_STRENGTH = 1


def iter_image_files(image_folder):
    for file_name in sorted(os.listdir(image_folder)):
        if file_name.lower().endswith(VALID_EXTENSIONS):
            yield file_name


def compute_target_brightness_stats(image_folder):
    pixel_sum = 0.0
    pixel_sq_sum = 0.0
    pixel_count = 0

    for file_name in iter_image_files(image_folder):
        img = cv2.imread(os.path.join(image_folder, file_name))
        if img is None:
            continue

        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        lightness = lab[:, :, 0].astype(np.float32)
        pixel_sum += float(lightness.sum())
        pixel_sq_sum += float(np.square(lightness).sum())
        pixel_count += lightness.size

    if pixel_count == 0:
        raise ValueError(f"No readable images found in {image_folder}")

    mean = pixel_sum / pixel_count
    variance = max(pixel_sq_sum / pixel_count - mean * mean, 1.0)
    std = float(np.sqrt(variance))
    return {"mean": float(mean), "std": std}


def match_brightness_stats(source, target_stats, strength=1.0):
    source_f = source.astype(np.float32)
    source_mean = float(source_f.mean())
    source_std = float(source_f.std())

    if source_std < 1e-6:
        normalized = np.full_like(source_f, target_stats["mean"])
    else:
        normalized = (
            (source_f - source_mean) / source_std * target_stats["std"]
            + target_stats["mean"]
        )

    matched = (1.0 - strength) * source_f + strength * normalized
    return np.clip(matched, 0, 255).astype(np.uint8)


def protect_tones(original, matched, max_gain=18, max_loss=6):
    # 高亮区域不要被提得过曝，也不要被整体压暗到失去发光感。
    protected = matched.astype(np.int16)
    original_i = original.astype(np.int16)

    highlight_mask = original_i >= 220
    protected = np.where(
        highlight_mask,
        np.clip(protected, original_i - max_loss, original_i + max_gain),
        protected,
    )

    # 很暗的区域也做轻微保护，避免黑位被压死。
    shadow_mask = original_i <= 30
    protected = np.where(
        shadow_mask,
        np.clip(protected, original_i - 4, original_i + 8),
        protected,
    )

    return np.clip(protected, 0, 255).astype(np.uint8)


def apply_histogram_matching(render_img, target_stats, strength=MATCH_STRENGTH):
    lab = cv2.cvtColor(render_img, cv2.COLOR_BGR2LAB)

    lightness = lab[:, :, 0]
    matched = match_brightness_stats(lightness, target_stats, strength=strength)
    matched = protect_tones(lightness, matched)

    lab[:, :, 0] = matched
    result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return result


def clear_output_folder(output_folder):
    if not os.path.exists(output_folder):
        return

    for file_name in os.listdir(output_folder):
        file_path = os.path.join(output_folder, file_name)
        if os.path.isdir(file_path):
            shutil.rmtree(file_path)
        else:
            os.remove(file_path)


def main():
    real_folder = "real_images"
    #"real_image"
    #"cut_test"
    #"real_image"
#"cut_cho"
    render_folder = "final"
    #"renders\data_lol_v2\cupcake"
    #"LOL_v2_real_output/milkcookie"
    #"popcorn_render"
    #"LOL_v2_real_output/popcorn"
    output_folder = "outputs"

    os.makedirs(output_folder, exist_ok=True)
    clear_output_folder(output_folder)

    print("Calculating target brightness stats...")
    target_stats = compute_target_brightness_stats(real_folder)
    print(target_stats["mean"], target_stats["std"])
    np.save("target_stats.npy", target_stats)

    print("Processing render images...")
    for file_name in tqdm(list(iter_image_files(render_folder))):
        img_path = os.path.join(render_folder, file_name)
        img = cv2.imread(img_path)
        if img is None:
            continue

        result = apply_histogram_matching(img, target_stats)
        cv2.imwrite(os.path.join(output_folder, file_name), result)

    print("Done. Results are in outputs/")


if __name__ == "__main__":
    main()
