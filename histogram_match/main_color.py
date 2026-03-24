import os
import shutil

import cv2
import numpy as np
from tqdm import tqdm


VALID_EXTENSIONS = (".png", ".jpg", ".jpeg")
COLOR_MATCH_STRENGTH = 0.5


def iter_image_files(image_folder):
    for file_name in sorted(os.listdir(image_folder)):
        if file_name.lower().endswith(VALID_EXTENSIONS):
            yield file_name


def clear_output_folder(output_folder):
    if not os.path.exists(output_folder):
        return

    for file_name in os.listdir(output_folder):
        file_path = os.path.join(output_folder, file_name)
        if os.path.isdir(file_path):
            shutil.rmtree(file_path)
        else:
            os.remove(file_path)


def compute_histogram(channel):
    hist = cv2.calcHist([channel], [0], None, [256], [0, 256]).flatten()
    hist_sum = hist.sum()
    if hist_sum == 0:
        return np.ones(256, dtype=np.float64) / 256.0
    return hist / hist_sum


def compute_avg_color_histograms(image_folder):
    hist_a_sum = np.zeros(256, dtype=np.float64)
    hist_b_sum = np.zeros(256, dtype=np.float64)
    count = 0

    for file_name in iter_image_files(image_folder):
        img = cv2.imread(os.path.join(image_folder, file_name))
        if img is None:
            continue

        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        hist_a_sum += compute_histogram(lab[:, :, 1])
        hist_b_sum += compute_histogram(lab[:, :, 2])
        count += 1

    if count == 0:
        raise ValueError(f"No readable images found in {image_folder}")

    hist_a = hist_a_sum / count
    hist_b = hist_b_sum / count
    return {
        "a": hist_a / hist_a.sum(),
        "b": hist_b / hist_b.sum(),
    }


def build_lut(source_hist, target_hist):
    cdf_source = np.cumsum(source_hist)
    cdf_target = np.cumsum(target_hist)

    lut = np.searchsorted(cdf_target, cdf_source, side="left")
    return np.clip(lut, 0, 255).astype(np.float32)


def match_histogram_channel(source, target_hist, strength=1.0):
    source_hist = compute_histogram(source)
    lut = build_lut(source_hist, target_hist)

    identity = np.arange(256, dtype=np.float32)
    lut = (1.0 - strength) * identity + strength * lut
    return cv2.LUT(source, lut.astype(np.uint8))


def apply_color_matching(image, target_hists, strength=COLOR_MATCH_STRENGTH):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    lab[:, :, 1] = match_histogram_channel(lab[:, :, 1], target_hists["a"], strength)
    lab[:, :, 2] = match_histogram_channel(lab[:, :, 2], target_hists["b"], strength)

    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def main():
    reference_folder = "real_images"
    input_folder = "outputs"
    output_folder = "outputs_color"

    os.makedirs(output_folder, exist_ok=True)
    clear_output_folder(output_folder)

    print("Calculating reference color histograms...")
    target_hists = compute_avg_color_histograms(reference_folder)
    np.save("target_color_hists.npy", target_hists)

    print("Processing output images...")
    for file_name in tqdm(list(iter_image_files(input_folder))):
        img_path = os.path.join(input_folder, file_name)
        img = cv2.imread(img_path)
        if img is None:
            continue

        result = apply_color_matching(img, target_hists)
        cv2.imwrite(os.path.join(output_folder, file_name), result)

    print("Done. Results are in outputs_color/")


if __name__ == "__main__":
    main()
