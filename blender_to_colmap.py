import argparse
import json
import shutil
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np


COLMAP_CAMERA_MODEL_IDS = {
    "SIMPLE_PINHOLE": 0,
    "PINHOLE": 1,
    "SIMPLE_RADIAL": 2,
    "RADIAL": 3,
    "OPENCV": 4,
    "OPENCV_FISHEYE": 5,
    "FULL_OPENCV": 6,
    "FOV": 7,
    "SIMPLE_RADIAL_FISHEYE": 8,
    "RADIAL_FISHEYE": 9,
    "THIN_PRISM_FISHEYE": 10,
}

COLMAP_CAMERA_MODEL_NAMES = {
    value: key for key, value in COLMAP_CAMERA_MODEL_IDS.items()
}

COLMAP_CAMERA_NUM_PARAMS = {
    "SIMPLE_PINHOLE": 3,
    "PINHOLE": 4,
    "SIMPLE_RADIAL": 4,
    "RADIAL": 5,
    "OPENCV": 8,
    "OPENCV_FISHEYE": 8,
    "FULL_OPENCV": 12,
    "FOV": 5,
    "SIMPLE_RADIAL_FISHEYE": 4,
    "RADIAL_FISHEYE": 5,
    "THIN_PRISM_FISHEYE": 12,
}

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".exr")
OPENGL_TO_OPENCV = np.diag([1.0, -1.0, -1.0]).astype(np.float64)


@dataclass
class Camera:
    camera_id: int
    model: str
    width: int
    height: int
    params: List[float]


@dataclass
class ImageEntry:
    image_id: int
    qvec: List[float]
    tvec: List[float]
    camera_id: int
    name: str
    xys: List[Tuple[float, float]]
    point3D_ids: List[int]


@dataclass
class Point3D:
    point3D_id: int
    xyz: List[float]
    rgb: List[int]
    error: float
    track: List[Tuple[int, int]]


@dataclass
class ConvertedImage:
    split: str
    entry: ImageEntry


def read_next_bytes(fid, num_bytes, fmt, endian="<"):
    data = fid.read(num_bytes)
    return struct.unpack(endian + fmt, data)


def write_next_bytes(fid, data, fmt, endian="<"):
    fid.write(struct.pack(endian + fmt, *data))


def rotmat_to_qvec(R):
    trace = np.trace(R)
    if trace > 0.0:
        root = np.sqrt(1.0 + trace)
        qw = 0.5 * root
        scale = 0.5 / root
        qx = (R[2, 1] - R[1, 2]) * scale
        qy = (R[0, 2] - R[2, 0]) * scale
        qz = (R[1, 0] - R[0, 1]) * scale
    else:
        diagonal = np.diag(R)
        idx = int(np.argmax(diagonal))
        if idx == 0:
            root = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            qx = 0.5 * root
            scale = 0.5 / root
            qy = (R[0, 1] + R[1, 0]) * scale
            qz = (R[0, 2] + R[2, 0]) * scale
            qw = (R[2, 1] - R[1, 2]) * scale
        elif idx == 1:
            root = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            qy = 0.5 * root
            scale = 0.5 / root
            qx = (R[0, 1] + R[1, 0]) * scale
            qz = (R[1, 2] + R[2, 1]) * scale
            qw = (R[0, 2] - R[2, 0]) * scale
        else:
            root = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            qz = 0.5 * root
            scale = 0.5 / root
            qx = (R[0, 2] + R[2, 0]) * scale
            qy = (R[1, 2] + R[2, 1]) * scale
            qw = (R[1, 0] - R[0, 1]) * scale
    return np.array([qw, qx, qy, qz], dtype=np.float64)


def normalize_rel_path(path_str):
    path_str = path_str.replace("\\", "/")
    while path_str.startswith("./"):
        path_str = path_str[2:]
    return path_str


def resolve_path(path_str, base_dir=None):
    if path_str is None:
        return None

    path = Path(path_str).expanduser()
    if path.is_absolute():
        return path

    if base_dir is None:
        base_dir = Path.cwd()
    else:
        base_dir = Path(base_dir)

    return (base_dir / path).resolve()


def stem_key(path_str):
    return Path(path_str).stem.lower()


def read_cameras_bin(path):
    cameras = {}
    with open(path, "rb") as fid:
        num_cameras = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_cameras):
            camera_id = read_next_bytes(fid, 4, "I")[0]
            model_id = read_next_bytes(fid, 4, "i")[0]
            width = read_next_bytes(fid, 8, "Q")[0]
            height = read_next_bytes(fid, 8, "Q")[0]
            model = COLMAP_CAMERA_MODEL_NAMES[model_id]
            num_params = COLMAP_CAMERA_NUM_PARAMS[model]
            params = list(read_next_bytes(fid, 8 * num_params, "d" * num_params))
            cameras[camera_id] = Camera(camera_id, model, width, height, params)
    return cameras


def read_images_bin(path):
    images = {}
    with open(path, "rb") as fid:
        num_images = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_images):
            image_id = read_next_bytes(fid, 4, "I")[0]
            qvec = list(read_next_bytes(fid, 32, "dddd"))
            tvec = list(read_next_bytes(fid, 24, "ddd"))
            camera_id = read_next_bytes(fid, 4, "I")[0]

            name_bytes = bytearray()
            while True:
                char = fid.read(1)
                if char == b"\x00":
                    break
                name_bytes.extend(char)
            name = name_bytes.decode("utf-8")

            num_points2d = read_next_bytes(fid, 8, "Q")[0]
            xys = []
            point3D_ids = []
            for _ in range(num_points2d):
                x, y = read_next_bytes(fid, 16, "dd")
                point3D_id = read_next_bytes(fid, 8, "q")[0]
                xys.append((float(x), float(y)))
                point3D_ids.append(int(point3D_id))

            images[image_id] = ImageEntry(
                image_id=image_id,
                qvec=qvec,
                tvec=tvec,
                camera_id=camera_id,
                name=name,
                xys=xys,
                point3D_ids=point3D_ids,
            )
    return images


def read_points3d_bin(path):
    points3d = {}
    with open(path, "rb") as fid:
        num_points = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_points):
            point3D_id = read_next_bytes(fid, 8, "Q")[0]
            xyz = list(read_next_bytes(fid, 24, "ddd"))
            rgb = list(read_next_bytes(fid, 3, "BBB"))
            error = read_next_bytes(fid, 8, "d")[0]
            track_length = read_next_bytes(fid, 8, "Q")[0]
            track = []
            for _ in range(track_length):
                image_id, point2d_idx = read_next_bytes(fid, 8, "II")
                track.append((int(image_id), int(point2d_idx)))
            points3d[point3D_id] = Point3D(
                point3D_id=point3D_id,
                xyz=xyz,
                rgb=rgb,
                error=float(error),
                track=track,
            )
    return points3d


def read_cameras_txt(path):
    cameras = {}
    with open(path, "r", encoding="utf-8") as fid:
        for line in fid:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            camera_id = int(parts[0])
            model = parts[1]
            width = int(parts[2])
            height = int(parts[3])
            params = [float(x) for x in parts[4:]]
            cameras[camera_id] = Camera(camera_id, model, width, height, params)
    return cameras


def read_images_txt(path):
    images = {}
    with open(path, "r", encoding="utf-8") as fid:
        lines = [line.rstrip("\n") for line in fid]

    idx = 0
    while idx < len(lines):
        line = lines[idx].strip()
        idx += 1
        if not line or line.startswith("#"):
            continue

        parts = line.split()
        image_id = int(parts[0])
        qvec = [float(x) for x in parts[1:5]]
        tvec = [float(x) for x in parts[5:8]]
        camera_id = int(parts[8])
        name = parts[9]

        xys = []
        point3D_ids = []
        if idx < len(lines):
            points_line = lines[idx].strip()
            idx += 1
            if points_line:
                values = points_line.split()
                for offset in range(0, len(values), 3):
                    xys.append((float(values[offset]), float(values[offset + 1])))
                    point3D_ids.append(int(values[offset + 2]))

        images[image_id] = ImageEntry(
            image_id=image_id,
            qvec=qvec,
            tvec=tvec,
            camera_id=camera_id,
            name=name,
            xys=xys,
            point3D_ids=point3D_ids,
        )
    return images


def read_points3d_txt(path):
    points3d = {}
    with open(path, "r", encoding="utf-8") as fid:
        for line in fid:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            point3D_id = int(parts[0])
            xyz = [float(x) for x in parts[1:4]]
            rgb = [int(x) for x in parts[4:7]]
            error = float(parts[7])
            track_vals = parts[8:]
            track = []
            for offset in range(0, len(track_vals), 2):
                image_id = int(track_vals[offset])
                point2d_idx = int(track_vals[offset + 1])
                track.append((image_id, point2d_idx))
            points3d[point3D_id] = Point3D(point3D_id, xyz, rgb, error, track)
    return points3d


def write_cameras_bin(cameras, path):
    with open(path, "wb") as fid:
        write_next_bytes(fid, [len(cameras)], "Q")
        for camera in cameras:
            write_next_bytes(fid, [camera.camera_id], "I")
            write_next_bytes(fid, [COLMAP_CAMERA_MODEL_IDS[camera.model]], "i")
            write_next_bytes(fid, [camera.width, camera.height], "QQ")
            write_next_bytes(fid, camera.params, "d" * len(camera.params))


def write_images_bin(images, path):
    with open(path, "wb") as fid:
        write_next_bytes(fid, [len(images)], "Q")
        for image in images:
            write_next_bytes(fid, [image.image_id], "I")
            write_next_bytes(fid, image.qvec, "dddd")
            write_next_bytes(fid, image.tvec, "ddd")
            write_next_bytes(fid, [image.camera_id], "I")
            fid.write(image.name.encode("utf-8"))
            fid.write(b"\x00")
            write_next_bytes(fid, [len(image.xys)], "Q")
            for (x, y), point3D_id in zip(image.xys, image.point3D_ids):
                write_next_bytes(fid, [x, y], "dd")
                write_next_bytes(fid, [point3D_id], "q")


def write_points3d_bin(points3d, path):
    with open(path, "wb") as fid:
        write_next_bytes(fid, [len(points3d)], "Q")
        for point in points3d:
            write_next_bytes(fid, [point.point3D_id], "Q")
            write_next_bytes(fid, point.xyz, "ddd")
            write_next_bytes(fid, point.rgb, "BBB")
            write_next_bytes(fid, [point.error], "d")
            write_next_bytes(fid, [len(point.track)], "Q")
            for image_id, point2d_idx in point.track:
                write_next_bytes(fid, [image_id, point2d_idx], "II")


def load_existing_model(model_dir):
    model_dir = Path(model_dir)

    cameras_bin = model_dir / "cameras.bin"
    images_bin = model_dir / "images.bin"
    points_bin = model_dir / "points3D.bin"
    cameras_txt = model_dir / "cameras.txt"
    images_txt = model_dir / "images.txt"
    points_txt = model_dir / "points3D.txt"

    if cameras_bin.exists() and images_bin.exists() and points_bin.exists():
        return (
            read_cameras_bin(cameras_bin),
            read_images_bin(images_bin),
            read_points3d_bin(points_bin),
        )

    if cameras_txt.exists() and images_txt.exists() and points_txt.exists():
        return (
            read_cameras_txt(cameras_txt),
            read_images_txt(images_txt),
            read_points3d_txt(points_txt),
        )

    raise FileNotFoundError(
        f"Could not find COLMAP model files in {model_dir}. "
        "Expected either cameras/images/points3D in .bin or .txt format."
    )


def build_camera_from_json(meta, camera_model):
    width = int(meta["w"])
    height = int(meta["h"])

    if "fl_x" in meta and "fl_y" in meta:
        fx = float(meta["fl_x"])
        fy = float(meta["fl_y"])
    elif "camera_angle_x" in meta:
        angle_x = float(meta["camera_angle_x"])
        fx = 0.5 * width / np.tan(0.5 * angle_x)
        fy = fx
    else:
        raise ValueError("JSON must contain fl_x/fl_y or camera_angle_x.")

    cx = float(meta.get("cx", width / 2.0))
    cy = float(meta.get("cy", height / 2.0))

    if camera_model == "SIMPLE_PINHOLE":
        params = [fx, cx, cy]
    elif camera_model == "PINHOLE":
        params = [fx, fy, cx, cy]
    else:
        raise ValueError(
            "This script currently writes Blender cameras as SIMPLE_PINHOLE or PINHOLE."
        )

    return Camera(camera_id=1, model=camera_model, width=width, height=height, params=params)


def resolve_image_name(file_path, image_root):
    file_path = normalize_rel_path(file_path)
    root = Path(image_root) if image_root else None

    if Path(file_path).suffix:
        return file_path

    if root is None:
        return file_path + ".png"

    candidate_base = root / file_path
    for extension in IMAGE_EXTENSIONS:
        candidate = candidate_base.with_suffix(extension)
        if candidate.exists():
            return normalize_rel_path(str(candidate.relative_to(root)))

    stem = Path(file_path).name
    matches = list(root.rglob(stem + ".*"))
    valid_matches = [path for path in matches if path.suffix.lower() in IMAGE_EXTENSIONS]
    if len(valid_matches) == 1:
        return normalize_rel_path(str(valid_matches[0].relative_to(root)))

    return file_path + ".png"


def blender_pose_to_colmap(frame):
    c2w = np.array(frame["transform_matrix"], dtype=np.float64)

    R_blender = c2w[:3, :3]
    t_world = c2w[:3, 3]

    R_opencv_c2w = R_blender @ OPENGL_TO_OPENCV
    R_wc = R_opencv_c2w.T
    t_wc = -R_wc @ t_world

    qvec = rotmat_to_qvec(R_wc).tolist()
    tvec = t_wc.astype(np.float64).tolist()
    return qvec, tvec


def build_existing_name_index(existing_images):
    exact = {}
    basename = {}
    stem = {}

    basename_counts = {}
    stem_counts = {}

    for image in existing_images.values():
        exact_key = normalize_rel_path(image.name)
        exact[exact_key] = image

        basename_key = Path(exact_key).name.lower()
        basename_counts[basename_key] = basename_counts.get(basename_key, 0) + 1

        stem_name = stem_key(exact_key)
        stem_counts[stem_name] = stem_counts.get(stem_name, 0) + 1

    for image in existing_images.values():
        exact_key = normalize_rel_path(image.name)
        basename_key = Path(exact_key).name.lower()
        if basename_counts[basename_key] == 1:
            basename[basename_key] = image

        stem_name = stem_key(exact_key)
        if stem_counts[stem_name] == 1:
            stem[stem_name] = image

    return exact, basename, stem


def match_existing_image(resolved_name, existing_index):
    exact, basename, stem = existing_index

    normalized = normalize_rel_path(resolved_name)
    if normalized in exact:
        return exact[normalized]

    basename_key = Path(normalized).name.lower()
    if basename_key in basename:
        return basename[basename_key]

    stem_name = stem_key(normalized)
    return stem.get(stem_name)


def collect_frames(json_path, split_name):
    if json_path is None:
        return None, []

    with open(json_path, "r", encoding="utf-8") as fid:
        meta = json.load(fid)

    frames = meta.get("frames", [])
    if not frames:
        raise ValueError(f"No frames found in {json_path}")

    return meta, [{"split": split_name, "frame": frame} for frame in frames]


def ensure_clean_output(output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in ("cameras.bin", "images.bin", "points3D.bin"):
        target = output_dir / name
        if target.exists():
            target.unlink()


def filter_points3d_for_images(points3d_dict, kept_image_ids):
    kept_point_ids = set()
    filtered_points = []
    for point_id in sorted(points3d_dict.keys()):
        point = points3d_dict[point_id]
        filtered_track = [
            (image_id, point2d_idx)
            for image_id, point2d_idx in point.track
            if image_id in kept_image_ids
        ]
        if not filtered_track:
            continue
        kept_point_ids.add(point_id)
        filtered_points.append(
            Point3D(
                point3D_id=point.point3D_id,
                xyz=point.xyz,
                rgb=point.rgb,
                error=point.error,
                track=filtered_track,
            )
        )
    return kept_point_ids, filtered_points


def clone_images_with_filtered_points(images, kept_point_ids):
    cloned = []
    for image in images:
        cloned.append(
            ImageEntry(
                image_id=image.image_id,
                qvec=list(image.qvec),
                tvec=list(image.tvec),
                camera_id=image.camera_id,
                name=image.name,
                xys=list(image.xys),
                point3D_ids=[
                    point_id if point_id in kept_point_ids else -1
                    for point_id in image.point3D_ids
                ],
            )
        )
    return cloned


def write_model(output_dir, camera, images, points3d):
    output_dir = Path(output_dir)
    ensure_clean_output(output_dir)
    write_cameras_bin([camera], output_dir / "cameras.bin")
    write_images_bin(images, output_dir / "images.bin")
    write_points3d_bin(points3d, output_dir / "points3D.bin")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Convert Blender/NeRF transforms JSON to a COLMAP sparse model, "
            "optionally reusing train observations and points from an existing COLMAP model."
        )
    )
    parser.add_argument("--train-json", required=True, help="Path to transforms_train.json")
    parser.add_argument("--test-json", default=None, help="Path to transforms_test.json")
    parser.add_argument(
        "--image-root",
        default=None,
        help=(
            "Dataset root used to resolve frame file_path values. "
            "For example, if JSON uses train/r_0 and files live under images/train/r_0.png, "
            "pass --image-root images."
        ),
    )
    parser.add_argument(
        "--existing-colmap",
        default=None,
        help=(
            "Existing COLMAP sparse model directory (for example sparse/0). "
            "When provided, train image observations and points3D are reused."
        ),
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to write cameras.bin, images.bin and points3D.bin",
    )
    parser.add_argument(
        "--camera-model",
        default="PINHOLE",
        choices=("PINHOLE", "SIMPLE_PINHOLE"),
        help="Camera model used when writing camera intrinsics from Blender JSON.",
    )
    parser.add_argument(
        "--copy-original-model",
        action="store_true",
        help=(
            "Also copy the original train COLMAP model into output_dir/original_train_model "
            "for inspection."
        ),
    )
    parser.add_argument(
        "--split-subdirs",
        action="store_true",
        help=(
            "Also write output_dir/all, output_dir/train and output_dir/test. "
            "The test model contains test poses for rendering and an empty points3D.bin."
        ),
    )
    args = parser.parse_args()

    work_dir = Path.cwd()
    train_json_path = resolve_path(args.train_json, work_dir)
    test_json_path = resolve_path(args.test_json, work_dir)
    image_root_path = resolve_path(args.image_root, work_dir)
    existing_colmap_path = resolve_path(args.existing_colmap, work_dir)
    output_dir_path = resolve_path(args.output_dir, work_dir)

    train_meta, train_frames = collect_frames(train_json_path, "train")
    _, test_frames = collect_frames(test_json_path, "test")

    all_frames = train_frames + test_frames
    if not all_frames:
        raise ValueError("No train/test frames were collected.")

    camera = build_camera_from_json(train_meta, args.camera_model)

    existing_images = {}
    existing_points3d = {}
    if existing_colmap_path:
        _, existing_images, existing_points3d = load_existing_model(existing_colmap_path)
        existing_index = build_existing_name_index(existing_images)
    else:
        existing_index = ({}, {}, {})

    used_existing_ids = set()
    next_new_id = (
        max(existing_images.keys()) + 1 if existing_images else 1
    )
    converted_images = []
    reused_train_count = 0
    test_count = 0

    for item in all_frames:
        split = item["split"]
        frame = item["frame"]

        resolved_name = resolve_image_name(frame["file_path"], image_root_path)
        qvec, tvec = blender_pose_to_colmap(frame)

        matched_existing = match_existing_image(resolved_name, existing_index)
        if split == "train" and matched_existing is not None:
            image_id = matched_existing.image_id
            used_existing_ids.add(image_id)
            xys = matched_existing.xys
            point3D_ids = matched_existing.point3D_ids
            reused_train_count += 1
        else:
            while next_new_id in used_existing_ids or next_new_id in existing_images:
                next_new_id += 1
            image_id = next_new_id
            used_existing_ids.add(image_id)
            next_new_id += 1
            xys = []
            point3D_ids = []

        if split == "test":
            test_count += 1

        converted_images.append(
            ConvertedImage(
                split=split,
                entry=ImageEntry(
                    image_id=image_id,
                    qvec=qvec,
                    tvec=tvec,
                    camera_id=camera.camera_id,
                    name=resolved_name,
                    xys=xys,
                    point3D_ids=point3D_ids,
                ),
            )
        )

    converted_images.sort(key=lambda item: item.entry.image_id)

    all_entries = [item.entry for item in converted_images]
    train_entries = [item.entry for item in converted_images if item.split == "train"]
    test_entries = [item.entry for item in converted_images if item.split == "test"]

    kept_image_ids_all = {image.image_id for image in all_entries}
    kept_image_ids_train = {image.image_id for image in train_entries}

    kept_point_ids_all, points3d_all = filter_points3d_for_images(
        existing_points3d, kept_image_ids_all
    )
    kept_point_ids_train, points3d_train = filter_points3d_for_images(
        existing_points3d, kept_image_ids_train
    )

    all_entries = clone_images_with_filtered_points(all_entries, kept_point_ids_all)
    train_entries = clone_images_with_filtered_points(train_entries, kept_point_ids_train)
    test_entries = clone_images_with_filtered_points(test_entries, set())

    output_dir = output_dir_path
    write_model(output_dir, camera, all_entries, points3d_all)

    if args.split_subdirs:
        write_model(output_dir / "all", camera, all_entries, points3d_all)
        write_model(output_dir / "train", camera, train_entries, points3d_train)
        write_model(output_dir / "test", camera, test_entries, [])

    if args.copy_original_model and existing_colmap_path:
        backup_dir = output_dir / "original_train_model"
        if backup_dir.exists():
            shutil.rmtree(backup_dir)
        shutil.copytree(existing_colmap_path, backup_dir)

    print(f"Wrote camera model to: {output_dir / 'cameras.bin'}")
    print(f"Wrote {len(all_entries)} images to: {output_dir / 'images.bin'}")
    print(f"Wrote {len(points3d_all)} points to: {output_dir / 'points3D.bin'}")
    print(f"Reused train observations for {reused_train_count} images.")
    print(f"Added {test_count} test images with empty 2D observations.")
    if args.split_subdirs:
        print(f"Wrote train-only model to: {output_dir / 'train'}")
        print(f"Wrote test-only model to: {output_dir / 'test'}")
        print(f"Wrote all-images model to: {output_dir / 'all'}")
    if existing_colmap_path and reused_train_count == 0:
        print(
            "Warning: no train frames matched the existing COLMAP image names. "
            "The output model still contains the original points3D, but train images "
            "will not carry over 2D-3D observations."
        )


if __name__ == "__main__":
    main()
