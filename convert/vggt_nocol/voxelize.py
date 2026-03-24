import os
import json
import struct
import numpy as np
from PIL import Image

# ======= PATHS (keep adjacent, init with 'your/path') =======
TRANSFORMS_JSON = "/home/ubuntu/datasets_2/Ujikintoki/transforms_train.json"
# IMAGES_ROOT     = "/home/liuyuhao/datasets/cvpr_comp/lita/BlueHawaii_down_nocolmap"     # e.g. folder that contains "train/0001.JPG"
DEPTH_ROOT      = "/home/ubuntu/datasets_2/Ujikintoki/depth_maps"       # where depth npy files live
OUT_COLMAP_DIR  = "/home/ubuntu/datasets_2/Ujikintoki/sparse/0"
# ============================================================

# ======= OPTIONS =======
CAMERA_MODEL = "PINHOLE"
NERF_JSON = True
DEPTH_SCALE = 1.0
STRIDE = 8
MAX_POINTS_PER_IMAGE = 6000
RESIZE_DEPTH_TO_IMAGE = True
DEPTH_NAMING = "stem"
DEPTH_SUFFIX = ".npy"
USE_IMAGE = False
DEFAULT_RGB = (128, 128, 128)
DEPTH_COLOR_MODE = "percentile"  # "percentile" or "fixed"
DEPTH_COLOR_PERCENTILE = (5.0, 95.0)
DEPTH_COLOR_RANGE = (0.0, 10.0)

# Optional per-frame random subsample before voxel fusionzuh
DOWNSAMPLE_RATE = 0.5  # keep ratio (0~1]
RNG_SEED = 42

# Voxel fusion controls
VOXEL_SIZE = 0.05  # meters
MIN_OBS = 1        # minimum observations per voxel to keep
# =======================

# --- COLMAP model ids ---
COLMAP_MODEL_IDS = {
    "SIMPLE_PINHOLE": 0,
    "PINHOLE": 1,
}

def write_next_bytes(fid, data, fmt):
    fid.write(struct.pack(fmt, *data))

def write_cameras_bin(cameras, path):
    with open(path, "wb") as f:
        write_next_bytes(f, [len(cameras)], "<Q")
        for cam in cameras:
            cam_id, model, width, height, params = cam
            write_next_bytes(f, [cam_id], "<I")
            write_next_bytes(f, [COLMAP_MODEL_IDS[model]], "<i")
            write_next_bytes(f, [width, height], "<QQ")
            write_next_bytes(f, params, "<" + "d"*len(params))

def write_images_bin(images, path):
    with open(path, "wb") as f:
        write_next_bytes(f, [len(images)], "<Q")
        for img in images:
            image_id, qvec, tvec, cam_id, name, xys, point3D_ids = img
            write_next_bytes(f, [image_id], "<I")
            write_next_bytes(f, qvec, "<dddd")
            write_next_bytes(f, tvec, "<ddd")
            write_next_bytes(f, [cam_id], "<I")
            f.write(name.encode("utf-8") + b"\x00")
            write_next_bytes(f, [len(xys)], "<Q")
            for (x, y), pid in zip(xys, point3D_ids):
                write_next_bytes(f, [x, y], "<dd")
                write_next_bytes(f, [pid], "<q")

def write_points3D_bin(points, path):
    with open(path, "wb") as f:
        write_next_bytes(f, [len(points)], "<Q")
        for p in points:
            pid, xyz, rgb, error, track = p
            write_next_bytes(f, [pid], "<Q")
            write_next_bytes(f, xyz, "<ddd")
            write_next_bytes(f, rgb, "<BBB")
            write_next_bytes(f, [error], "<d")
            write_next_bytes(f, [len(track)], "<Q")
            for image_id, point2D_idx in track:
                write_next_bytes(f, [image_id, point2D_idx], "<II")

def rotmat_to_qvec(R):
    # From COLMAP convention
    m = R
    t = np.trace(m)
    if t > 0:
        r = np.sqrt(1.0 + t)
        w = 0.5 * r
        r = 0.5 / r
        x = (m[2,1] - m[1,2]) * r
        y = (m[0,2] - m[2,0]) * r
        z = (m[1,0] - m[0,1]) * r
    else:
        i = np.argmax([m[0,0], m[1,1], m[2,2]])
        if i == 0:
            r = np.sqrt(1.0 + m[0,0] - m[1,1] - m[2,2])
            x = 0.5 * r
            r = 0.5 / r
            y = (m[0,1] + m[1,0]) * r
            z = (m[0,2] + m[2,0]) * r
            w = (m[2,1] - m[1,2]) * r
        elif i == 1:
            r = np.sqrt(1.0 + m[1,1] - m[0,0] - m[2,2])
            y = 0.5 * r
            r = 0.5 / r
            x = (m[0,1] + m[1,0]) * r
            z = (m[1,2] + m[2,1]) * r
            w = (m[0,2] - m[2,0]) * r
        else:
            r = np.sqrt(1.0 + m[2,2] - m[0,0] - m[1,1])
            z = 0.5 * r
            r = 0.5 / r
            x = (m[0,2] + m[2,0]) * r
            y = (m[1,2] + m[2,1]) * r
            w = (m[1,0] - m[0,1]) * r
    return np.array([w, x, y, z], dtype=np.float64)

def resize_depth(depth, target_hw):
    th, tw = target_hw
    if depth.shape[0] == th and depth.shape[1] == tw:
        return depth
    if not RESIZE_DEPTH_TO_IMAGE:
        raise ValueError("Depth size mismatch and RESIZE_DEPTH_TO_IMAGE=False")
    d = Image.fromarray(depth.astype(np.float32), mode="F")
    d = d.resize((tw, th), resample=Image.BILINEAR)
    return np.array(d, dtype=np.float32)

def depth_path_for_frame(frame, idx):
    if DEPTH_NAMING == "stem":
        base = os.path.splitext(os.path.basename(frame["file_path"]))[0]
        return os.path.join(DEPTH_ROOT, base + DEPTH_SUFFIX)
    if DEPTH_NAMING == "index":
        return os.path.join(DEPTH_ROOT, f"depth_{idx:04d}.npy")
    raise ValueError("Unknown DEPTH_NAMING")

def _voxel_key(p_world, voxel_size):
    return tuple((p_world / voxel_size).astype(np.int64))

def main():
    os.makedirs(OUT_COLMAP_DIR, exist_ok=True)

    with open(TRANSFORMS_JSON, "r", encoding="utf-8") as f:
        meta = json.load(f)

    fx, fy = float(meta["fl_x"]), float(meta["fl_y"])
    cx, cy = float(meta["cx"]), float(meta["cy"])
    width, height = int(meta["w"]), int(meta["h"])

    # Build single camera
    cameras = [(1, CAMERA_MODEL, width, height, [fx, fy, cx, cy])]

    # Per-image metadata and observations
    images_meta = []          # (image_id, qvec, tvec, cam_id, name)
    obs_per_image = []        # list of list: [{"xy":(x,y), "world":p, "color":rgb}, ...]

    # Convert from NeRF(OpenGL) camera to OpenCV camera if needed
    C = np.diag([1, -1, -1]).astype(np.float64)

    rng_base = None
    if DOWNSAMPLE_RATE is not None and DOWNSAMPLE_RATE < 1.0:
        rng_base = np.random.default_rng(RNG_SEED)

    for img_id, frame in enumerate(meta["frames"], start=1):
        img_rel = frame["file_path"]

        c2w = np.array(frame["transform_matrix"], dtype=np.float64)
        R = c2w[:3, :3]
        t = c2w[:3, 3]

        if NERF_JSON:
            R = R @ C  # c2w in OpenCV camera coords
            # t unchanged

        # w2c
        R_wc = R.T
        t_wc = -R_wc @ t

        qvec = rotmat_to_qvec(R_wc)
        tvec = t_wc.astype(np.float64)

        # Load depth only
        depth_path = depth_path_for_frame(frame, img_id-1)
        depth = np.load(depth_path).astype(np.float32) * DEPTH_SCALE
        depth = resize_depth(depth, (height, width))
        if USE_IMAGE:
            img_path = os.path.join(IMAGES_ROOT, img_rel)
            if not os.path.exists(img_path):
                raise FileNotFoundError(img_path)
            rgb = np.array(Image.open(img_path).convert("RGB"))
        else:
            rgb = None
            if DEPTH_COLOR_MODE == "percentile":
                valid = depth[np.isfinite(depth) & (depth > 0)]
                if valid.size > 0:
                    dmin, dmax = np.percentile(valid, DEPTH_COLOR_PERCENTILE)
                else:
                    dmin, dmax = DEPTH_COLOR_RANGE
            else:
                dmin, dmax = DEPTH_COLOR_RANGE
            if dmax <= dmin:
                dmax = dmin + 1e-6

        frame_obs = []

        # sample pixels
        ys = range(0, height, STRIDE)
        xs = range(0, width, STRIDE)
        for y in ys:
            for x in xs:
                z = depth[y, x]
                if not np.isfinite(z) or z <= 0:
                    continue

                # camera coords (OpenCV)
                X = (x - cx) / fx * z
                Y = (y - cy) / fy * z
                Z = z
                p_cam = np.array([X, Y, Z], dtype=np.float64)

                # convert to Nerf cam coords before applying original c2w if needed
                if NERF_JSON:
                    p_cam = C @ p_cam

                # world point
                p_world = (c2w[:3, :3] @ p_cam) + c2w[:3, 3]

                if rgb is None:
                    v = (z - dmin) / (dmax - dmin)
                    v = float(np.clip(v, 0.0, 1.0))
                    g = int(round(v * 255.0))
                    color = [g, g, g]
                else:
                    color = rgb[y, x].tolist()
                    color = [int(c) for c in color]

                frame_obs.append({
                    "xy": (float(x), float(y)),
                    "world": p_world,
                    "color": color,
                })

                if MAX_POINTS_PER_IMAGE and len(frame_obs) >= MAX_POINTS_PER_IMAGE:
                    break
            if MAX_POINTS_PER_IMAGE and len(frame_obs) >= MAX_POINTS_PER_IMAGE:
                break

        # Optional per-frame random subsample
        if rng_base is not None and len(frame_obs) > 0:
            keep_num = max(1, int(len(frame_obs) * DOWNSAMPLE_RATE))
            keep_idx = rng_base.choice(len(frame_obs), size=keep_num, replace=False)
            frame_obs = [frame_obs[i] for i in np.sort(keep_idx)]

        images_meta.append((img_id, qvec.tolist(), tvec.tolist(), 1, img_rel))
        obs_per_image.append(frame_obs)

    if VOXEL_SIZE is None or VOXEL_SIZE <= 0:
        raise ValueError("VOXEL_SIZE must be > 0")

    # Voxel fusion across all frames
    voxel_map = {}
    for img_index, frame_obs in enumerate(obs_per_image):
        for obs_index, obs in enumerate(frame_obs):
            p_world = obs["world"]
            key = _voxel_key(p_world, VOXEL_SIZE)
            if key not in voxel_map:
                voxel_map[key] = {
                    "sum_xyz": np.zeros(3, dtype=np.float64),
                    "sum_rgb": np.zeros(3, dtype=np.float64),
                    "count": 0,
                    "obs_refs": [],
                }
            v = voxel_map[key]
            v["sum_xyz"] += p_world
            v["sum_rgb"] += np.array(obs["color"], dtype=np.float64)
            v["count"] += 1
            v["obs_refs"].append((img_index, obs_index))

    # Keep only voxels with enough observations
    kept_keys = [k for k, v in voxel_map.items() if v["count"] >= MIN_OBS]

    # Assign point ids
    voxel_to_pid = {}
    points3D = []
    tracks = {}
    next_pid = 1

    for key in kept_keys:
        voxel_to_pid[key] = next_pid
        tracks[next_pid] = []
        next_pid += 1

    # Build images with filtered points and track lists
    images = []
    for img_index, frame_obs in enumerate(obs_per_image):
        image_id, qvec, tvec, cam_id, name = images_meta[img_index]
        xys = []
        pids = []

        for obs in frame_obs:
            key = _voxel_key(obs["world"], VOXEL_SIZE)
            pid = voxel_to_pid.get(key)
            if pid is None:
                continue
            point2D_idx = len(xys)
            xys.append(obs["xy"])
            pids.append(pid)
            tracks[pid].append((image_id, point2D_idx))

        images.append((image_id, qvec, tvec, cam_id, name, xys, pids))

    # Build points3D
    for key in kept_keys:
        v = voxel_map[key]
        pid = voxel_to_pid[key]
        xyz = (v["sum_xyz"] / v["count"]).tolist()
        rgb = (v["sum_rgb"] / v["count"]).astype(np.int64)
        rgb = [int(np.clip(c, 0, 255)) for c in rgb.tolist()]
        points3D.append((pid, xyz, rgb, 1.0, tracks[pid]))

    # Write bins
    write_cameras_bin(cameras, os.path.join(OUT_COLMAP_DIR, "cameras.bin"))
    write_images_bin(images, os.path.join(OUT_COLMAP_DIR, "images.bin"))
    write_points3D_bin(points3D, os.path.join(OUT_COLMAP_DIR, "points3D.bin"))

    print("Done:", OUT_COLMAP_DIR)

if __name__ == "__main__":
    main()
