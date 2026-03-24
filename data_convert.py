import os
import json
import struct
import numpy as np
import shutil

# ========= 多场景 =========
DATASETS = [
    "Chocolate",
    "Laboratory",
    "GearWorks",
    "Cupcake",
    "Popcorn",
    "Ujikintoki",
    "MilkCookie",   
]

ROOT_DIR = "./dataset_v2"
OUT_ROOT = "./dataset_colmap"

CAMERA_MODEL = "PINHOLE"
NERF_JSON = True

COLMAP_MODEL_IDS = {"PINHOLE": 1}

# ========= IO =========
def write_next_bytes(fid, data, fmt):
    fid.write(struct.pack(fmt, *data))

def write_cameras_bin(cameras, path):
    with open(path, "wb") as f:
        write_next_bytes(f, [len(cameras)], "<Q")
        for cam_id, model, w, h, params in cameras:
            write_next_bytes(f, [cam_id], "<I")
            write_next_bytes(f, [COLMAP_MODEL_IDS[model]], "<i")
            write_next_bytes(f, [w, h], "<QQ")
            write_next_bytes(f, params, "<" + "d"*len(params))

def write_images_bin(images, path):
    with open(path, "wb") as f:
        write_next_bytes(f, [len(images)], "<Q")
        for img in images:
            image_id, qvec, tvec, cam_id, name, _, _ = img
            write_next_bytes(f, [image_id], "<I")
            write_next_bytes(f, qvec, "<dddd")
            write_next_bytes(f, tvec, "<ddd")
            write_next_bytes(f, [cam_id], "<I")
            f.write(name.encode() + b"\x00")
            write_next_bytes(f, [0], "<Q")

def write_points3D_bin(points, path):
    with open(path, "wb") as f:
        write_next_bytes(f, [len(points)], "<Q")
        for pid, xyz, rgb, error, _ in points:
            write_next_bytes(f, [pid], "<Q")
            write_next_bytes(f, xyz, "<ddd")
            write_next_bytes(f, rgb, "<BBB")
            write_next_bytes(f, [error], "<d")
            write_next_bytes(f, [0], "<Q")

# ========= 数学 =========
def rotmat_to_qvec(R):
    t = np.trace(R)
    if t > 0:
        r = np.sqrt(1 + t)
        w = 0.5 * r
        r = 0.5 / r
        x = (R[2,1]-R[1,2])*r
        y = (R[0,2]-R[2,0])*r
        z = (R[1,0]-R[0,1])*r
    else:
        i = np.argmax([R[0,0], R[1,1], R[2,2]])
        if i==0:
            r = np.sqrt(1+R[0,0]-R[1,1]-R[2,2])
            x=0.5*r; r=0.5/r
            y=(R[0,1]+R[1,0])*r
            z=(R[0,2]+R[2,0])*r
            w=(R[2,1]-R[1,2])*r
        elif i==1:
            r=np.sqrt(1+R[1,1]-R[0,0]-R[2,2])
            y=0.5*r; r=0.5/r
            x=(R[0,1]+R[1,0])*r
            z=(R[1,2]+R[2,1])*r
            w=(R[0,2]-R[2,0])*r
        else:
            r=np.sqrt(1+R[2,2]-R[0,0]-R[1,1])
            z=0.5*r; r=0.5/r
            x=(R[0,2]+R[2,0])*r
            y=(R[1,2]+R[2,1])*r
            w=(R[1,0]-R[0,1])*r
    return np.array([w,x,y,z])

def convert_pose(c2w, C):
    R = c2w[:3,:3]
    t = c2w[:3,3]
    if NERF_JSON:
        R = R @ C
    R_wc = R.T
    t_wc = -R_wc @ t
    return R_wc, t_wc

# ========= 单场景处理 =========
def process_scene(scene_name):
    print(f"\n🚀 Processing {scene_name}")

    data_dir = os.path.join(ROOT_DIR, scene_name)
    out_dir = os.path.join(OUT_ROOT, scene_name)

    sparse_dir = os.path.join(out_dir, "sparse/0")
    image_out = os.path.join(out_dir, "images")

    os.makedirs(sparse_dir, exist_ok=True)
    os.makedirs(image_out, exist_ok=True)

    train_json = os.path.join(data_dir, "transforms_train.json")
    test_json  = os.path.join(data_dir, "transforms_test.json")

    with open(train_json) as f:
        train_meta = json.load(f)
    with open(test_json) as f:
        test_meta = json.load(f)

    fx = float(train_meta["fl_x"])
    fy = float(train_meta["fl_y"])
    cx = float(train_meta["cx"])
    cy = float(train_meta["cy"])
    W  = int(train_meta["w"])
    H  = int(train_meta["h"])

    cameras = [(1, CAMERA_MODEL, W, H, [fx, fy, cx, cy])]
    write_cameras_bin(cameras, os.path.join(sparse_dir, "cameras.bin"))

    C = np.diag([1,-1,-1])
    images = []

    # ===== TRAIN =====
    for img_id, frame in enumerate(train_meta["frames"], 1):
        c2w = np.array(frame["transform_matrix"])
        R_wc, t_wc = convert_pose(c2w, C)
        qvec = rotmat_to_qvec(R_wc)

        rel = frame["file_path"].replace("./","")
        src = os.path.join(data_dir, rel + ".png")

        name = f"{img_id:04d}.png"
        dst = os.path.join(image_out, name)

        if not os.path.exists(dst):
            shutil.copy(src, dst)

        images.append((img_id, qvec, t_wc.tolist(), 1, name, [], []))

    write_images_bin(images, os.path.join(sparse_dir, "images.bin"))

    # ===== 随机点云 =====
    points = []
    for pid in range(1, 100000):
        xyz = np.random.uniform(-3,3,3).tolist()
        points.append((pid, xyz, [128,128,128], 1.0, []))

    write_points3D_bin(points, os.path.join(sparse_dir, "points3D.bin"))

    # ===== TEST =====
    test_poses = []
    for frame in test_meta["frames"]:
        c2w = np.array(frame["transform_matrix"])
        R_wc, t_wc = convert_pose(c2w, C)

        pose = np.eye(4)
        pose[:3,:3] = R_wc
        pose[:3,3] = t_wc
        test_poses.append(pose)

    np.save(os.path.join(out_dir, "test_poses.npy"), np.stack(test_poses))

    print(f"✅ {scene_name} DONE | train:{len(images)} test:{len(test_poses)}")


# ========= 主入口 =========
def main():
    for scene in DATASETS:
        process_scene(scene)

    print("\n🎉 ALL SCENES DONE")

if __name__ == "__main__":
    main()