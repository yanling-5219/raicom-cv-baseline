# scripts/make_synth_data.py
# 生成合成小数据集（YOLO 格式）：随机矩形 / 圆形，供基线训练验证
# 用法: python scripts/make_synth_data.py [--n-train 80] [--n-val 20]
import argparse
import glob
import os
import random
import shutil

import cv2
import numpy as np

IMG_SIZE = 320


def draw_sample(rng):
    """画一张图，返回 (图片, [(cls, cx, cy, w, h), ...])，坐标已归一化。"""
    bg = [rng.randint(200, 255) for _ in range(3)]
    img = np.full((IMG_SIZE, IMG_SIZE, 3), 255, dtype=np.uint8)
    img[:] = bg
    labels = []
    for _ in range(rng.randint(1, 4)):
        cls = rng.randint(0, 1)  # 0=矩形, 1=圆形
        color = (rng.randint(0, 150), rng.randint(0, 150), rng.randint(0, 150))
        cx, cy = rng.uniform(0.15, 0.85), rng.uniform(0.15, 0.85)
        w, h = rng.uniform(0.10, 0.35), rng.uniform(0.10, 0.35)
        if cls == 0:
            x1, y1 = int((cx - w / 2) * IMG_SIZE), int((cy - h / 2) * IMG_SIZE)
            x2, y2 = int((cx + w / 2) * IMG_SIZE), int((cy + h / 2) * IMG_SIZE)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)
        else:
            r = int(min(w, h) * IMG_SIZE / 2)
            ccx, ccy = int(cx * IMG_SIZE), int(cy * IMG_SIZE)
            cv2.circle(img, (ccx, ccy), r, color, -1)
            # 标签用圆形的外接正方形
            w = h = 2 * r / IMG_SIZE
        labels.append((cls, cx, cy, w, h))
    return img, labels


def make_split(root, split, n, seed):
    rng = random.Random(seed)
    img_dir = os.path.join(root, split, "images")
    lbl_dir = os.path.join(root, split, "labels")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)
    for i in range(n):
        img, labels = draw_sample(rng)
        name = "%s_%04d" % (split, i)
        cv2.imwrite(os.path.join(img_dir, name + ".jpg"), img)
        with open(os.path.join(lbl_dir, name + ".txt"), "w") as f:
            for cls, cx, cy, w, h in labels:
                f.write("%d %.4f %.4f %.4f %.4f\n" % (cls, cx, cy, w, h))
    print("%s: %d images -> %s" % (split, n, img_dir))


def main():
    ap = argparse.ArgumentParser(description="生成合成 YOLO 小数据集")
    ap.add_argument("--n-train", type=int, default=80)
    ap.add_argument("--n-val", type=int, default=20)
    ap.add_argument("--root", default="data/synth")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    make_split(args.root, "train", args.n_train, args.seed)
    make_split(args.root, "val", args.n_val, args.seed + 1)
    # 演示用图：从 val 里拷 3 张到 samples/
    os.makedirs("samples", exist_ok=True)
    for j, p in enumerate(
        sorted(glob.glob(os.path.join(args.root, "val", "images", "*.jpg")))[:3]
    ):
        shutil.copy(p, os.path.join("samples", "demo%d.jpg" % (j + 1)))
    print("samples/ demo images ready")


if __name__ == "__main__":
    main()
