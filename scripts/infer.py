# scripts/infer.py
# 推理脚本：在示例图片上跑 YOLOv8 检测并保存标注结果
# 用法示例:
#   python scripts/infer.py                                  # yolov8n.pt 跑 samples/
#   python scripts/infer.py --weights runs/synth_baseline/weights/best.pt
import argparse
import glob
import os

from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser(description="YOLOv8 推理基线")
    ap.add_argument("--weights", default="yolov8n.pt",
                    help="模型权重（官方权重首次运行自动下载）")
    ap.add_argument("--source", default="samples", help="图片目录")
    ap.add_argument("--out", default="outputs", help="标注结果输出目录")
    ap.add_argument("--conf", type=float, default=0.25, help="置信度阈值")
    ap.add_argument("--imgsz", type=int, default=640, help="推理图片尺寸")
    args = ap.parse_args()

    model = YOLO(args.weights)
    os.makedirs(args.out, exist_ok=True)
    imgs = sorted(glob.glob(os.path.join(args.source, "*.*")))
    assert imgs, "在 %s 里没找到图片，先跑 python scripts/make_synth_data.py" % args.source
    for p in imgs:
        res = model.predict(p, conf=args.conf, imgsz=args.imgsz, verbose=False)[0]
        n = len(res.boxes)
        msg = "%s: 检出 %d 个目标" % (os.path.basename(p), n)
        if n:
            names = model.names
            dets = ", ".join(
                "%s(%.2f)" % (names[int(c)], float(cf))
                for c, cf in zip(res.boxes.cls, res.boxes.conf)
            )
            msg += " -> " + dets
        print(msg)
        res.save(filename=os.path.join(args.out, os.path.basename(p)))
    print("标注图已保存到 %s/" % args.out)


if __name__ == "__main__":
    main()
