# scripts/train.py
# 训练脚本：CPU 小规模训练闭环（合成数据集，CPU 几分钟可跑完）
# 用法: python scripts/train.py [--epochs 10]
import argparse
import os

from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser(description="YOLOv8 CPU 训练基线")
    ap.add_argument("--epochs", type=int, default=10, help="训练轮数")
    ap.add_argument("--imgsz", type=int, default=320, help="训练图片尺寸")
    ap.add_argument("--batch", type=int, default=16, help="batch 大小")
    ap.add_argument("--data", default="configs/data.yaml", help="数据集配置")
    ap.add_argument("--name", default="synth_baseline", help="本次训练命名")
    args = ap.parse_args()

    assert os.path.exists(args.data), \
        "找不到 %s，先跑 python scripts/make_synth_data.py 生成数据" % args.data
    model = YOLO("yolov8n.pt")  # 官方权重首次运行自动下载
    # project 用绝对路径，避免 ultralytics 默认 runs_dir 导致目录嵌套
    model.train(data=args.data, epochs=args.epochs, imgsz=args.imgsz,
                batch=args.batch, device="cpu", workers=2,
                project=os.path.abspath("runs"), name=args.name, verbose=True)
    print("训练完成，权重在 runs/%s/weights/best.pt" % args.name)
    print("用它推理: python scripts/infer.py "
          "--weights runs/%s/weights/best.pt" % args.name)


if __name__ == "__main__":
    main()
