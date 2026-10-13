# RAICOM CV 检测基线

赛题无关的 OpenCV + YOLOv8 视觉检测基线：环境、推理、训练闭环全部跑通。
12 月校赛题一落地，直接往上加功能，不用再花一周搭环境。

## 快速开始

```bash
pip install -r requirements.txt

# 1. 生成合成小数据集 + 演示图片
python scripts/make_synth_data.py

# 2. 推理：在示例图片上跑检测（官方 yolov8n.pt 首次运行自动下载）
python scripts/infer.py
# 标注结果在 outputs/

# 3. 训练：CPU 小规模训练闭环
python scripts/train.py --epochs 10
# 权重在 runs/synth_baseline/weights/best.pt

# 4. 用自己训的模型推理
python scripts/infer.py --weights runs/synth_baseline/weights/best.pt
```

## 目录结构

```
├── requirements.txt        # ultralytics + opencv-python + numpy
├── configs/data.yaml       # 数据集配置（YOLO 格式）
├── scripts/
│   ├── make_synth_data.py  # 生成合成数据集（矩形/圆形，80 训 + 20 验）
│   ├── infer.py            # 推理脚本：图片目录 -> 标注结果 + 检测打印
│   └── train.py            # 训练脚本：CPU，默认 10 epoch，320 尺寸
├── docs/训练与扩展说明.md  # 训练流程、调参、赛题扩展路线（队友必看）
├── samples/                # 演示图片（脚本生成，git 不提交）
├── data/                   # 数据集（脚本生成，git 不提交）
├── runs/                   # 训练输出（git 不提交）
└── outputs/                # 推理标注结果（git 不提交）
```

## 验证状态（2026-10-13 沙箱实测）

- 推理脚本在示例图片上跑通（官方 yolov8n.pt 和自训权重两种都验证过）
- 训练流程在合成小数据集（80 训 / 20 验）上完整跑完 10 epoch，产出 `best.pt`，
  验证集 mAP50 = 0.96（rect 0.99 / circle 0.93），CPU 约 2 分钟
- 全 CPU 环境，不依赖 GPU

细节和扩展路线见 `docs/训练与扩展说明.md`。
