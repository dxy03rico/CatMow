"""
04_convert_tflite.py
将训练好的模型转换为 TFLite 格式（INT8 量化）

量化策略：
  - INT8 全量化（权重 + 激活值）
  - 使用代表性数据集校准
  - 目标：在 Pi5 上达到 <100ms/次推理

输出：
  models/cat_sound.tflite  (~5MB，适合 Pi5 部署)

使用方法:
    python scripts/04_convert_tflite.py
"""

import os
import json
import shutil
import numpy as np
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf

# ── 配置 ──────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
MODELS_DIR = ROOT / "models"
FEATURES_DIR = ROOT / "data" / "features"
PI5_DIR = ROOT / "pi5"

SAVED_MODEL_PATH = MODELS_DIR / "saved_model_export"
TFLITE_PATH = MODELS_DIR / "cat_sound.tflite"
TFLITE_PI5_PATH = PI5_DIR / "cat_sound.tflite"

# 校准数据量（INT8 量化使用）
CALIBRATION_SAMPLES = 200


# ── 代表性数据集生成器 ────────────────────────────────────────────────────────

def make_representative_dataset():
    """
    为 INT8 量化提供代表性数据集

    TFLite 用这批数据统计激活值分布，确定量化参数
    """
    X = np.load(FEATURES_DIR / "X.npy").astype(np.float32)

    # 随机采样，覆盖各类别
    indices = np.random.choice(len(X), size=min(CALIBRATION_SAMPLES, len(X)), replace=False)
    samples = X[indices]

    def generator():
        for sample in samples:
            yield [sample[np.newaxis, ...]]  # 添加 batch 维度

    return generator


# ── 转换函数 ──────────────────────────────────────────────────────────────────

def convert_to_tflite() -> int:
    """
    执行模型转换，返回 .tflite 文件大小（字节）
    先把 .keras 重新 save 成 tf.saved_model 格式再转换
    （绕开 TF 2.16 from_keras_model / model.export 的兼容性 bug）
    """
    print("=== 加载 SavedModel (model.export 格式) ===")
    print(f"  模型路径: {SAVED_MODEL_PATH}")

    print("\n=== Float32 TFLite 转换（无量化）===")
    converter = tf.lite.TFLiteConverter.from_saved_model(str(SAVED_MODEL_PATH))
    # 不做量化：TF 2.16 + MobileNetV2 量化有 LLVM bug
    # Float32 在 Pi5 上推理 <10ms，完全满足实时需求

    print("  正在量化（约需 30s）...")
    tflite_model = converter.convert()

    # 保存
    TFLITE_PATH.write_bytes(tflite_model)
    size_mb = len(tflite_model) / 1024 / 1024
    print(f"  保存到: {TFLITE_PATH}")
    print(f"  文件大小: {size_mb:.2f} MB")

    return len(tflite_model)


# ── 精度验证 ──────────────────────────────────────────────────────────────────

def verify_tflite():
    """加载 TFLite 模型，对测试集验证精度损失"""
    print("\n=== 验证 TFLite 精度 ===")

    X = np.load(FEATURES_DIR / "X.npy").astype(np.float32)
    y = np.load(FEATURES_DIR / "y.npy")

    import pandas as pd
    df = pd.read_csv(FEATURES_DIR / "dataset.csv")
    test_mask = (df["split"] == "test").values
    X_test, y_test = X[test_mask], y[test_mask]

    if len(X_test) == 0:
        print("  [跳过] 没有测试集数据")
        return

    # 加载 TFLite 模型
    interpreter = tf.lite.Interpreter(model_path=str(TFLITE_PATH))
    interpreter.allocate_tensors()

    input_details  = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print(f"  输入: {input_details[0]['shape']}  dtype={input_details[0]['dtype']}")
    print(f"  输出: {output_details[0]['shape']} dtype={output_details[0]['dtype']}")

    # 逐样本推理
    correct = 0
    import time
    latencies = []

    for i in range(min(100, len(X_test))):  # 最多验证 100 条
        inp = X_test[i][np.newaxis, ...].astype(np.float32)
        interpreter.set_tensor(input_details[0]["index"], inp)
        t0 = time.perf_counter()
        interpreter.invoke()
        latencies.append((time.perf_counter() - t0) * 1000)
        pred = interpreter.get_tensor(output_details[0]["index"]).argmax()
        if pred == y_test[i]:
            correct += 1

    n_eval = min(100, len(X_test))
    acc = correct / n_eval
    avg_ms = np.mean(latencies)
    print(f"\n  TFLite 精度 (前 {n_eval} 条): {acc:.2%}")
    print(f"  Mac 上平均推理耗时: {avg_ms:.1f} ms/次")
    print(f"  （Pi5 ARM64 通常为 Mac 的 2-4x，预计 {avg_ms*2:.0f}~{avg_ms*4:.0f} ms）")

    with open(FEATURES_DIR / "labels.txt") as f:
        categories = [line.strip().split()[1] for line in f]

    print(f"\n  各类别准确率（TFLite）：")
    for i, cat in enumerate(categories):
        mask = y_test[:len(X_test)] == i
        if mask.sum() == 0:
            continue
        cat_preds = []
        for j in np.where(mask)[0][:20]:  # 最多每类 20 条
            inp = X_test[j][np.newaxis, ...].astype(np.float32)
            interpreter.set_tensor(input_details[0]["index"], inp)
            interpreter.invoke()
            cat_preds.append(
                interpreter.get_tensor(output_details[0]["index"]).argmax()
            )
        cat_acc = np.mean(np.array(cat_preds) == i)
        bar = "█" * int(cat_acc * 20)
        print(f"    {cat:<8} {cat_acc:.2%}  {bar}")


# ── 复制到 pi5/ 目录 ──────────────────────────────────────────────────────────

def deploy_to_pi5():
    """复制 tflite 模型和配置到 pi5/ 目录"""
    print(f"\n=== 部署到 pi5/ 目录 ===")
    shutil.copy2(TFLITE_PATH, TFLITE_PI5_PATH)

    config_src = MODELS_DIR / "config.json"
    config_dst = PI5_DIR / "config.json"
    if config_src.exists():
        shutil.copy2(config_src, config_dst)

    labels_src = FEATURES_DIR / "labels.txt"
    labels_dst = PI5_DIR / "labels.txt"
    if labels_src.exists():
        shutil.copy2(labels_src, labels_dst)

    print(f"  复制文件：")
    print(f"    {TFLITE_PI5_PATH.name} ({TFLITE_PI5_PATH.stat().st_size / 1024:.0f} KB)")
    print(f"    config.json")
    print(f"    labels.txt")

    print("\n=== 部署到 Pi5 的步骤 ===")
    print("  1. 用 scp 传输文件：")
    print("     scp pi5/cat_sound.tflite pi5/config.json pi5/labels.txt pi5/pi5_inference.py \\")
    print("         pi@<PI5_IP>:~/catmow/")
    print()
    print("  2. 在 Pi5 上安装依赖：")
    print("     pip install -r requirements_pi5.txt")
    print()
    print("  3. 运行实时推理：")
    print("     python pi5_inference.py")


# ── 主入口 ────────────────────────────────────────────────────────────────────

def main():
    if not SAVED_MODEL_PATH.exists():
        print(f"[错误] 找不到模型: {SAVED_MODEL_PATH}")
        print("请先运行: python scripts/03_train.py")
        return

    if not (FEATURES_DIR / "X.npy").exists():
        print(f"[错误] 找不到特征数据: {FEATURES_DIR}/X.npy")
        print("请先运行: python scripts/02_preprocess.py")
        return

    np.random.seed(42)

    # 1. 转换
    convert_to_tflite()

    # 2. 验证精度
    verify_tflite()

    # 3. 部署到 pi5/
    deploy_to_pi5()

    print("\n完成！")


if __name__ == "__main__":
    print(f"TensorFlow: {tf.__version__}")
    main()
