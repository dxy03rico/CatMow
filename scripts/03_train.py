"""
03_train.py
模型训练：MobileNetV2 迁移学习 + 猫声分类

训练策略（两阶段）：
  Phase 1: 冻结 MobileNetV2 基础层，只训练分类头 (10 epochs)
  Phase 2: 解冻顶部 30 层，以小学习率微调 (40 epochs)

使用方法:
    python scripts/03_train.py
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks

# ── 配置 ──────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
FEATURES_DIR = ROOT / "data" / "features"
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

IMG_SIZE = 128
BATCH_SIZE = 32
PHASE1_EPOCHS = 10     # 冻结基础层训练轮数
PHASE2_EPOCHS = 40     # 微调轮数
PHASE1_LR = 1e-3
PHASE2_LR = 1e-4
UNFREEZE_LAYERS = 30   # 微调时解冻的顶部层数


# ── 数据加载 ──────────────────────────────────────────────────────────────────

def load_data():
    print("=== 加载数据 ===")
    X = np.load(FEATURES_DIR / "X.npy")   # (N, 128, 128, 3)
    y = np.load(FEATURES_DIR / "y.npy")   # (N,)
    df = pd.read_csv(FEATURES_DIR / "dataset.csv")

    # 按 split 分割
    train_mask = df["split"] == "train"
    val_mask = df["split"] == "val"
    test_mask = df["split"] == "test"

    X_train, y_train = X[train_mask], y[train_mask]
    X_val,   y_val   = X[val_mask],   y[val_mask]
    X_test,  y_test  = X[test_mask],  y[test_mask]

    # 读取类别
    with open(FEATURES_DIR / "labels.txt") as f:
        categories = [line.strip().split()[1] for line in f]
    num_classes = len(categories)

    print(f"  Train: {len(X_train)}  Val: {len(X_val)}  Test: {len(X_test)}")
    print(f"  类别 ({num_classes}): {categories}")
    print(f"  特征形状: {X_train.shape[1:]}")

    return X_train, y_train, X_val, y_val, X_test, y_test, categories


# ── 模型构建 ──────────────────────────────────────────────────────────────────

def build_model(num_classes: int) -> keras.Model:
    """
    MobileNetV2 迁移学习模型

    输入: (128, 128, 3) Mel 频谱图
    输出: num_classes 个类别的概率
    """
    base_model = keras.applications.MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False   # Phase 1: 冻结

    inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = keras.applications.mobilenet_v2.preprocess_input(inputs)   # [-1, 1]
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    return keras.Model(inputs, outputs), base_model


def oversample_to_balance(X, y):
    """
    过采样少数类，使每个类别样本数与最大类相同
    比 sample_weight 更直接，防止模型坍缩到多数类
    """
    classes, counts = np.unique(y, return_counts=True)
    max_count = counts.max()
    X_parts, y_parts = [X], [y]
    for cls, count in zip(classes, counts):
        if count < max_count:
            idx = np.where(y == cls)[0]
            extra = max_count - count
            chosen = np.random.choice(idx, size=extra, replace=True)
            X_parts.append(X[chosen])
            y_parts.append(y[chosen])
    X_bal = np.concatenate(X_parts)
    y_bal = np.concatenate(y_parts)
    # 打乱
    perm = np.random.permutation(len(X_bal))
    return X_bal[perm], y_bal[perm]


def make_dataset(X, y, shuffle=False) -> tf.data.Dataset:
    """构建 tf.data.Dataset，使用 sparse 整数标签"""
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(X), reshuffle_each_iteration=True)
    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds


# ── 训练 ──────────────────────────────────────────────────────────────────────

def train():
    X_train, y_train, X_val, y_val, X_test, y_test, categories = load_data()
    num_classes = len(categories)

    # 过采样平衡各类别（防止模型坍缩到多数类）
    print("\n  过采样前各类别数量:")
    for cls, cnt in zip(*np.unique(y_train, return_counts=True)):
        print(f"    {categories[cls]:<8} {cnt}")
    X_train, y_train = oversample_to_balance(X_train, y_train)
    print(f"  过采样后训练集大小: {len(X_train)}")

    ds_train = make_dataset(X_train, y_train, shuffle=True)
    ds_val   = make_dataset(X_val,   y_val)
    ds_test  = make_dataset(X_test,  y_test)

    model, base_model = build_model(num_classes)
    model.summary(line_length=70)

    # ── Phase 1: 训练分类头 ─────────────────────────────────────────────────
    print("\n=== Phase 1: 训练分类头（基础层冻结）===")
    model.compile(
        optimizer=keras.optimizers.Adam(PHASE1_LR),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    cb_phase1 = [
        callbacks.EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True, mode="min"
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6
        ),
    ]

    hist1 = model.fit(
        ds_train,
        validation_data=ds_val,
        epochs=PHASE1_EPOCHS,

        callbacks=cb_phase1,
        verbose=1,
    )

    # ── Phase 2: 微调顶部层 ─────────────────────────────────────────────────
    print(f"\n=== Phase 2: 微调顶部 {UNFREEZE_LAYERS} 层 ===")
    base_model.trainable = True
    # 只解冻最后 UNFREEZE_LAYERS 层
    for layer in base_model.layers[:-UNFREEZE_LAYERS]:
        layer.trainable = False

    trainable_count = sum(
        np.prod(v.shape) for v in model.trainable_variables
    )
    print(f"  可训练参数: {trainable_count:,}")

    model.compile(
        optimizer=keras.optimizers.Adam(PHASE2_LR),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    ckpt_path = MODELS_DIR / "best_model.keras"
    cb_phase2 = [
        callbacks.EarlyStopping(
            monitor="val_loss", patience=10, restore_best_weights=True, mode="min"
        ),
        callbacks.ModelCheckpoint(
            str(ckpt_path),
            monitor="val_loss",
            save_best_only=True,
            mode="min",
            verbose=1,
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.3, patience=5, min_lr=1e-7
        ),
    ]

    hist2 = model.fit(
        ds_train,
        validation_data=ds_val,
        epochs=PHASE2_EPOCHS,

        callbacks=cb_phase2,
        verbose=1,
    )

    # ── 测试集评估 ──────────────────────────────────────────────────────────
    print("\n=== 测试集评估 ===")
    test_loss, test_acc = model.evaluate(ds_test, verbose=0)
    print(f"  Test Loss:     {test_loss:.4f}")
    print(f"  Test Accuracy: {test_acc:.4f} ({test_acc*100:.1f}%)")

    # 逐类别准确率
    y_pred = model.predict(ds_test, verbose=0).argmax(axis=1)
    print("\n  各类别准确率:")
    for i, cat in enumerate(categories):
        mask = y_test == i
        if mask.sum() == 0:
            continue
        acc = (y_pred[mask] == i).mean()
        bar = "█" * int(acc * 20)
        print(f"    {cat:<8} {acc:.2%}  {bar}")

    # ── 保存 ──────────────────────────────────────────────────────────────
    # 保存为 .keras 格式（Keras 3 要求）
    saved_model_path = MODELS_DIR / "saved_model.keras"
    model.save(str(saved_model_path))
    # 同时导出 SavedModel 格式供 TFLite 转换使用
    model.export(str(MODELS_DIR / "saved_model_export"))
    print(f"\n  模型保存到: {saved_model_path}")

    # 保存训练配置
    config = {
        "categories": categories,
        "num_classes": num_classes,
        "img_size": IMG_SIZE,
        "target_sr": 22050,
        "n_mels": 128,
        "n_fft": 1024,
        "hop_length": 256,
        "clip_duration": 1.0,
        "test_accuracy": float(test_acc),
    }
    with open(MODELS_DIR / "config.json", "w") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    print(f"  配置保存到: {MODELS_DIR}/config.json")

    # ── 绘制训练曲线 ────────────────────────────────────────────────────────
    _plot_history(hist1, hist2, MODELS_DIR / "training_curves.png")

    print("\n下一步运行：python scripts/04_convert_tflite.py")
    return model


def _plot_history(hist1, hist2, save_path: Path):
    """合并两阶段训练历史并保存图表"""
    acc = hist1.history["accuracy"] + hist2.history["accuracy"]
    val_acc = hist1.history["val_accuracy"] + hist2.history["val_accuracy"]
    loss = hist1.history["loss"] + hist2.history["loss"]
    val_loss = hist1.history["val_loss"] + hist2.history["val_loss"]
    p1_end = len(hist1.history["accuracy"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("CatMow Training Curves")

    ax1.plot(acc, label="Train Acc")
    ax1.plot(val_acc, label="Val Acc")
    ax1.axvline(p1_end, color="gray", linestyle="--", label="Phase 2 start")
    ax1.set_title("Accuracy")
    ax1.set_xlabel("Epoch")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(loss, label="Train Loss")
    ax2.plot(val_loss, label="Val Loss")
    ax2.axvline(p1_end, color="gray", linestyle="--", label="Phase 2 start")
    ax2.set_title("Loss")
    ax2.set_xlabel("Epoch")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    print(f"  训练曲线保存到: {save_path}")


if __name__ == "__main__":
    print(f"TensorFlow: {tf.__version__}")
    gpus = tf.config.list_physical_devices("GPU")
    print(f"GPU: {gpus if gpus else '未检测到（使用 CPU）'}")
    train()
