"""
02_preprocess.py
音频预处理：将原始音频转换为 Mel 频谱图，生成训练数据集

流程：
  原始音频 (.wav/.mp3/.ogg)
    → 重采样到 22050 Hz
    → 切成 1 秒片段（50% 重叠）
    → 生成 Mel 频谱图 (128×128)
    → 数据增强
    → 保存为 .npy 数组
    → 生成 dataset.csv

使用方法:
    python scripts/02_preprocess.py
"""

import os
import warnings
import numpy as np
import pandas as pd
import librosa
import librosa.display
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

# ── 配置 ──────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
PROCESSED_DIR = ROOT / "data" / "processed"
FEATURES_DIR = ROOT / "data" / "features"

TARGET_SR = 22050       # 重采样目标采样率（Hz）
CLIP_DURATION = 1.0     # 每个片段时长（秒）
HOP_DURATION = 0.5      # 滑窗步长（秒，50% 重叠）
N_MELS = 128            # Mel 频率通道数
N_FFT = 1024            # FFT 窗口大小
HOP_LENGTH = 256        # STFT 跳步
IMG_SIZE = 128          # 输出图像尺寸（正方形）

# 4 类方案：
#   meow       = 原 meow + hungry + happy（所有日常喵叫）
#   purr       = 呼噜声（原有 54 + 新录 8）
#   yowl       = 精选 growl（前15高质量）+ 新录 yowl（嚎叫/激动）
#                移除了 hiss（宽频低沉，与 purr 重叠，污染 yowl 类边界）
#   other      = 背景噪音（风扇/空调/日常环境声）
CATEGORIES = ["meow", "purr", "yowl", "other"]

# 源目录 → 目标类别 的映射（目录名 : 目标类别索引）
SOURCE_TO_LABEL = {
    "meow":       0,   # meow → meow
    "hungry":     0,   # hungry → meow
    "happy":      0,   # happy → meow
    "purr":       1,   # purr → purr
    # "hiss": 2 已移除 — hiss 是宽频低沉声，与 purr 频谱重叠，导致 purr→yowl 误判
    "growl":      2,   # growl → yowl（仅保留高质量前15个，filter_growl.py已过滤）
    "yowl":       2,   # yowl → yowl（新录，情绪激动音调声）
    "background": 3,   # background → other（背景噪音）
}

SUPPORTED_EXT = {".wav", ".mp3", ".ogg", ".flac", ".m4a"}

# 数据集划分比例
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15


# ── Mel 频谱图提取 ────────────────────────────────────────────────────────────

def audio_to_melspec(y: np.ndarray, sr: int) -> np.ndarray:
    """
    将音频片段转换为归一化 Mel 频谱图

    Returns:
        shape (IMG_SIZE, IMG_SIZE) float32，值域 [0, 1]
    """
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        fmax=sr // 2,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)

    # 裁剪/填充到固定宽度
    target_frames = IMG_SIZE
    if mel_db.shape[1] < target_frames:
        pad = target_frames - mel_db.shape[1]
        mel_db = np.pad(mel_db, ((0, 0), (0, pad)), mode="constant", constant_values=-80)
    else:
        mel_db = mel_db[:, :target_frames]

    # 归一化到 [0, 1]
    mel_db = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-8)
    return mel_db.astype(np.float32)


# ── 数据增强 ──────────────────────────────────────────────────────────────────

def augment_audio(y: np.ndarray, sr: int) -> list[np.ndarray]:
    """
    对音频片段进行增强，返回包含原始 + 增强版本的列表

    增强方法：
    - 加高斯噪声（模拟环境噪音）
    - 时间拉伸（0.9x 和 1.1x）
    - 音高偏移（±2 半音）
    """
    augmented = [y]  # 原始版本

    # 加噪声
    noise = np.random.normal(0, 0.005, len(y))
    augmented.append(y + noise)

    # 时间拉伸
    try:
        augmented.append(librosa.effects.time_stretch(y, rate=0.9))
        augmented.append(librosa.effects.time_stretch(y, rate=1.1))
    except Exception:
        pass

    # 音高偏移
    try:
        augmented.append(librosa.effects.pitch_shift(y, sr=sr, n_steps=2))
        augmented.append(librosa.effects.pitch_shift(y, sr=sr, n_steps=-2))
    except Exception:
        pass

    return augmented


# ── 音频切片处理 ──────────────────────────────────────────────────────────────

def process_audio_file(
    audio_path: Path,
    label: int,
    augment: bool = True,
) -> list[dict]:
    """
    读取音频文件，切片 + 可选增强，返回特征列表

    Returns:
        list of {"feature": np.ndarray, "label": int}
    """
    try:
        y, sr = librosa.load(str(audio_path), sr=TARGET_SR, mono=True)
    except Exception as e:
        print(f"  [跳过] 无法读取 {audio_path.name}: {e}")
        return []

    # 静音检测：能量过低的片段跳过
    if np.abs(y).max() < 0.005:
        return []

    clip_samples = int(CLIP_DURATION * TARGET_SR)
    hop_samples = int(HOP_DURATION * TARGET_SR)

    clips = []
    # 滑动窗口切片
    for start in range(0, len(y) - clip_samples + 1, hop_samples):
        clip = y[start : start + clip_samples]
        clips.append(clip)

    # 文件太短则整段使用（pad）
    if not clips:
        pad = clip_samples - len(y)
        clip = np.pad(y, (0, pad), mode="constant")
        clips.append(clip)

    results = []
    for clip in clips:
        if augment:
            variants = augment_audio(clip, TARGET_SR)
        else:
            variants = [clip]

        for variant in variants:
            # 确保长度一致
            if len(variant) > clip_samples:
                variant = variant[:clip_samples]
            elif len(variant) < clip_samples:
                variant = np.pad(variant, (0, clip_samples - len(variant)))

            mel = audio_to_melspec(variant, TARGET_SR)
            results.append({"feature": mel, "label": label})

    return results


# ── 主处理流程 ────────────────────────────────────────────────────────────────

def main():
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)

    all_features = []
    all_labels = []
    all_filenames = []

    print("=== 音频预处理 ===\n")
    print(f"  采样率: {TARGET_SR} Hz")
    print(f"  片段长度: {CLIP_DURATION}s，步长: {HOP_DURATION}s")
    print(f"  Mel 频谱: {N_MELS} mels → {IMG_SIZE}×{IMG_SIZE}\n")

    for source_dir, label_idx in SOURCE_TO_LABEL.items():
        cat_dir = PROCESSED_DIR / source_dir
        if not cat_dir.exists():
            print(f"  [跳过] {source_dir}/ 目录不存在")
            continue

        audio_files = [
            f for f in cat_dir.iterdir()
            if f.suffix.lower() in SUPPORTED_EXT
        ]

        if not audio_files:
            print(f"  [跳过] {source_dir}/ 没有音频文件")
            continue

        target_cat = CATEGORIES[label_idx]
        print(f"  处理 [{source_dir} → {target_cat}] {len(audio_files)} 个文件...")
        cat_count = 0

        # 测试/验证集不做数据增强，避免数据泄露
        train_files, temp_files = train_test_split(
            audio_files, test_size=(VAL_RATIO + TEST_RATIO), random_state=42
        )
        val_files, test_files = train_test_split(
            temp_files, test_size=TEST_RATIO / (VAL_RATIO + TEST_RATIO), random_state=42
        )

        split_map = (
            [(f, "train") for f in train_files]
            + [(f, "val") for f in val_files]
            + [(f, "test") for f in test_files]
        )

        for audio_path, split in tqdm(split_map, desc=f"  {source_dir}", leave=False):
            do_augment = (split == "train")
            items = process_audio_file(audio_path, label_idx, augment=do_augment)

            for item in items:
                all_features.append(item["feature"])
                all_labels.append(item["label"])
                all_filenames.append(
                    f"{split}/{target_cat}/{audio_path.stem}"
                )
                cat_count += 1

        print(f"  {source_dir} → {target_cat}: {cat_count} 个样本（含增强）")

    if not all_features:
        print("\n[错误] 没有找到任何音频文件！")
        print("请先运行: python scripts/01_download_data.py")
        return

    # 转换为 numpy 数组
    X = np.array(all_features)          # (N, 128, 128)
    y = np.array(all_labels)            # (N,)

    print(f"\n  总样本数: {len(X)}")
    print(f"  特征维度: {X.shape}")

    # 扩展维度为 3 通道（MobileNetV2 需要）
    X_rgb = np.stack([X, X, X], axis=-1)  # (N, 128, 128, 3)

    # 保存
    print("\n保存特征数据...")
    np.save(FEATURES_DIR / "X.npy", X_rgb)
    np.save(FEATURES_DIR / "y.npy", y)

    # 保存元数据 CSV
    df = pd.DataFrame({
        "filename": all_filenames,
        "label": y,
        "category": [CATEGORIES[l] for l in y],
        "split": [fn.split("/")[0] for fn in all_filenames],
    })
    df.to_csv(FEATURES_DIR / "dataset.csv", index=False)

    # 保存类别映射
    with open(FEATURES_DIR / "labels.txt", "w") as f:
        for i, cat in enumerate(CATEGORIES):
            f.write(f"{i} {cat}\n")

    # 统计
    print("\n=== 数据集统计 ===")
    for split in ["train", "val", "test"]:
        mask = df["split"] == split
        split_df = df[mask]
        print(f"\n  {split.upper()} ({len(split_df)} 样本):")
        for i, cat in enumerate(CATEGORIES):
            n = (split_df["label"] == i).sum()
            if n > 0:
                print(f"    {cat:<8} {n:>5}")

    print(f"\n  特征文件保存到: {FEATURES_DIR}/")
    print("  - X.npy       (特征矩阵)")
    print("  - y.npy       (标签)")
    print("  - dataset.csv (元数据)")
    print("  - labels.txt  (类别映射)")
    print("\n下一步运行：python scripts/03_train.py")


if __name__ == "__main__":
    main()
