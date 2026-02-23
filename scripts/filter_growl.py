"""
filter_growl.py
按频谱强度过滤 growl 训练数据

策略：
  - 保留：频率高、情绪激动的（spectral centroid 高、调制明显、平坦度低）
  - 归档：平淡的宽频低沉声（与 purr 频谱重叠，污染 yowl 类边界）

使用方法:
    python scripts/filter_growl.py          # 默认保留前 15 个
    python scripts/filter_growl.py --keep 12   # 自定义保留数量
    python scripts/filter_growl.py --dry-run   # 只看报告，不移动文件
"""

import argparse
import shutil
import numpy as np
from pathlib import Path

try:
    import librosa
except ImportError:
    print("[错误] 请安装 librosa: pip install librosa")
    raise

ROOT = Path(__file__).parent.parent
GROWL_DIR   = ROOT / "data" / "processed" / "growl"
EXCLUDE_DIR = ROOT / "data" / "processed" / "growl_excluded"


def score_file(path: Path) -> dict:
    """计算单个 growl 文件的频谱强度综合评分"""
    y, sr = librosa.load(str(path), sr=22050, mono=True)
    duration = len(y) / sr

    # 频谱质心（主频率高 → 更像 yowl 音调）
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    centroid_mean = float(centroid.mean())
    centroid_std  = float(centroid.std())

    # RMS 能量及动态范围
    rms = librosa.feature.rms(y=y)[0]
    rms_mean = float(rms.mean())
    rms_std  = float(rms.std())

    # 频谱平坦度（越低越有调性；纯噪声接近 1.0）
    flatness = float(librosa.feature.spectral_flatness(y=y)[0].mean())

    # 综合评分：高频 × 调制程度 / 平坦度
    #   - centroid_mean 高 → 音调高 → 更像 yowl
    #   - centroid_std/centroid_mean 高 → 频率调制强 → 更有情绪
    #   - flatness 低 → 更有调性 → 更像猫叫而非噪声
    score = centroid_mean * (1 + centroid_std / (centroid_mean + 1e-8)) / (flatness * 10000 + 1)

    return {
        "file":          path.name,
        "path":          path,
        "duration":      duration,
        "centroid_mean": centroid_mean,
        "centroid_std":  centroid_std,
        "rms_mean":      rms_mean,
        "rms_std":       rms_std,
        "flatness":      flatness,
        "score":         score,
    }


def main():
    parser = argparse.ArgumentParser(description="Filter growl files by spectral intensity")
    parser.add_argument("--keep",    type=int,  default=15,    help="保留前 N 个高质量文件 (默认 15)")
    parser.add_argument("--dry-run", action="store_true",      help="只打印报告，不移动文件")
    args = parser.parse_args()

    # ── 收集并评分 ────────────────────────────────────────────────────────────
    files = sorted(GROWL_DIR.glob("*.mp3"))
    if not files:
        print(f"[错误] 未找到 .mp3 文件：{GROWL_DIR}")
        return

    print(f"正在分析 {len(files)} 个 growl 文件……\n")
    results = []
    for f in files:
        try:
            r = score_file(f)
            results.append(r)
            print(f"  ✓ {f.name:<30}  centroid={r['centroid_mean']:>7.1f}Hz  flatness={r['flatness']:.4f}  score={r['score']:>8.2f}")
        except Exception as e:
            print(f"  ✗ {f.name}: {e}")

    # 按评分降序排列
    results.sort(key=lambda x: x["score"], reverse=True)

    keep_n    = min(args.keep, len(results))
    exclude_n = len(results) - keep_n

    # ── 打印分类报告 ──────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  评分排名（共 {len(results)} 个，保留前 {keep_n} 个，归档后 {exclude_n} 个）")
    print(f"{'='*70}")
    print(f"{'Rank':<5} {'文件':<30} {'质心Hz':>8} {'调制':>7} {'平坦度':>8} {'评分':>9}")
    print(f"{'─'*70}")

    for i, r in enumerate(results):
        tag = "✅ 保留" if i < keep_n else "📦 归档"
        sep = "─" * 70 if i == keep_n - 1 else ""
        print(f"  {i+1:<3} {r['file']:<30} {r['centroid_mean']:>8.1f} {r['centroid_std']:>7.1f} {r['flatness']:>8.4f} {r['score']:>9.2f}  {tag}")
        if sep:
            print(f"{'─'*70}  ← 分割线（保留/归档）")

    kept    = [r for r in results[:keep_n]]
    exclude = [r for r in results[keep_n:]]

    print(f"\n  保留 {len(kept)} 个文件，分值范围: {kept[0]['score']:.2f} → {kept[-1]['score']:.2f}")
    if exclude:
        print(f"  归档 {len(exclude)} 个文件，分值范围: {exclude[0]['score']:.2f} → {exclude[-1]['score']:.2f}")

    if args.dry_run:
        print("\n[dry-run] 未移动任何文件。去掉 --dry-run 参数后执行实际操作。")
        return

    # ── 执行文件移动 ──────────────────────────────────────────────────────────
    if not exclude:
        print("\n没有文件需要归档。")
        return

    EXCLUDE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\n归档目录: {EXCLUDE_DIR}")
    moved = 0
    for r in exclude:
        dst = EXCLUDE_DIR / r["file"]
        shutil.move(str(r["path"]), str(dst))
        print(f"  移动: {r['file']} → growl_excluded/")
        moved += 1

    print(f"\n完成！已归档 {moved} 个低质量 growl 文件。")
    print(f"  data/processed/growl/          → {keep_n} 个文件（高质量，参与训练）")
    print(f"  data/processed/growl_excluded/ → {moved} 个文件（归档备份）")
    print(f"\n下一步：")
    print(f"  1. 更新 scripts/02_preprocess.py（移除 hiss 映射）")
    print(f"  2. 运行 python scripts/02_preprocess.py")
    print(f"  3. 运行 python scripts/03_train.py")


if __name__ == "__main__":
    main()
