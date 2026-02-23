"""
record_purr.py
录制猫呼噜（purr）声音，用于训练新模型的 purr 类别

使用方法：
  python record_purr.py              # 每段默认 90 秒
  python record_purr.py --duration 60 # 每段 60 秒
  python record_purr.py --list        # 查看音频设备

录制建议：
  - 在手机/电脑上播放猫嚎叫视频，让 Pi5 麦克风拾音
  - 每段 90 秒，录 8-10 段（合计约 12-15 分钟）
  - 可以录自己的猫，或播放 purr ASMR 视频
  - Ctrl+C 可随时中断并保存已录内容，再次 Ctrl+C 退出程序
"""

import argparse
import time
from pathlib import Path

import numpy as np
import sounddevice as sd
import scipy.io.wavfile as wavfile

SAVE_DIR    = Path(__file__).parent / "purr_recordings"
SAMPLE_RATE = 44100
CHANNELS    = 1


def list_devices():
    print("\n可用音频设备：")
    print(sd.query_devices())
    print()


def rms_bar(rms: float, width: int = 30) -> str:
    filled = min(int(rms * width / 0.15), width)
    bar = "█" * filled + "░" * (width - filled)
    if rms < 0.005:
        level = "静音"
    elif rms < 0.02:
        level = "微弱"
    elif rms < 0.08:
        level = "正常"
    else:
        level = "较响"
    return f"[{bar}] {rms:.4f}  {level}"


def save_wav(out_path: Path, audio: np.ndarray, n_recorded: int):
    """保存已录制的部分（支持提前中断保存）"""
    data = audio[:n_recorded, 0] if n_recorded > 0 else audio[:, 0]
    audio_int16 = (np.clip(data, -1.0, 1.0) * 32767).astype(np.int16)
    wavfile.write(str(out_path), SAMPLE_RATE, audio_int16)


def record_segment(seg_idx: int, duration: int, device=None) -> Path:
    """录制一段 purr 音频，使用 sd.rec() 简单可靠方式"""
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    out_path = SAVE_DIR / f"purr_{seg_idx:02d}.wav"

    n_samples = SAMPLE_RATE * duration

    print(f"\n  ─────────────────────────────────────────")
    print(f"  第 {seg_idx} 段  │  时长 {duration}s  │  保存 → {out_path.name}")
    print(f"  ─────────────────────────────────────────")
    print(f"  现在开始录制呼噜声，录音中...（Ctrl+C 提前结束并保存）\n")

    # sd.rec() 异步开始录音，audio 数组边录边写入
    audio = sd.rec(
        n_samples,
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype="float32",
        device=device,
    )

    n_recorded = n_samples  # 默认认为录完整
    start = time.time()

    try:
        while True:
            elapsed   = time.time() - start
            remaining = duration - elapsed
            if remaining <= 0:
                break

            # 估算当前已录位置，读取最近 0.1s 的 RMS
            pos          = min(int(elapsed * SAMPLE_RATE), n_samples - 1)
            recent_start = max(0, pos - int(SAMPLE_RATE * 0.1))
            chunk        = audio[recent_start:pos, 0] if pos > recent_start else np.zeros(1)
            rms          = float(np.sqrt(np.mean(chunk ** 2)))

            print(
                f"\r  ⏺ {rms_bar(rms)}   剩余 {remaining:5.1f}s",
                end="",
                flush=True,
            )
            time.sleep(0.1)

        sd.wait()  # 等待录音彻底完成

    except KeyboardInterrupt:
        # Ctrl+C：停止录音，保存已录部分
        sd.stop()
        n_recorded = min(int((time.time() - start) * SAMPLE_RATE), n_samples)
        actual_sec = n_recorded / SAMPLE_RATE
        print(f"\n\n  ⚠️  提前中断，已录 {actual_sec:.1f}s，正在保存...")

    # 统计有效内容比例
    frame_size    = SAMPLE_RATE // 10
    data_to_check = audio[:n_recorded, 0]
    n_frames      = len(data_to_check) // frame_size
    active_frames = sum(
        1 for i in range(n_frames)
        if np.sqrt(np.mean(data_to_check[i*frame_size:(i+1)*frame_size] ** 2)) > 0.005
    )
    active_pct = active_frames / max(n_frames, 1) * 100

    print(f"\r  ✅ 录音完成！                                           ")
    print(f"     有效内容占比: {active_pct:.0f}%  "
          f"({'充足 ✓' if active_pct >= 30 else '偏少，建议让猫发出更多呼噜声 ⚠️'})")

    save_wav(out_path, audio, n_recorded)
    print(f"     已保存 → {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="CatMow purr 录音工具")
    parser.add_argument("--duration", type=int, default=90,
                        help="每段录音时长（秒），默认 90s")
    parser.add_argument("--device",   type=int, default=None,
                        help="音频设备 ID（用 --list 查看）")
    parser.add_argument("--list",     action="store_true",
                        help="列出所有音频设备")
    args = parser.parse_args()

    if args.list:
        list_devices()
        return

    print("=" * 50)
    print("  CatMow  呼噜（purr）录音工具")
    print("=" * 50)
    print(f"\n  每段时长：{args.duration} 秒")
    print(f"  目标段数：8-10 段（合计 ≥12 分钟）")
    print(f"  保存目录：{SAVE_DIR}")
    print()
    print("  可搜索播放：cat purring ASMR / cat purring 1 hour / kitten purring")
    print("  Ctrl+C 一次：提前结束当前段并保存")
    print("  Ctrl+C 两次：退出程序")
    print()

    existing = sorted(SAVE_DIR.glob("purr_*.wav")) if SAVE_DIR.exists() else []
    seg_idx  = len(existing) + 1
    total_done = 0

    while True:
        print(f"\n  ── 准备第 {seg_idx} 段（按 Enter 开始，Ctrl+C 退出）──")
        try:
            input()
        except KeyboardInterrupt:
            break

        try:
            record_segment(seg_idx, args.duration, device=args.device)
            total_done += 1
            seg_idx    += 1
            total_min   = total_done * args.duration / 60
            print(f"\n  已录 {total_done} 段，合计约 {total_min:.1f} 分钟", end="")
            if total_done >= 8:
                print("  ✓ 数据量已足够，可继续或 Ctrl+C 退出")
            else:
                print(f"  （建议录到 8 段以上）")
        except KeyboardInterrupt:
            # 已在 record_segment 内部保存，这里只更新计数
            total_done += 1
            seg_idx    += 1
            print(f"\n  已录 {total_done} 段（含本次中断段）")
            print("  再按 Ctrl+C 退出，或按 Enter 继续录下一段")
            try:
                input()
            except KeyboardInterrupt:
                break

    all_files = sorted(SAVE_DIR.glob("purr_*.wav"))
    print(f"\n  ── 录音结束，共 {len(all_files)} 个文件 ──")
    for f in all_files:
        size_mb = f.stat().st_size / 1024 / 1024
        print(f"     {f.name}  {size_mb:.1f}MB")


if __name__ == "__main__":
    main()
