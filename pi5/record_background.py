"""
record_background.py
录制家庭背景噪音，用于训练 "other" 类别

使用方法：
  python record_background.py              # 默认录 5 分钟
  python record_background.py --duration 3 # 录 3 分钟
  python record_background.py --list       # 查看音频设备
"""

import argparse
import time
import sys
from pathlib import Path
import numpy as np
import sounddevice as sd
import scipy.io.wavfile as wavfile

SAVE_DIR = Path(__file__).parent / "background_recordings"
SAMPLE_RATE = 44100

SCENES = {
    "1": "ac_fan",          # 空调/风扇
    "2": "tv_speech",       # 电视/说话声
    "3": "keyboard_steps",  # 键盘/脚步声
    "4": "quiet",           # 安静环境
    "5": "other",           # 其他背景
}

SCENE_NAMES = {
    "ac_fan":         "空调/风扇运行声",
    "tv_speech":      "电视/说话声背景",
    "keyboard_steps": "键盘打字/走路声",
    "quiet":          "安静环境（轻微底噪）",
    "other":          "其他背景噪音",
}


def list_devices():
    print("\n可用音频设备：")
    print(sd.query_devices())


def record_once(scene_name: str, duration: int, device=None) -> Path:
    """录制一段背景噪音并保存为 wav"""
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    # 自动编号：避免覆盖已有录音
    existing = list(SAVE_DIR.glob(f"{scene_name}_*.wav"))
    idx = len(existing) + 1
    out_path = SAVE_DIR / f"{scene_name}_{idx:02d}.wav"

    n_samples = SAMPLE_RATE * duration

    print(f"\n  场景：{SCENE_NAMES.get(scene_name, scene_name)}")
    print(f"  时长：{duration} 秒")
    print(f"  保存：{out_path.name}")
    print(f"\n  ⏺  录音中", end="", flush=True)

    # 倒计时提示
    def progress(elapsed):
        remaining = duration - int(elapsed)
        if elapsed < duration:
            print(f"\r  ⏺  录音中... 剩余 {remaining:3d}s", end="", flush=True)

    start = time.time()
    audio = sd.rec(
        n_samples,
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        device=device,
    )

    while not sd.get_stream().stopped:
        elapsed = time.time() - start
        if elapsed >= duration:
            break
        progress(elapsed)
        time.sleep(0.5)

    sd.wait()
    print(f"\r  ✅ 录音完成！已保存 → {out_path}")

    # 检查音量
    rms = np.sqrt(np.mean(audio ** 2))
    print(f"     平均音量 RMS: {rms:.4f}", end="")
    if rms < 0.0005:
        print("  ⚠️  音量极低，请检查麦克风是否正常")
    elif rms > 0.1:
        print("  ⚠️  音量过高，可能录到了非背景声音")
    else:
        print("  ✓ 音量正常")

    # 保存
    audio_int16 = (audio * 32767).astype(np.int16)
    wavfile.write(str(out_path), SAMPLE_RATE, audio_int16)
    return out_path


def main():
    parser = argparse.ArgumentParser(description="录制背景噪音")
    parser.add_argument("--duration", type=int, default=300, help="录音时长（秒），默认 300s = 5分钟")
    parser.add_argument("--device", type=int, default=None, help="音频设备 ID")
    parser.add_argument("--list", action="store_true", help="列出音频设备")
    args = parser.parse_args()

    if args.list:
        list_devices()
        return

    print("=" * 50)
    print("  CatMow 背景噪音录制工具")
    print("=" * 50)
    print("\n请选择当前录制的场景（输入数字）：\n")
    for k, v in SCENES.items():
        print(f"  {k}. {SCENE_NAMES[v]}")
    print("  0. 退出")

    while True:
        print()
        choice = input("场景编号 > ").strip()

        if choice == "0":
            print("退出")
            break

        if choice not in SCENES:
            print("  无效编号，请重新输入")
            continue

        scene = SCENES[choice]
        print(f"\n  将要录制：{SCENE_NAMES[scene]}，时长 {args.duration} 秒")
        print("  请现在布置好录音环境，准备好后按 Enter 开始...")
        input()

        record_once(scene, args.duration, device=args.device)

        print("\n继续录制其他场景吗？（输入场景编号，或 0 退出）")
        for k, v in SCENES.items():
            print(f"  {k}. {SCENE_NAMES[v]}")


if __name__ == "__main__":
    main()
