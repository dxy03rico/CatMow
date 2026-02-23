"""
pi5_inference.py
Raspberry Pi 5 实时猫声识别

功能：
  - 持续录音（滑动窗口）
  - 每 500ms 推理一次
  - 打印识别结果 + 置信度

硬件要求：
  - 麦克风（USB 或 I2S）
  - Raspberry Pi 5

安装依赖：
  pip install -r requirements_pi5.txt

使用方法：
  python pi5_inference.py
  python pi5_inference.py --device 1        # 指定音频设备 ID
  python pi5_inference.py --list-devices    # 列出所有音频设备
  python pi5_inference.py --threshold 0.6  # 调整置信度阈值
"""

import sys
import time
import json
import queue
import argparse
import threading
import numpy as np
from pathlib import Path
from collections import deque
from scipy import signal as scipy_signal

# ── 检查依赖 ──────────────────────────────────────────────────────────────────
try:
    import sounddevice as sd
except ImportError:
    print("[错误] 请安装 sounddevice: pip install sounddevice")
    sys.exit(1)

try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    try:
        # 备选：完整 TF 的 lite 模块
        import tensorflow.lite as tflite
    except ImportError:
        print("[错误] 请安装 tflite-runtime: pip install tflite-runtime")
        sys.exit(1)

# ── 配置 ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent
MODEL_PATH = SCRIPT_DIR / "cat_sound.tflite"
CONFIG_PATH = SCRIPT_DIR / "config.json"
LABELS_PATH = SCRIPT_DIR / "labels.txt"

# 音频参数（必须与训练时一致）
RECORD_SR = 44100       # Pi5 录音采样率（你的硬件设置）
TARGET_SR = 22050       # 模型期望采样率
CLIP_DURATION = 1.0     # 推理窗口（秒）
INFERENCE_INTERVAL = 0.5  # 推理间隔（秒）

# 显示参数
# 分类别置信度阈值：
#   other 不需要阈值（由模型直接输出，不显示）
#   yowl 适中，有音调特征较易区分
CONFIDENCE_THRESHOLD = 0.75   # 全局默认（--threshold 命令行参数使用此值）
CLASS_THRESHOLDS = {
    "meow": 0.60,   # 喵叫漏报成本高，阈值低一些
    "purr": 0.55,   # 降低：提高 purr 召回率（purr 目前漏报严重）
    "yowl": 0.88,   # 提高：减少 yowl 误报（purr 被错判为 yowl 问题）
    "other": 0.50,  # 背景噪音，低阈值即可（反正不显示给用户）
}
EMOJI = {
    "meow":  "😺",   # 日常喵叫
    "purr":  "😻",   # 呼噜
    "yowl":  "😿",   # 嚎叫/痛苦/恐惧
    "other": "🔇",   # 背景噪音（通常不会显示）
}


# ── 模型加载 ──────────────────────────────────────────────────────────────────

class CatSoundClassifier:
    def __init__(self, model_path: Path, config_path: Path, labels_path: Path):
        # 加载类别
        if labels_path.exists():
            with open(labels_path) as f:
                self.categories = [line.strip().split()[1] for line in f]
        elif config_path.exists():
            with open(config_path) as f:
                cfg = json.load(f)
            self.categories = cfg["categories"]
        else:
            self.categories = ["meow", "purr", "hiss", "growl", "other"]

        # 加载 TFLite 模型
        self.interpreter = tflite.Interpreter(model_path=str(model_path))
        self.interpreter.allocate_tensors()
        self.input_details  = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        # 模型期望的输入形状
        self.input_shape = self.input_details[0]["shape"]  # [1, 128, 128, 3]
        print(f"  模型加载成功")
        print(f"  类别: {self.categories}")
        print(f"  输入形状: {self.input_shape}")

    def predict(self, audio: np.ndarray) -> tuple[str, float, list[float]]:
        """
        推理单个音频片段

        Args:
            audio: float32 array, shape (TARGET_SR,), 已重采样

        Returns:
            (predicted_class, confidence, all_probs)
        """
        mel = self._audio_to_melspec(audio)
        mel_rgb = np.stack([mel, mel, mel], axis=-1)          # (128, 128, 3)
        inp = mel_rgb[np.newaxis, ...].astype(np.float32)      # (1, 128, 128, 3)

        self.interpreter.set_tensor(self.input_details[0]["index"], inp)
        self.interpreter.invoke()
        probs = self.interpreter.get_tensor(self.output_details[0]["index"])[0]

        pred_idx = int(np.argmax(probs))
        return self.categories[pred_idx], float(probs[pred_idx]), probs.tolist()

    def _audio_to_melspec(self, y: np.ndarray) -> np.ndarray:
        """音频 → Mel 频谱图 (128×128)，与训练时一致"""
        from scipy.signal import stft as scipy_stft

        # 手动计算 mel-spectrogram（避免在 Pi5 上依赖 librosa）
        n_fft = 1024
        hop_length = 256
        n_mels = 128
        img_size = 128

        # STFT
        freqs, times, Zxx = scipy_signal.stft(
            y,
            fs=TARGET_SR,
            nperseg=n_fft,
            noverlap=n_fft - hop_length,
            boundary=None,
        )
        power = np.abs(Zxx) ** 2

        # Mel 滤波器组
        mel_fb = _mel_filterbank(TARGET_SR, n_fft, n_mels)
        mel_spec = mel_fb @ power   # (n_mels, T)

        # dB 转换
        mel_db = 10.0 * np.log10(mel_spec + 1e-10)
        mel_db -= mel_db.max()  # 参考最大值

        # 裁剪/填充
        if mel_db.shape[1] < img_size:
            pad = img_size - mel_db.shape[1]
            mel_db = np.pad(mel_db, ((0, 0), (0, pad)), constant_values=-80)
        else:
            mel_db = mel_db[:, :img_size]

        # 归一化
        mel_db = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-8)
        return mel_db.astype(np.float32)


def _mel_filterbank(sr: int, n_fft: int, n_mels: int) -> np.ndarray:
    """生成 Mel 滤波器组矩阵"""
    fmax = sr / 2
    fmin = 0.0

    def hz_to_mel(f):
        return 2595.0 * np.log10(1.0 + f / 700.0)

    def mel_to_hz(m):
        return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

    mel_min = hz_to_mel(fmin)
    mel_max = hz_to_mel(fmax)
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = mel_to_hz(mel_points)

    bin_points = np.floor((n_fft + 1) * hz_points / sr).astype(int)
    n_freq_bins = n_fft // 2 + 1
    fb = np.zeros((n_mels, n_freq_bins))

    for m in range(1, n_mels + 1):
        f_m_minus = bin_points[m - 1]
        f_m       = bin_points[m]
        f_m_plus  = bin_points[m + 1]
        for k in range(f_m_minus, f_m):
            if f_m != f_m_minus:
                fb[m - 1, k] = (k - f_m_minus) / (f_m - f_m_minus)
        for k in range(f_m, f_m_plus):
            if f_m_plus != f_m:
                fb[m - 1, k] = (f_m_plus - k) / (f_m_plus - f_m)

    return fb


# ── 实时录音 ──────────────────────────────────────────────────────────────────

class AudioStream:
    def __init__(self, device_id: int | None, record_sr: int):
        self.record_sr = record_sr
        self.device_id = device_id
        self.audio_queue = queue.Queue()
        self._buffer = deque(maxlen=int(record_sr * CLIP_DURATION * 2))
        self._lock = threading.Lock()

    def _callback(self, indata, frames, time_info, status):
        # 不在 callback 里 print，避免 GIL 阻塞导致 overflow 加剧
        mono = indata[:, 0] if indata.ndim > 1 else indata.flatten()
        with self._lock:
            self._buffer.extend(mono)

    def get_clip(self) -> np.ndarray | None:
        """获取最新 1 秒的音频，重采样到 TARGET_SR"""
        clip_samples_rec = int(self.record_sr * CLIP_DURATION)
        with self._lock:
            if len(self._buffer) < clip_samples_rec:
                return None
            clip = np.array(list(self._buffer)[-clip_samples_rec:])

        # 重采样：44100 → 22050
        if self.record_sr != TARGET_SR:
            num_samples = int(len(clip) * TARGET_SR / self.record_sr)
            clip = scipy_signal.resample(clip, num_samples)

        return clip.astype(np.float32)

    def start(self) -> sd.InputStream:
        stream = sd.InputStream(
            samplerate=self.record_sr,
            channels=1,
            dtype="float32",
            device=self.device_id,
            blocksize=int(self.record_sr * 0.2),  # 200ms 块（减少 overflow）
            callback=self._callback,
        )
        stream.start()
        return stream


# ── 主推理循环 ────────────────────────────────────────────────────────────────

def run_inference(classifier: CatSoundClassifier, audio_stream: AudioStream, threshold: float):
    print("\n" + "="*50)
    print("  CatMow 实时猫声识别")
    print("  按 Ctrl+C 停止")
    print("="*50 + "\n")

    # ── 防抖参数 ──────────────────────────────────────────────────────────────
    # 连续 N 帧（每帧 INFERENCE_INTERVAL 秒）预测相同类别才输出，避免单帧误报
    SUSTAIN_FRAMES  = 2     # 至少连续 2 帧 = 1.0s 相同预测才确认
    COOLDOWN_SAME   = 3.0   # 相同类别上次输出后，冷却 3s 才再次输出（避免同一声音重复报告）
    COOLDOWN_DIFF   = 0.5   # 切换到不同类别，最短间隔 0.5s

    # ── 防抖状态 ──────────────────────────────────────────────────────────────
    pending_cat   = None    # 当前累积中的候选类别
    pending_count = 0       # 候选类别已连续出现的帧数
    pending_conf  = 0.0     # 候选事件中的最高置信度
    pending_probs = None    # 对应最高置信度帧的概率分布

    last_output_cat  = None   # 上次实际输出的类别
    last_output_time = 0.0    # 上次实际输出的时间戳

    with audio_stream.start():
        while True:
            time.sleep(INFERENCE_INTERVAL)

            clip = audio_stream.get_clip()
            if clip is None:
                continue

            # 三级过滤：RMS 能量 → 猫叫频段比 → 频谱平坦度
            rms = np.sqrt(np.mean(clip ** 2))
            if rms < 0.01:
                continue  # 太安静，跳过

            # 猫叫频段检测（100-3500 Hz）：
            # 下限从 300 Hz 降至 100 Hz，确保 purr（基频 25-150 Hz）的低次谐波
            # 被计入猫叫频段，而非被误判为"背景噪音频段"
            n_fft = 1024
            spectrum = np.abs(np.fft.rfft(clip[:n_fft]))
            freqs = np.fft.rfftfreq(n_fft, 1.0 / TARGET_SR)
            cat_band = (freqs >= 100) & (freqs <= 3500)
            noise_band = (freqs < 100) | (freqs > 3500)
            cat_energy = spectrum[cat_band].mean()
            noise_energy = spectrum[noise_band].mean() + 1e-8
            # 猫叫频段能量至少是背景频段的 1.5 倍才处理（适当放宽以兼容安静的 purr）
            if cat_energy / noise_energy < 1.5:
                continue

            # 频谱平坦度（Spectral Flatness）检测：
            # 背景噪音（风扇/空调）是宽频白噪，平坦度接近 1.0
            # 猫叫有共振峰（调性），平坦度较低（meow/purr<0.3, hiss<0.6）
            # 只在猫叫频段内计算，避免低频干扰
            cat_pow = (spectrum[cat_band].astype(np.float64) ** 2) + 1e-10
            geo_mean = np.exp(np.mean(np.log(cat_pow)))
            arith_mean = np.mean(cat_pow)
            flatness = float(geo_mean / arith_mean)
            # 平坦度 > 0.5 → 宽频背景噪音，跳过
            if flatness > 0.5:
                continue

            t0 = time.perf_counter()
            try:
                cat, conf, probs = classifier.predict(clip)
            except Exception as e:
                print(f"  [推理错误] {e}")
                continue
            latency_ms = (time.perf_counter() - t0) * 1000

            # 置信度过滤（分类别阈值）
            cls_threshold = CLASS_THRESHOLDS.get(cat, threshold)
            if conf < cls_threshold:
                cat = "other"

            # ── 防抖逻辑 ──────────────────────────────────────────────────────
            if cat != "other" and cat == pending_cat:
                # 同类别连续出现，累积帧数；记录最高置信度帧
                pending_count += 1
                if conf > pending_conf:
                    pending_conf  = conf
                    pending_probs = probs
            else:
                # 类别切换（或 other），重置候选
                pending_cat   = cat if cat != "other" else None
                pending_count = 1   if cat != "other" else 0
                pending_conf  = conf
                pending_probs = probs

            # 未达到最小持续帧数，继续等待
            if pending_count < SUSTAIN_FRAMES or pending_cat is None:
                continue

            # 检查冷却期
            now = time.time()
            if pending_cat == last_output_cat:
                if now - last_output_time < COOLDOWN_SAME:
                    continue  # 同类别冷却中，跳过
            else:
                if now - last_output_time < COOLDOWN_DIFF:
                    continue  # 类别切换最短间隔，跳过

            # ── 确认输出 ──────────────────────────────────────────────────────
            out_cat   = pending_cat
            out_conf  = pending_conf
            out_probs = pending_probs

            last_output_cat  = out_cat
            last_output_time = now
            pending_count    = 0    # 重置计数，避免立即重复触发

            emoji     = EMOJI.get(out_cat, "?")
            timestamp = time.strftime("%H:%M:%S")
            bars = " | ".join(
                f"{c}: {'█' * int(p * 10)}{int(p*100):3d}%"
                for c, p in zip(classifier.categories, out_probs)
            )
            print(
                f"  [{timestamp}] {emoji} {out_cat.upper():<6} {out_conf:.0%}"
                f"  推理: {latency_ms:.0f}ms"
                f"\n           {bars}\n"
            )


# ── 主入口 ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="CatMow 实时猫声识别")
    parser.add_argument(
        "--device", type=int, default=None,
        help="音频输入设备 ID（用 --list-devices 查看）"
    )
    parser.add_argument(
        "--list-devices", action="store_true",
        help="列出所有音频设备并退出"
    )
    parser.add_argument(
        "--threshold", type=float, default=CONFIDENCE_THRESHOLD,
        help=f"置信度阈值（默认 {CONFIDENCE_THRESHOLD}）"
    )
    args = parser.parse_args()

    if args.list_devices:
        print("可用音频设备：")
        print(sd.query_devices())
        return

    # 检查文件
    if not MODEL_PATH.exists():
        print(f"[错误] 模型文件不存在: {MODEL_PATH}")
        print("请先在 Mac 上运行训练流程，然后用 scp 传输模型")
        sys.exit(1)

    print("=== CatMow 初始化 ===")
    print(f"  模型: {MODEL_PATH.name}")
    print(f"  录音采样率: {RECORD_SR} Hz → 推理采样率: {TARGET_SR} Hz")
    print(f"  置信度阈值: meow≥{CLASS_THRESHOLDS['meow']:.0%}  purr≥{CLASS_THRESHOLDS['purr']:.0%}  yowl≥{CLASS_THRESHOLDS['yowl']:.0%}  other≥{CLASS_THRESHOLDS['other']:.0%}")

    classifier = CatSoundClassifier(MODEL_PATH, CONFIG_PATH, LABELS_PATH)
    audio_stream = AudioStream(args.device, RECORD_SR)

    try:
        run_inference(classifier, audio_stream, args.threshold)
    except KeyboardInterrupt:
        print("\n\n  已停止。")


if __name__ == "__main__":
    main()
