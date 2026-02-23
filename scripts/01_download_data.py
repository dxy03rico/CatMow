"""
01_download_data.py
数据下载脚本：从 ESC-50 和 FreeSound 下载猫声数据

数据来源：
1. ESC-50: 开放数据集，直接下载，含 cat 类别 (~40 clips)
2. FreeSound: 需要免费注册获取 API Key，搜索各类猫声

使用方法:
    python scripts/01_download_data.py
    python scripts/01_download_data.py --freesound-key YOUR_API_KEY

FreeSound API Key 获取：
    1. 注册账号: https://freesound.org/apiv2/apply/
    2. 创建应用，获得 Client ID & Secret
    3. Token: https://freesound.org/apiv2/oauth2/logout_and_authorize/
"""

import os
import sys
import csv
import shutil
import zipfile
import argparse
import requests
import tarfile
from pathlib import Path
from tqdm import tqdm

# ── 路径配置 ──────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
ESC50_DIR = RAW_DIR / "esc50"

CATEGORIES = ["meow", "purr", "hiss", "growl", "hungry", "happy", "other"]

# FreeSound 搜索关键词 → 本地类别映射
FREESOUND_QUERIES = {
    "meow":   ["cat meow", "kitten meow", "cat calling", "cat crying"],
    "purr":   ["cat purring", "cat purr"],
    "hiss":   ["cat hissing", "cat hiss angry"],
    "growl":  ["cat growling", "cat snarling"],
    "hungry": ["cat hungry meow", "cat food meow", "cat begging", "kitten crying hungry"],
    "happy":  ["cat happy chirp", "cat trilling", "cat greeting", "happy cat sounds"],
    "other":  ["cat sound", "cat vocalization"],
}

# FreeSound 每个关键词最多下载多少条（防止超限）
MAX_PER_QUERY = 30


# ── 工具函数 ──────────────────────────────────────────────────────────────────

def download_file(url: str, dest: Path, desc: str = "") -> bool:
    """带进度条地下载文件"""
    try:
        resp = requests.get(url, stream=True, timeout=30)
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        dest.parent.mkdir(parents=True, exist_ok=True)
        with open(dest, "wb") as f, tqdm(
            total=total, unit="B", unit_scale=True, desc=desc or dest.name
        ) as pbar:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))
        return True
    except Exception as e:
        print(f"  [错误] 下载失败 {url}: {e}")
        return False


# ── ESC-50 ────────────────────────────────────────────────────────────────────

def download_esc50():
    """下载 ESC-50 数据集并提取猫声"""
    print("\n=== [1/2] 下载 ESC-50 数据集 ===")
    zip_path = RAW_DIR / "ESC-50.zip"
    url = "https://github.com/karoldvl/ESC-50/archive/master.zip"

    if ESC50_DIR.exists() and any(ESC50_DIR.iterdir()):
        print("  ESC-50 已存在，跳过下载")
    else:
        print(f"  下载到 {zip_path} ...")
        if not download_file(url, zip_path, "ESC-50"):
            print("  ESC-50 下载失败，请手动下载：", url)
            return
        print("  解压中...")
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(RAW_DIR)
        extracted = RAW_DIR / "ESC-50-master"
        if extracted.exists():
            extracted.rename(ESC50_DIR)
        zip_path.unlink(missing_ok=True)
        print("  ESC-50 解压完成")

    _extract_cat_from_esc50()


def _extract_cat_from_esc50():
    """从 ESC-50 中提取 cat 类别的音频，放入 data/processed/meow/"""
    meta_path = ESC50_DIR / "meta" / "esc50.csv"
    audio_dir = ESC50_DIR / "audio"
    dest_dir = PROCESSED_DIR / "meow"
    dest_dir.mkdir(parents=True, exist_ok=True)

    if not meta_path.exists():
        print("  [警告] 找不到 ESC-50 metadata，跳过")
        return

    copied = 0
    with open(meta_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["category"] == "cat":
                src = audio_dir / row["filename"]
                dst = dest_dir / f"esc50_{row['filename']}"
                if src.exists() and not dst.exists():
                    shutil.copy2(src, dst)
                    copied += 1

    print(f"  ESC-50 猫声: 提取 {copied} 条 → data/processed/meow/")


# ── FreeSound ─────────────────────────────────────────────────────────────────

class FreeSoundClient:
    BASE = "https://freesound.org/apiv2"

    def __init__(self, api_key: str):
        self.headers = {"Authorization": f"Token {api_key}"}

    def search(self, query: str, max_results: int = 30):
        """搜索音效，返回结果列表"""
        params = {
            "query": query,
            "fields": "id,name,previews,duration,tags",
            "filter": "duration:[0.5 TO 8]",  # 0.5~8 秒
            "page_size": min(max_results, 150),
        }
        try:
            resp = requests.get(
                f"{self.BASE}/search/text/",
                headers=self.headers,
                params=params,
                timeout=15,
            )
            resp.raise_for_status()
            return resp.json().get("results", [])
        except Exception as e:
            print(f"  [错误] FreeSound 搜索失败 ({query}): {e}")
            return []

    def download_preview(self, sound: dict, dest: Path) -> bool:
        """下载 HQ preview（无需 OAuth，MP3 格式）"""
        url = sound.get("previews", {}).get("preview-hq-mp3")
        if not url:
            return False
        return download_file(url, dest)


def download_freesound(api_key: str):
    """使用 FreeSound API 下载各类猫声"""
    print("\n=== [2/2] FreeSound 下载 ===")
    client = FreeSoundClient(api_key)

    total_downloaded = 0
    for category, queries in FREESOUND_QUERIES.items():
        dest_dir = PROCESSED_DIR / category
        dest_dir.mkdir(parents=True, exist_ok=True)
        cat_count = 0

        for query in queries:
            print(f"  搜索: '{query}' → {category}/")
            results = client.search(query, max_results=MAX_PER_QUERY)
            if not results:
                continue

            for sound in results:
                sound_id = sound["id"]
                dest = dest_dir / f"fs_{sound_id}.mp3"
                if dest.exists():
                    continue
                if client.download_preview(sound, dest):
                    cat_count += 1
                    total_downloaded += 1

        print(f"  {category}: 共 {cat_count} 条")

    print(f"\n  FreeSound 共下载 {total_downloaded} 条音频")


# ── Zenodo CatMeows ───────────────────────────────────────────────────────────
# 数据集: "Meows" – 440 条有上下文标注的猫叫声
# 来源: https://zenodo.org/record/4008297
# 上下文映射:
#   food     → hungry（猫在要食物时叫）
#   brushing → happy （被梳毛时叫/呼噜）
#   waiting  → meow  （等待/孤独时叫）

ZENODO_RECORD_ID = "4008297"
ZENODO_CONTEXT_MAP = {
    "food":     "hungry",
    "brushing": "happy",
    "waiting":  "meow",
}


def download_zenodo_catmeows():
    """下载 Zenodo CatMeows 数据集并按上下文分类"""
    print("\n=== [2/3] 下载 Zenodo CatMeows 数据集 ===")

    zenodo_dir = RAW_DIR / "zenodo_catmeows"

    # 通过 Zenodo REST API 获取文件列表
    api_url = f"https://zenodo.org/api/records/{ZENODO_RECORD_ID}"
    try:
        resp = requests.get(api_url, timeout=15)
        resp.raise_for_status()
        record = resp.json()
    except Exception as e:
        print(f"  [错误] 无法访问 Zenodo API: {e}")
        return

    files = record.get("files", [])
    if not files:
        print("  [警告] Zenodo 记录中没有文件")
        return

    print(f"  找到 {len(files)} 个文件，开始下载...")
    zenodo_dir.mkdir(parents=True, exist_ok=True)

    downloaded_files = []
    for file_info in files:
        fname = file_info["key"]
        furl  = file_info["links"]["self"]
        fpath = zenodo_dir / fname

        if fpath.exists():
            print(f"  已存在: {fname}")
            downloaded_files.append(fpath)
            continue

        print(f"  下载: {fname}")
        if download_file(furl, fpath, fname):
            downloaded_files.append(fpath)

    # 解压并按上下文分类
    _classify_zenodo_files(zenodo_dir, downloaded_files)


def _classify_zenodo_files(zenodo_dir: Path, archive_files: list):
    """
    解压 Zenodo 压缩包，按文件名前缀分到对应类别文件夹

    CatMeows 数据集命名规范:
      F_xxx → food context    → hungry
      B_xxx → brushing context → happy
      I_xxx → isolation context → meow
    """
    # 文件名首字母前缀 → 类别
    PREFIX_MAP = {
        "F_": "hungry",
        "B_": "happy",
        "I_": "meow",
    }
    total_classified = {}

    for archive in archive_files:
        suffix = archive.suffix.lower()

        # 解压
        extract_dir = zenodo_dir / archive.stem
        if not extract_dir.exists():
            try:
                if suffix == ".zip":
                    with zipfile.ZipFile(archive, "r") as z:
                        z.extractall(extract_dir)
                elif suffix in (".tar", ".gz", ".tgz"):
                    with tarfile.open(archive, "r:*") as t:
                        t.extractall(extract_dir)
                else:
                    extract_dir = zenodo_dir
            except Exception as e:
                print(f"  [警告] 无法解压 {archive.name}: {e}")
                continue

        audio_files = list(extract_dir.rglob("*.wav")) + list(extract_dir.rglob("*.mp3"))
        if not audio_files and suffix in (".wav", ".mp3"):
            audio_files = [archive]

        for audio_path in audio_files:
            # 按文件名前缀判断上下文
            fname_upper = audio_path.name.upper()
            matched_cat = "meow"  # 默认
            for prefix, cat in PREFIX_MAP.items():
                if fname_upper.startswith(prefix):
                    matched_cat = cat
                    break

            dest_dir = PROCESSED_DIR / matched_cat
            dest_dir.mkdir(parents=True, exist_ok=True)
            dst = dest_dir / f"zenodo_{audio_path.name}"
            if not dst.exists():
                shutil.copy2(audio_path, dst)
                total_classified[matched_cat] = total_classified.get(matched_cat, 0) + 1

    print("  Zenodo 分类结果:")
    for cat, count in sorted(total_classified.items()):
        print(f"    {cat:<8} {count} 条")


# ── 统计汇总 ──────────────────────────────────────────────────────────────────

def print_summary():
    print("\n=== 数据统计 ===")
    total = 0
    for cat in CATEGORIES:
        d = PROCESSED_DIR / cat
        files = list(d.glob("*.*")) if d.exists() else []
        count = len(files)
        total += count
        bar = "█" * min(count, 50)
        print(f"  {cat:<8} {count:>4} 条  {bar}")
    print(f"  {'合计':<8} {total:>4} 条")
    print()
    if total < 200:
        print("  [提示] 数据量偏少（建议每类 ≥ 100 条）。")
        print("         可手动录制或从以下网站补充：")
        print("         - https://xeno-canto.org (鸟/动物声音)")
        print("         - https://www.zapsplat.com")
        print("         - https://soundbible.com")


# ── 主入口 ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="下载猫声训练数据")
    parser.add_argument(
        "--freesound-key",
        default="",
        help="FreeSound API Token（从 freesound.org/apiv2/apply/ 申请）",
    )
    args = parser.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for cat in CATEGORIES:
        (PROCESSED_DIR / cat).mkdir(parents=True, exist_ok=True)

    # 1. 下载 ESC-50（无需 API Key）
    download_esc50()

    # 2. 下载 Zenodo CatMeows（无需 API Key，含 hungry/happy 标注）
    download_zenodo_catmeows()

    # 3. 下载 FreeSound（需要 API Key）
    if args.freesound_key:
        download_freesound(args.freesound_key)
    else:
        print("\n=== [3/3] FreeSound 跳过 ===")
        print("  未提供 API Key，跳过 FreeSound 下载")
        print("  获取方法：https://freesound.org/apiv2/apply/")
        print("  获取后运行: python scripts/01_download_data.py --freesound-key YOUR_KEY")

    print_summary()

    print("完成！下一步运行：")
    print("  python scripts/02_preprocess.py")


if __name__ == "__main__":
    main()
