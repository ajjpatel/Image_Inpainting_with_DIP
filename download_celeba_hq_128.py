import os
import sys
import zipfile
import urllib.request
import subprocess
from tqdm import tqdm
from PIL import Image

CELEBA_HQ_URL = "https://data.vision.ee.ethz.ch/cvl/celeba-hq/celeba-hq.zip"
CELEBA_HQ_ZIP = "celeba-hq.zip"
CELEBA_HQ_DIR = "celeba-hq"
CELEBA_HQ_128_DIR = "celeba-hq-128"


def download_file(url, dest):
    class DownloadProgressBar(tqdm):
        def update_to(self, b=1, bsize=1, tsize=None):
            if tsize is not None:
                self.total = tsize
            self.update(b * bsize - self.n)
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=dest) as t:
        urllib.request.urlretrieve(url, filename=dest, reporthook=t.update_to)

def extract_zip(zip_path, extract_to):
    print(f"Extracting {zip_path} to {extract_to} ...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

def check_kaggle_cli():
    from shutil import which
    if which('kaggle') is None:
        print("[ERROR] Kaggle CLI is not installed. Please run: pip install kaggle")
        sys.exit(1)

def check_kaggle_json():
    kaggle_json = os.path.expanduser('~/.kaggle/kaggle.json')
    if not os.path.exists(kaggle_json):
        print("[ERROR] Kaggle API credentials not found at ~/.kaggle/kaggle.json.")
        print("Go to https://www.kaggle.com/settings, create a new API token, and place kaggle.json in ~/.kaggle/")
        sys.exit(1)

def download_from_kaggle(dataset, download_dir):
    os.makedirs(download_dir, exist_ok=True)
    print(f"Downloading {dataset} from Kaggle to {download_dir} ...")
    cmd = [
        "kaggle", "datasets", "download", "-d", dataset, "-p", download_dir, "--unzip"
    ]
    try:
        subprocess.check_call(cmd)
    except subprocess.CalledProcessError as e:
        print("[ERROR] Kaggle download failed. Check your internet connection and Kaggle credentials.")
        sys.exit(1)
    print("Download and extraction complete.")

def downsample_images(src_dir, dst_dir, size=128):
    os.makedirs(dst_dir, exist_ok=True)
    files = [f for f in os.listdir(src_dir) if f.endswith('.png') or f.endswith('.jpg')]
    print(f"Downsampling {len(files)} images to {size}x{size} ...")
    for fname in tqdm(files):
        src_path = os.path.join(src_dir, fname)
        dst_path = os.path.join(dst_dir, fname)
        img = Image.open(src_path).convert('RGB')
        img = img.resize((size, size), Image.LANCZOS)
        img.save(dst_path)

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Download CelebA-HQ 256x256 from Kaggle and resize to 128x128")
    parser.add_argument('--root', type=str, default='data', help='Root directory for dataset')
    parser.add_argument('--kaggle-dataset', type=str, default='badasstechie/celebahq-resized-256x256', help='Kaggle dataset name')
    args = parser.parse_args()
    root = args.root
    kaggle_dataset = args.kaggle_dataset

    download_dir = os.path.join(root, "celebahq-256")
    output_dir = os.path.join(root, "celebahq-128")

    check_kaggle_cli()
    check_kaggle_json()

    if not os.path.exists(download_dir) or not os.listdir(download_dir):
        download_from_kaggle(kaggle_dataset, download_dir)
    else:
        print(f"{download_dir} already exists and is not empty, skipping download.")

    downsample_images(download_dir, output_dir, size=128)
    print("--------------Done------------------")

if __name__ == '__main__':
    main() 