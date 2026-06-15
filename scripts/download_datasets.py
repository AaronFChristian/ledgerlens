"""Download CORD v2 and SROIE datasets for LedgerLens evaluation.

Run once before evaluating:
    python scripts/download_datasets.py

Downloads to:
    data/samples/cord/      — invoice images
    data/ground_truth/cord/ — ground truth JSON files
"""

import json
import sys
from pathlib import Path

from tqdm import tqdm

# Allow running from project root
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

DATA_DIR = Path("data")
SAMPLES_DIR = DATA_DIR / "samples"
GT_DIR = DATA_DIR / "ground_truth"


def download_cord(n_samples: int = 100) -> None:
    """Download CORD v2 (Clova AI receipt OCR dataset, CC BY 4.0).

    HuggingFace: naver-clova-ix/cord-v2
    Structure: image (PIL) + ground_truth (JSON string with gt_parse)
    """
    from datasets import load_dataset

    print(f"\n📥 Downloading CORD v2 ({n_samples} samples)...")

    cord_img_dir = SAMPLES_DIR / "cord"
    cord_gt_dir = GT_DIR / "cord"
    cord_img_dir.mkdir(parents=True, exist_ok=True)
    cord_gt_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(
        "naver-clova-ix/cord-v2",
        split=f"test[:{n_samples}]",
        trust_remote_code=True,
    )

    for i, sample in enumerate(tqdm(dataset, desc="CORD")):
        # Save image
        img_path = cord_img_dir / f"cord_{i:04d}.png"
        sample["image"].save(img_path)

        # Parse and save ground truth
        try:
            gt_raw = sample["ground_truth"]
            gt_parsed = json.loads(gt_raw) if isinstance(gt_raw, str) else gt_raw
        except (json.JSONDecodeError, TypeError):
            gt_parsed = {"raw": str(sample.get("ground_truth", ""))}

        gt_record = {
            "image_path": str(img_path),
            "dataset": "cord-v2",
            "sample_index": i,
            "ground_truth": gt_parsed,
        }
        with open(cord_gt_dir / f"cord_{i:04d}.json", "w") as f:
            json.dump(gt_record, f, indent=2)

    print(f"✅ CORD: {n_samples} samples → {cord_img_dir}")


def download_sroie(n_samples: int = 50) -> None:
    """Download SROIE receipt dataset (ICDAR 2019).

    Fields available: company, date, address, total
    """
    from datasets import load_dataset

    print(f"\n📥 Downloading SROIE ({n_samples} samples)...")

    sroie_img_dir = SAMPLES_DIR / "sroie"
    sroie_gt_dir = GT_DIR / "sroie"
    sroie_img_dir.mkdir(parents=True, exist_ok=True)
    sroie_gt_dir.mkdir(parents=True, exist_ok=True)

    try:
        dataset = load_dataset(
            "darentang/sroie",
            split=f"test[:{n_samples}]",
            trust_remote_code=True,
        )

        for i, sample in enumerate(tqdm(dataset, desc="SROIE")):
            img_path = sroie_img_dir / f"sroie_{i:04d}.png"

            img = sample.get("image")
            if img and hasattr(img, "save"):
                img.save(img_path)

            gt_record = {
                "image_path": str(img_path),
                "dataset": "sroie",
                "sample_index": i,
                "ground_truth": {
                    "vendor_name": sample.get("company", ""),
                    "invoice_date": sample.get("date", ""),
                    "vendor_address": sample.get("address", ""),
                    "total_amount": sample.get("total", ""),
                },
            }
            with open(sroie_gt_dir / f"sroie_{i:04d}.json", "w") as f:
                json.dump(gt_record, f, indent=2)

        print(f"✅ SROIE: {n_samples} samples → {sroie_img_dir}")

    except Exception as exc:
        print(f"⚠️  SROIE download failed: {exc}")
        print("    Continuing with CORD only — SROIE is optional for Day 1 eval.")


if __name__ == "__main__":
    for d in [DATA_DIR, SAMPLES_DIR, GT_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    download_cord(n_samples=100)
    download_sroie(n_samples=50)

    print("\n🎉 Datasets ready.")
    print("   Next: python scripts/run_eval.py")
