"""Extract CORD images and save ExtractionResult JSON files.

Run this before load_graph.py:
    python scripts/extract_and_save.py --n 50
"""

import argparse
import sys
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ledgerlens.extraction.pipeline import ExtractionPipeline
from ledgerlens.utils.logging import setup_logging


def main(n_samples: int = 50) -> None:
    setup_logging(level="WARNING")

    cord_dir   = Path("data/samples/cord")
    output_dir = Path("data/extraction_results")
    output_dir.mkdir(parents=True, exist_ok=True)

    images = sorted(cord_dir.glob("*.png"))[:n_samples]
    if not images:
        print("No CORD images found. Run: python scripts/download_datasets.py first")
        return

    print(f"\nExtracting {len(images)} CORD images → {output_dir}")
    print("(Each image costs ~$0.009 — 50 images ≈ $0.45)\n")

    pipeline = ExtractionPipeline()
    saved = 0
    total_cost = 0.0

    for img in tqdm(images, desc="Extracting"):
        try:
            result  = pipeline.extract(img)
            out     = output_dir / f"{img.stem}.json"
            out.write_text(result.model_dump_json(indent=2))
            saved      += 1
            total_cost += result.cost_usd
        except Exception as exc:
            print(f"\nError on {img.name}: {exc}")

    print(f"\nDone: {saved}/{len(images)} saved | Total cost: ${total_cost:.4f}")
    print("Next: python scripts/load_graph.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract CORD images and save JSON results")
    parser.add_argument("--n", type=int, default=50, help="Number of images to extract")
    args = parser.parse_args()
    main(args.n)
