"""Field-level accuracy evaluation against CORD / SROIE ground truth.

Run after downloading datasets:
    python scripts/run_eval.py
    python scripts/run_eval.py --dataset sroie --n 30

Outputs:
  - Per-field accuracy table
  - Auto-approval rate
  - Cost-per-document vs $3.50 manual baseline
"""

import argparse
import json
import re
import sys
from pathlib import Path

from rich.console import Console
from rich.table import Table
from rich import box

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ledgerlens.extraction.pipeline import ExtractionPipeline
from ledgerlens.utils.logging import setup_logging

console = Console()


# ── Ground truth parsers ──────────────────────────────────────────────────────

def parse_cord_gt(gt: dict) -> dict:
    result = {}
    try:
        gt_parse = gt.get("gt_parse", {})
        total_section = gt_parse.get("total", {})
        if "total_price" in total_section:
            result["total_amount"] = str(total_section["total_price"])
        menus = gt_parse.get("menu", [])
        result["line_item_count"] = len(menus)
    except Exception:
        pass
    return result


def parse_sroie_gt(gt: dict) -> dict:
    """SROIE ground truth is already flat."""
    return {
        "vendor_name": gt.get("vendor_name", ""),
        "invoice_date": gt.get("invoice_date", ""),
        "total_amount": gt.get("total_amount", ""),
    }


# ── Normalisation ─────────────────────────────────────────────────────────────

def _normalize(value: str) -> str:
    s = str(value).strip().lower()
    s = re.sub(r"[$,\s₩¥€£]", "", s)
    try:
        f = float(s)
        s = str(int(f)) if f == int(f) else f"{f:.2f}"
    except (ValueError, OverflowError):
        pass
    return s


def fields_match(predicted, ground_truth) -> bool | None:
    """Compare predicted vs ground truth. Returns None if GT is empty."""
    if ground_truth is None or str(ground_truth).strip() == "":
        return None  # No ground truth to compare
    if predicted is None:
        return False
    return _normalize(str(predicted)) == _normalize(str(ground_truth))


# ── Evaluation loop ───────────────────────────────────────────────────────────

def run_eval(dataset: str = "cord", n_samples: int = 20) -> None:
    setup_logging(level="WARNING")  # Suppress per-file logs during batch eval

    gt_dir = Path(f"data/ground_truth/{dataset}")
    if not gt_dir.exists():
        console.print(f"[red]Ground truth not found at {gt_dir}[/red]")
        console.print("[yellow]Run: python scripts/download_datasets.py first[/yellow]")
        return

    gt_files = sorted(gt_dir.glob("*.json"))[:n_samples]
    if not gt_files:
        console.print(f"[red]No .json files in {gt_dir}[/red]")
        return

    pipeline = ExtractionPipeline()
    parse_gt = parse_cord_gt if dataset == "cord" else parse_sroie_gt

    # Metric accumulators
    counters: dict[str, dict[str, int]] = {
        "total_amount": {"correct": 0, "total": 0},
        "vendor_name": {"correct": 0, "total": 0},
        "invoice_date": {"correct": 0, "total": 0},
        "line_items": {"correct": 0, "total": 0},
    }
    review_count = 0
    processed = 0
    total_cost = 0.0
    total_ms = 0.0

    console.print(f"\n[bold]LedgerLens Day 1 — Evaluating {len(gt_files)} {dataset.upper()} samples[/bold]\n")

    with console.status("[dim]Extracting...[/dim]"):
        for gt_file in gt_files:
            with open(gt_file) as f:
                record = json.load(f)

            img_path = Path(record["image_path"])
            if not img_path.exists():
                console.print(f"[dim]Skip (missing): {img_path.name}[/dim]")
                continue

            try:
                result = pipeline.extract(img_path)
                gt = parse_gt(record["ground_truth"])
                processed += 1
                review_count += result.needs_human_review
                total_cost += result.cost_usd
                total_ms += result.extraction_time_ms

                inv = result.invoice

                # Total amount
                match = fields_match(inv.total_amount, gt.get("total_amount"))
                if match is not None:
                    counters["total_amount"]["total"] += 1
                    if match:
                        counters["total_amount"]["correct"] += 1

                # Vendor name
                match = fields_match(inv.vendor_name, gt.get("vendor_name"))
                if match is not None:
                    counters["vendor_name"]["total"] += 1
                    if match:
                        counters["vendor_name"]["correct"] += 1

                # Invoice date
                match = fields_match(inv.invoice_date, gt.get("invoice_date"))
                if match is not None:
                    counters["invoice_date"]["total"] += 1
                    if match:
                        counters["invoice_date"]["correct"] += 1

                # Line item count
                if "line_item_count" in gt:
                    counters["line_items"]["total"] += 1
                    if len(inv.line_items) == gt["line_item_count"]:
                        counters["line_items"]["correct"] += 1

            except Exception as exc:
                console.print(f"[red]Error {gt_file.stem}: {exc}[/red]")

    if processed == 0:
        console.print("[red]No samples processed.[/red]")
        return

    # ── Results table ────────────────────────────────────────────────────────

    table = Table(
        title=f"LedgerLens Extraction Accuracy — {dataset.upper()} ({processed} docs)",
        box=box.SIMPLE_HEAD,
        show_footer=False,
    )
    table.add_column("Metric", style="bold")
    table.add_column("Correct", justify="right")
    table.add_column("Evaluated", justify="right")
    table.add_column("Accuracy", justify="right", style="bold green")

    def pct(c: dict) -> str:
        if c["total"] == 0:
            return "n/a"
        return f"{c['correct'] / c['total'] * 100:.1f}%"

    table.add_row(
        "Total amount",
        str(counters["total_amount"]["correct"]),
        str(counters["total_amount"]["total"]),
        pct(counters["total_amount"]),
    )
    table.add_row(
        "Vendor name",
        str(counters["vendor_name"]["correct"]),
        str(counters["vendor_name"]["total"]),
        pct(counters["vendor_name"]),
    )
    table.add_row(
        "Invoice date",
        str(counters["invoice_date"]["correct"]),
        str(counters["invoice_date"]["total"]),
        pct(counters["invoice_date"]),
    )
    table.add_row(
        "Line item count",
        str(counters["line_items"]["correct"]),
        str(counters["line_items"]["total"]),
        pct(counters["line_items"]),
    )

    console.print(table)

    # ── Operational summary ──────────────────────────────────────────────────

    auto = processed - review_count
    console.print(f"[bold]Routing:[/bold]")
    console.print(f"  ✅ Auto-approved : {auto}/{processed} ({auto/processed*100:.0f}%)")
    console.print(f"  🔴 Human review  : {review_count}/{processed} ({review_count/processed*100:.0f}%)")

    console.print(f"\n[bold]Performance:[/bold]")
    console.print(f"  Avg extraction time : {total_ms/processed:.0f}ms")
    console.print(f"  Total API cost      : ${total_cost:.4f}")
    console.print(f"  Cost per document   : ${total_cost/processed:.5f}")

    monthly_10k = total_cost / processed * 10_000
    console.print(
        f"\n[dim]At scale: 10k invoices/mo ≈ ${monthly_10k:.2f} LLM cost "
        f"vs $35,000 manual (at $3.50/invoice) — "
        f"{(1 - monthly_10k / 35_000) * 100:.1f}% savings[/dim]"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LedgerLens extraction accuracy eval")
    parser.add_argument("--dataset", choices=["cord", "sroie"], default="cord")
    parser.add_argument("--n", type=int, default=20, help="Number of samples to evaluate")
    args = parser.parse_args()

    run_eval(dataset=args.dataset, n_samples=args.n)
