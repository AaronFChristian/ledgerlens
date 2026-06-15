"""Run the full LedgerLens eval suite.

    python scripts/run_evals.py
    python scripts/run_evals.py --extraction-only   # skip agent evals
"""

import argparse
import json
import sys
from pathlib import Path

from rich.console import Console
from rich.table import Table
from rich import box

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ledgerlens.evals.harness import run_full_eval, evaluate_extraction_accuracy
from ledgerlens.utils.logging import setup_logging

console = Console()

CI_THRESHOLDS = {
    "total_amount_accuracy":   70.0,
    "avg_groundedness":        0.70,
    "avg_context_relevance":   0.70,
}


def main(extraction_only: bool = False, n_extraction: int = 20, n_agent: int = 3) -> bool:
    setup_logging(level="WARNING")
    console.print("\n[bold]LedgerLens — Full Eval Suite[/bold]\n")

    if extraction_only:
        metrics = {"extraction": evaluate_extraction_accuracy(n_samples=n_extraction)}
        metrics["avg_groundedness"] = None
        metrics["avg_context_relevance"] = None
    else:
        metrics = run_full_eval(
            n_extraction_samples=n_extraction,
            n_agent_questions=n_agent,
        )

    ext = metrics["extraction"]

    # Extraction table
    ext_table = Table(title="Extraction accuracy", box=box.SIMPLE_HEAD)
    ext_table.add_column("Metric", style="bold")
    ext_table.add_column("Correct", justify="right")
    ext_table.add_column("Evaluated", justify="right")
    ext_table.add_column("Accuracy", justify="right", style="bold")

    ext_table.add_row(
        "Total amount",
        str(ext["total_amount"]["correct"]),
        str(ext["total_amount"]["total"]),
        f"{ext['total_amount']['accuracy']}%",
    )
    ext_table.add_row(
        "Line item count",
        str(ext["line_item_count"]["correct"]),
        str(ext["line_item_count"]["total"]),
        f"{ext['line_item_count']['accuracy']}%",
    )
    console.print(ext_table)

    console.print(f"  Auto-approval rate : {ext['auto_approval_rate']}%")
    console.print(f"  Cost per document  : ${ext['cost_per_doc']}")
    console.print(f"  Samples evaluated  : {ext['samples_evaluated']}")

    # Agent evals
    if not extraction_only and metrics.get("agent"):
        agent_table = Table(title="\nAgent eval (groundedness + context relevance)", box=box.SIMPLE_HEAD)
        agent_table.add_column("Question", max_width=50)
        agent_table.add_column("Groundedness", justify="right")
        agent_table.add_column("Relevance", justify="right")

        for m in metrics["agent"]:
            agent_table.add_row(
                m["question"][:50],
                f"{m['groundedness']:.2f}",
                f"{m['context_relevance']:.2f}",
            )

        console.print(agent_table)
        console.print(f"  Avg groundedness       : {metrics['avg_groundedness']:.2f}")
        console.print(f"  Avg context relevance  : {metrics['avg_context_relevance']:.2f}")

    # CI gate
    console.print("\n[bold]CI gate[/bold]")
    passed = True

    checks = [
        ("Total amount accuracy ≥ 70%", ext["total_amount"]["accuracy"] >= CI_THRESHOLDS["total_amount_accuracy"]),
    ]
    if not extraction_only:
        checks += [
            ("Avg groundedness ≥ 0.70",     (metrics.get("avg_groundedness") or 0) >= CI_THRESHOLDS["avg_groundedness"]),
            ("Avg context relevance ≥ 0.70", (metrics.get("avg_context_relevance") or 0) >= CI_THRESHOLDS["avg_context_relevance"]),
        ]

    for name, ok in checks:
        icon = "[green]✅[/green]" if ok else "[red]❌[/red]"
        console.print(f"  {icon} {name}")
        if not ok:
            passed = False

    if passed:
        console.print("\n[green bold]All checks passed — CI green[/green bold]")
    else:
        console.print("\n[red bold]CI failed — review metrics above[/red bold]")

    # Save results
    out = Path("data/eval_results.json")
    out.write_text(json.dumps(metrics, indent=2, default=str))
    console.print(f"\n[dim]Results saved to {out}[/dim]")

    return passed


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--extraction-only", action="store_true")
    parser.add_argument("--n-extraction", type=int, default=20)
    parser.add_argument("--n-agent", type=int, default=3)
    args = parser.parse_args()

    ok = main(
        extraction_only=args.extraction_only,
        n_extraction=args.n_extraction,
        n_agent=args.n_agent,
    )
    sys.exit(0 if ok else 1)
