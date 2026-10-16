import json
from pathlib import Path
from rich.console import Console
from rich.table import Table
from src.rag.pipeline import RAGPipeline
from src.cag.pipeline import CAGPipeline
from src.evaluation.metrics import llm_judge
from src.config import RESULTS_DIR

console = Console()


def run_comparison(
    rag: RAGPipeline,
    cag: CAGPipeline,
    questions: list[str],
    ground_truth_context: str,
    evaluate_quality: bool = True,
) -> list[dict]:
    """Run the same questions through both pipelines and collect metrics."""
    results = []

    for i, question in enumerate(questions, 1):
        console.print(f"\n[bold cyan]Question {i}/{len(questions)}:[/] {question}")

        rag_result = rag.query(question)
        console.print(f"  [green]RAG[/] answered in {rag_result['latency_seconds']}s")

        cag_result = cag.query(question)
        console.print(f"  [yellow]CAG[/] answered in {cag_result['latency_seconds']}s")

        entry = {
            "question": question,
            "rag_answer": rag_result["answer"],
            "cag_answer": cag_result["answer"],
            "rag_latency": rag_result["latency_seconds"],
            "cag_latency": cag_result["latency_seconds"],
            "rag_chunks_retrieved": rag_result["num_chunks_retrieved"],
            "cag_context_tokens": cag_result["approx_context_tokens"],
        }

        if evaluate_quality:
            console.print("  [dim]Running LLM judge...[/]")
            rag_scores = llm_judge(question, rag_result["answer"], ground_truth_context)
            cag_scores = llm_judge(question, cag_result["answer"], ground_truth_context)
            entry["rag_scores"] = rag_scores
            entry["cag_scores"] = cag_scores

        results.append(entry)

    return results


def print_results(results: list[dict]):
    """Print a comparison table to the console."""
    table = Table(title="RAG vs CAG Comparison Results", show_lines=True)
    table.add_column("Question", style="cyan", max_width=40)
    table.add_column("RAG Latency", justify="right")
    table.add_column("CAG Latency", justify="right")
    table.add_column("RAG Quality", justify="center")
    table.add_column("CAG Quality", justify="center")
    table.add_column("Winner", justify="center")

    for r in results:
        rag_avg = cag_avg = 0
        if "rag_scores" in r and "cag_scores" in r:
            rag_avg = sum(r["rag_scores"].values()) / max(len(r["rag_scores"]), 1)
            cag_avg = sum(r["cag_scores"].values()) / max(len(r["cag_scores"]), 1)

        if rag_avg > cag_avg:
            winner = "[green]RAG[/]"
        elif cag_avg > rag_avg:
            winner = "[yellow]CAG[/]"
        else:
            winner = "Tie"

        table.add_row(
            r["question"][:40],
            f"{r['rag_latency']}s",
            f"{r['cag_latency']}s",
            f"{rag_avg:.1f}/5",
            f"{cag_avg:.1f}/5",
            winner,
        )

    console.print(table)

    # Aggregate summary
    avg_rag_latency = sum(r["rag_latency"] for r in results) / len(results)
    avg_cag_latency = sum(r["cag_latency"] for r in results) / len(results)

    console.print(f"\n[bold]Average Latency:[/]  RAG={avg_rag_latency:.3f}s  |  CAG={avg_cag_latency:.3f}s")

    if any("rag_scores" in r for r in results):
        scored = [r for r in results if "rag_scores" in r]
        avg_rag_q = sum(
            sum(r["rag_scores"].values()) / max(len(r["rag_scores"]), 1)
            for r in scored
        ) / len(scored)
        avg_cag_q = sum(
            sum(r["cag_scores"].values()) / max(len(r["cag_scores"]), 1)
            for r in scored
        ) / len(scored)
        console.print(f"[bold]Average Quality:[/]  RAG={avg_rag_q:.2f}/5  |  CAG={avg_cag_q:.2f}/5")


def save_results(results: list[dict], filename: str = "comparison_results.json"):
    """Save results to a JSON file."""
    out_dir = Path(RESULTS_DIR)
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / filename
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    console.print(f"\n[bold green]Results saved to {out_path}[/]")
