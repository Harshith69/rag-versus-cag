"""RAG vs CAG Comparison Tool

Usage:
    python main.py ingest              Ingest PDFs into the RAG vector store
    python main.py query-rag "..."     Query using the RAG pipeline
    python main.py query-cag "..."     Query using the CAG pipeline
    python main.py compare             Run comparison on sample questions
    python main.py interactive         Interactive mode — ask questions to both pipelines
"""

import sys
from rich.console import Console
from rich.panel import Panel
from src.config import DATA_DIR
from src.document_loader import load_pdfs, get_chunks, get_full_text
from src.rag.pipeline import RAGPipeline
from src.cag.pipeline import CAGPipeline
from src.evaluation.comparator import run_comparison, print_results, save_results

console = Console()

SAMPLE_QUESTIONS = [
    "What are the main applications of AI in banking and finance?",
    "How is machine learning used for fraud detection in banks?",
    "What are the risks and challenges of implementing AI in financial services?",
    "How does AI improve customer experience in banking?",
    "What regulatory considerations exist for AI adoption in finance?",
]


def cmd_ingest():
    console.print("[bold]Loading PDFs...[/]")
    documents = load_pdfs(DATA_DIR)
    console.print(f"Loaded {len(documents)} pages from PDFs")

    chunks = get_chunks(documents)
    console.print(f"Split into {len(chunks)} chunks")

    rag = RAGPipeline()
    count = rag.ingest(chunks)
    console.print(f"[bold green]Ingested {count} chunks into ChromaDB[/]")


def cmd_query_rag(question: str):
    rag = RAGPipeline()
    rag.load_vectorstore()
    result = rag.query(question)

    console.print(Panel(result["answer"], title="[green]RAG Answer[/]", border_style="green"))
    console.print(f"Latency: {result['latency_seconds']}s | Chunks retrieved: {result['num_chunks_retrieved']}")

    console.print("\n[dim]Sources:[/]")
    for i, src in enumerate(result["sources"], 1):
        console.print(f"  {i}. {src['content'][:120]}...")


def cmd_query_cag(question: str):
    console.print("[dim]Loading full document context...[/]")
    documents = load_pdfs(DATA_DIR)
    full_text = get_full_text(documents)

    cag = CAGPipeline()
    cag.load_context(full_text)
    result = cag.query(question)

    console.print(Panel(result["answer"], title="[yellow]CAG Answer[/]", border_style="yellow"))
    console.print(
        f"Latency: {result['latency_seconds']}s | "
        f"Context: ~{result['approx_context_tokens']} tokens"
    )


def cmd_compare():
    console.print("[bold]Setting up pipelines...[/]\n")

    documents = load_pdfs(DATA_DIR)
    chunks = get_chunks(documents)
    full_text = get_full_text(documents)

    rag = RAGPipeline()
    rag.ingest(chunks)
    console.print(f"[green]RAG:[/] Ingested {len(chunks)} chunks")

    cag = CAGPipeline()
    cag.load_context(full_text)
    console.print(f"[yellow]CAG:[/] Loaded {len(full_text)} chars of context")

    console.print("\n[bold]Running comparison...[/]")
    results = run_comparison(rag, cag, SAMPLE_QUESTIONS, full_text)

    console.print("\n")
    print_results(results)
    save_results(results)


def cmd_interactive():
    console.print("[bold]Setting up both pipelines...[/]\n")

    documents = load_pdfs(DATA_DIR)
    chunks = get_chunks(documents)
    full_text = get_full_text(documents)

    rag = RAGPipeline()
    rag.ingest(chunks)

    cag = CAGPipeline()
    cag.load_context(full_text)

    console.print("[bold green]Ready! Type your questions (Ctrl+C to exit)[/]\n")

    while True:
        try:
            question = input("Question: ").strip()
            if not question:
                continue

            rag_result = rag.query(question)
            cag_result = cag.query(question)

            console.print(Panel(rag_result["answer"], title="[green]RAG Answer[/]", border_style="green"))
            console.print(f"  Latency: {rag_result['latency_seconds']}s\n")

            console.print(Panel(cag_result["answer"], title="[yellow]CAG Answer[/]", border_style="yellow"))
            console.print(f"  Latency: {cag_result['latency_seconds']}s\n")

        except (KeyboardInterrupt, EOFError):
            console.print("\n[bold]Goodbye![/]")
            break


def main():
    if len(sys.argv) < 2:
        console.print(__doc__)
        return

    command = sys.argv[1].lower()

    if command == "ingest":
        cmd_ingest()
    elif command == "query-rag":
        if len(sys.argv) < 3:
            console.print("[red]Usage: python main.py query-rag \"your question\"[/]")
            return
        cmd_query_rag(sys.argv[2])
    elif command == "query-cag":
        if len(sys.argv) < 3:
            console.print("[red]Usage: python main.py query-cag \"your question\"[/]")
            return
        cmd_query_cag(sys.argv[2])
    elif command == "compare":
        cmd_compare()
    elif command == "interactive":
        cmd_interactive()
    else:
        console.print(f"[red]Unknown command: {command}[/]")
        console.print(__doc__)


if __name__ == "__main__":
    main()
