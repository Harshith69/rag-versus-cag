from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from src.config import CHUNK_SIZE, CHUNK_OVERLAP, DATA_DIR


def load_pdfs(directory: str = DATA_DIR) -> list[Document]:
    """Load all PDF files from a directory and return raw documents."""
    pdf_dir = Path(directory)
    if not pdf_dir.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")

    documents = []
    for pdf_path in sorted(pdf_dir.glob("*.pdf")):
        loader = PyPDFLoader(str(pdf_path))
        documents.extend(loader.load())

    if not documents:
        raise ValueError(f"No PDF files found in {directory}")

    return documents


def get_chunks(documents: list[Document]) -> list[Document]:
    """Split documents into chunks for RAG pipeline."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(documents)


def get_full_text(documents: list[Document]) -> str:
    """Concatenate all document text for CAG pipeline."""
    return "\n\n".join(doc.page_content for doc in documents)
