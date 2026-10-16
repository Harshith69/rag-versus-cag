import time
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from src.config import (
    GROQ_API_KEY,
    GROQ_MODEL,
    EMBEDDING_MODEL,
    CHROMA_PERSIST_DIR,
    CHROMA_COLLECTION_NAME,
    RETRIEVAL_TOP_K,
)


RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", (
        "You are a helpful assistant. Answer the question based ONLY on the "
        "provided context. If the context does not contain enough information, "
        "say so clearly.\n\nContext:\n{context}"
    )),
    ("human", "{question}"),
])


class RAGPipeline:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        self.llm = ChatGroq(
            api_key=GROQ_API_KEY,
            model=GROQ_MODEL,
            temperature=0,
        )
        self.vectorstore: Chroma | None = None
        self.chain = None

    def ingest(self, chunks: list[Document]) -> int:
        """Create vector store from document chunks."""
        self.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=CHROMA_PERSIST_DIR,
            collection_name=CHROMA_COLLECTION_NAME,
        )
        return len(chunks)

    def load_vectorstore(self):
        """Load an existing vector store from disk."""
        self.vectorstore = Chroma(
            persist_directory=CHROMA_PERSIST_DIR,
            embedding_function=self.embeddings,
            collection_name=CHROMA_COLLECTION_NAME,
        )

    def _build_chain(self):
        if self.vectorstore is None:
            raise RuntimeError("Vector store not initialized. Run ingest() or load_vectorstore() first.")

        retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": RETRIEVAL_TOP_K},
        )

        def format_docs(docs: list[Document]) -> str:
            return "\n\n---\n\n".join(doc.page_content for doc in docs)

        self.chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | RAG_PROMPT
            | self.llm
            | StrOutputParser()
        )

    def query(self, question: str) -> dict:
        """Run a query through the RAG pipeline and return answer + metrics."""
        if self.chain is None:
            self._build_chain()

        retrieved_docs = self.vectorstore.as_retriever(
            search_kwargs={"k": RETRIEVAL_TOP_K}
        ).invoke(question)

        start = time.perf_counter()
        answer = self.chain.invoke(question)
        latency = time.perf_counter() - start

        sources = [
            {
                "content": doc.page_content[:200],
                "metadata": doc.metadata,
            }
            for doc in retrieved_docs
        ]

        return {
            "answer": answer,
            "latency_seconds": round(latency, 3),
            "num_chunks_retrieved": len(retrieved_docs),
            "sources": sources,
            "pipeline": "RAG",
        }
