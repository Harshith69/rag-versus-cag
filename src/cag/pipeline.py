import time
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.config import GROQ_API_KEY, GROQ_MODEL


CAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", (
        "You are a helpful assistant. You have been given the ENTIRE knowledge "
        "base below. Answer the question using this knowledge. If the knowledge "
        "base does not contain enough information, say so clearly.\n\n"
        "=== FULL KNOWLEDGE BASE ===\n{knowledge_base}\n=== END KNOWLEDGE BASE ==="
    )),
    ("human", "{question}"),
])


class CAGPipeline:
    """Cache Augmented Generation — loads the full document context into the
    prompt instead of retrieving chunks.  Repeated queries reuse the same
    cached context, trading higher per-query token cost for zero retrieval
    latency and complete context availability."""

    def __init__(self):
        self.llm = ChatGroq(
            api_key=GROQ_API_KEY,
            model=GROQ_MODEL,
            temperature=0,
        )
        self.knowledge_base: str | None = None
        self.chain = None

    def load_context(self, full_text: str):
        """Cache the entire knowledge base in memory."""
        self.knowledge_base = full_text
        self.chain = (
            CAG_PROMPT
            | self.llm
            | StrOutputParser()
        )

    def query(self, question: str) -> dict:
        """Run a query with the full knowledge base as context."""
        if self.knowledge_base is None or self.chain is None:
            raise RuntimeError("Knowledge base not loaded. Run load_context() first.")

        start = time.perf_counter()
        answer = self.chain.invoke({
            "knowledge_base": self.knowledge_base,
            "question": question,
        })
        latency = time.perf_counter() - start

        context_length = len(self.knowledge_base)
        approx_tokens = context_length // 4  # rough estimate

        return {
            "answer": answer,
            "latency_seconds": round(latency, 3),
            "context_chars": context_length,
            "approx_context_tokens": approx_tokens,
            "pipeline": "CAG",
        }
