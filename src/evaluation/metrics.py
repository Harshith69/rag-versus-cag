from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.config import GROQ_API_KEY, GROQ_MODEL


JUDGE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", (
        "You are an impartial judge evaluating the quality of an AI answer.\n"
        "Score the answer on each dimension from 1 (worst) to 5 (best).\n\n"
        "Dimensions:\n"
        "- relevance: Does the answer address the question?\n"
        "- completeness: Does it cover all aspects of the question?\n"
        "- accuracy: Is the information factually correct based on the source?\n"
        "- conciseness: Is it appropriately concise without losing information?\n\n"
        "Respond ONLY in this exact format (no extra text):\n"
        "relevance: <score>\n"
        "completeness: <score>\n"
        "accuracy: <score>\n"
        "conciseness: <score>"
    )),
    ("human", (
        "Question: {question}\n\n"
        "Answer: {answer}\n\n"
        "Source context (ground truth): {context}"
    )),
])


def llm_judge(question: str, answer: str, context: str) -> dict[str, int]:
    """Use an LLM to score an answer on four quality dimensions."""
    llm = ChatGroq(api_key=GROQ_API_KEY, model=GROQ_MODEL, temperature=0)
    chain = JUDGE_PROMPT | llm | StrOutputParser()

    raw = chain.invoke({
        "question": question,
        "answer": answer,
        "context": context[:3000],
    })

    scores = {}
    for line in raw.strip().splitlines():
        line = line.strip()
        if ":" in line:
            key, value = line.split(":", 1)
            key = key.strip().lower()
            try:
                scores[key] = int(value.strip())
            except ValueError:
                pass

    for dim in ("relevance", "completeness", "accuracy", "conciseness"):
        scores.setdefault(dim, 0)

    return scores
