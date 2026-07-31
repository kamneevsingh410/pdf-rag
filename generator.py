import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

from retriever import retrieve_relevant_chunks

load_dotenv()

PROMPT_TEMPLATE = """You are a helpful study assistant answering questions
based ONLY on the provided context from the uploaded document.

Rules:
- Answer using only the information in the context below.
- If the context doesn't contain enough information to answer,
  say "I don't have enough information in the document to answer that."
- Keep answers clear, accurate, and concise.
{history_section}
Context:
{context}

Question: {question}

Answer:"""

CONDENSE_PROMPT_TEMPLATE = """Given the conversation below and a follow-up question,
rewrite the follow-up question as a single standalone question that can be
understood without the conversation. Keep it short. If the question is already
standalone, return it unchanged. Return ONLY the rewritten question.

Conversation:
{history}

Follow-up question: {question}

Standalone question:"""

# max number of past turns to carry into the prompt
MAX_HISTORY_TURNS = 4


def format_history(history):
    # history is a list of (question, answer) tuples
    lines = []
    for past_question, past_answer in history[-MAX_HISTORY_TURNS:]:
        lines.append(f"Student: {past_question}")
        lines.append(f"Assistant: {past_answer}")
    return "\n".join(lines)


def condense_question(question: str, history):
    """rewrite a follow-up question into a standalone one so vector retrieval
    has something meaningful to match. no-op when there is no history."""
    if not history:
        return question

    prompt = CONDENSE_PROMPT_TEMPLATE.format(
        history=format_history(history),
        question=question
    )
    llm = get_llm()
    response = llm.invoke(prompt)
    standalone = response.content.strip()

    # fall back to the raw question if the model returns something odd
    return standalone if standalone else question


def get_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-3.5-flash-lite",
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0.2  # low temperature keeps answers factual and consistent
    )




def generate_answer(question: str, k: int = 4, history=None, collection_name: str = None):

    history = history or []

    # step 1: rewrite follow-ups into standalone questions for better retrieval
    standalone_question = condense_question(question, history)

    # step 2: retrieve relevant chunks using the standalone question
    chunks = retrieve_relevant_chunks(standalone_question, k=k, collection_name=collection_name)


    # step 3: combine chunk text into one context block
    context = "\n\n---\n\n".join([doc.page_content for doc in chunks])

    # step 4: build the final prompt with context and recent conversation
    if history:
        history_section = (
            "\nConversation so far (for reference only — "
            "facts must still come from the context):\n"
            f"{format_history(history)}\n"
        )
    else:
        history_section = ""

    prompt = PROMPT_TEMPLATE.format(
        history_section=history_section,
        context=context,
        question=question
    )

    # step 5: call gemini and return structured result
    llm = get_llm()
    response = llm.invoke(prompt)

    return {
        "answer": response.content,
        "sources": chunks,
        "standalone_question": standalone_question
    }


if __name__ == "__main__":
    test_question = "What is the difference between bagging and boosting?"

    result = generate_answer(test_question)

    print(f"question: {test_question}\n")
    print(f"answer:\n{result['answer']}\n")

    print("--- sources used ---")
    for i, doc in enumerate(result["sources"]):
        print(f"source {i+1} (page {doc.metadata.get('page_label')}): {doc.page_content[:100]}...")