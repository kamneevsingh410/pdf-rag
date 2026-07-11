import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma

from loader import load_and_chunk_pdf

load_dotenv()

PERSIST_DIRECTORY = "chroma_db"
COLLECTION_NAME = "pdf_study_assistant"


def get_embedding_model():
    return GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )


def build_vector_store(chunks, persist_directory=PERSIST_DIRECTORY):
    embedding_model = get_embedding_model()

    print(f"embedding {len(chunks)} chunks... (calls the gemini api)")
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        collection_name=COLLECTION_NAME,
        persist_directory=persist_directory
    )

    print(f"vector store saved to '{persist_directory}/'")
    return vector_store


def load_existing_vector_store(persist_directory=PERSIST_DIRECTORY, collection_name=None):
    embedding_model = get_embedding_model()
    name = collection_name if collection_name is not None else COLLECTION_NAME

    vector_store = Chroma(
        collection_name=name,
        embedding_function=embedding_model,
        persist_directory=persist_directory
    )

    return vector_store



if __name__ == "__main__":
    pdf_path = "convert_file.pdf"
    chunks = load_and_chunk_pdf(pdf_path)
    vector_store = build_vector_store(chunks)

    # quick check: run a similarity search to verify the store is working
    results = vector_store.similarity_search("What is bias-variance tradeoff?", k=2)

    print("\n--- top 2 similar chunks ---")
    for i, doc in enumerate(results):
        print(f"\nresult {i+1}:")
        print(doc.page_content[:200], "...")