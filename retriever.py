from embedder import load_existing_vector_store


def retrieve_relevant_chunks(query: str, k: int = 4, collection_name: str = None):
    vector_store = load_existing_vector_store(collection_name=collection_name)
    results = vector_store.similarity_search(query, k=k)
    return results


def retrieve_with_scores(query: str, k: int = 4, collection_name: str = None):
    vector_store = load_existing_vector_store(collection_name=collection_name)
    results = vector_store.similarity_search_with_score(query, k=k)
    return results



if __name__ == "__main__":
    test_query = "What is the difference between bagging and boosting?"

    print(f"query: {test_query}\n")

    results = retrieve_with_scores(test_query, k=3)

    for i, (doc, score) in enumerate(results):
        print(f"--- result {i+1} (distance score: {score:.4f}) ---")
        print(doc.page_content[:250], "...")
        print(f"source page: {doc.metadata.get('page_label')}\n")