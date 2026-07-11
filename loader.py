from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_and_chunk_pdf(pdf_path: str, chunk_size: int = 800, chunk_overlap: int = 150):
    # load pdf page by page
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    print(f"loaded pdf with {len(pages)} pages.")

    # split into smaller overlapping chunks so questions don't get cut off from their answers
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    chunks = splitter.split_documents(pages)
    print(f"split into {len(chunks)} chunks.")

    return chunks


if __name__ == "__main__":
    test_path = "convert_file.pdf"
    chunks = load_and_chunk_pdf(test_path)

    print("\n--- sample chunk 0 ---")
    print(chunks[0].page_content)
    print("\nmetadata:", chunks[0].metadata)