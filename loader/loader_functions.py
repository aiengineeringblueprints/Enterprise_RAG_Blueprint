import os

from dotenv import load_dotenv
from langchain_community.document_loaders import (
    PyPDFLoader,
    UnstructuredMarkdownLoader,
    UnstructuredWordDocumentLoader,
)
from langchain_core.document_loaders.base import BaseLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tag_management import get_tag_bitmask
from tqdm import tqdm
from vector_store import upload_documents_to_vectorstore

load_dotenv()
INDEX_NAME = os.environ.get("INDEX_NAME", "langchain-test-index")
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", 1000))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", 200))


class MultiEncodingLoader(BaseLoader):
    """
    A custom loader that attempts to open a file with multiple encodings.
    Works for .txt, .csv
    """

    def __init__(self, file_path: str, encodings=None, errors: str = "strict"):
        if encodings is None:
            encodings = ["utf-8", "latin-1", "cp1252"]
        self.file_path = file_path
        self.encodings = encodings
        self.errors = errors

    def load(self):
        return list(self.lazy_load())

    def lazy_load(self):
        for encoding in self.encodings:
            try:
                with open(
                    self.file_path, encoding=encoding, errors=self.errors
                ) as f:
                    text = f.read()
                yield Document(
                    page_content=text, metadata={"source": self.file_path}
                )
                return  # Success – file loaded, stop loop
            except UnicodeDecodeError:
                # If this encoding doesn't work, try the next one
                continue
        # If none of the encodings work, raise an error
        raise RuntimeError(
            f"Error loading {self.file_path} with encodings {self.encodings}"
        )


def upload_all_docs_to_vector_store(
    root_path: str = r".\test_documents",
    document_tag: str = None,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> None:
    """
    Uploads all documents from the specified root path to the local Chroma vector store.
    This function loads documents from the given directory, splits them into chunks using
    a recursive character text splitter, and uploads the resulting document splits to a
    local index.
    Is used to upload the documents chosen in the frontend
    Args:
        root_path (str, optional): The root directory path containing documents to upload.
            Defaults to r".\test_documents".
        document_tag (str, optional): An optional tag to filter or identify documents.
            Defaults to None.
    Returns:
        None
    Side Effects:
        Prints a message if no documents are found.
    Uploads document chunks to the local index specified in INDEX_NAME environment variable.
    """

    documents = load_docs_from_dict(root_path, document_tag=document_tag)
    if not documents:
        print("Keine Dokumente zum Hochladen gefunden.")
        return

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    all_splits = text_splitter.split_documents(documents)

    upload_documents_to_vectorstore(
        all_splits=all_splits, index_name=INDEX_NAME
    )


def load_docs_from_dict(
    root_path: str = r".\test_documents", document_tag: str = "General"
) -> list:
    """
    Loads documents from the specified root directory and its subdirectories using appropriate loaders based on file extensions.
    The loaded documents must still be splitted into chunks before being used in a vector store. Adds a tag to each document's metadata if provided.

    Args:
        root_path (str, optional): The root directory path to scan for documents. Defaults to r".\test_documents".
        document_tag (str, optional): A tag/category to be added to all documents as metadata. If no tag is provided, "General" is used.

    Returns:
        list: A list of loaded document objects from all supported files found in the directory tree.

    Supported file types and loaders:
        - .docx: UnstructuredWordDocumentLoader
        - .pdf: PyPDFLoader
        - .txt: MultiEncodingLoader
        - .md: UnstructuredMarkdownLoader
        - .csv: MultiEncodingLoader
    """

    loader_mapping = {
        ".docx": UnstructuredWordDocumentLoader,
        ".pdf": PyPDFLoader,
        ".txt": MultiEncodingLoader,
        ".md": UnstructuredMarkdownLoader,
        ".csv": MultiEncodingLoader,
    }

    documents = []
    all_files = []

    # Alle Dateien sammeln
    print("Scanne Dateisystem...")
    for dirpath, dirnames, filenames in os.walk(root_path):
        for file in filenames:
            file_path = os.path.join(dirpath, file)
            all_files.append(file_path)

    print(f"Verarbeite {len(all_files)} Dateien...\n")
    for file_path in tqdm(
        all_files, desc="Lade Dokumente", unit="Datei", disable=False
    ):
        ext = os.path.splitext(file_path)[1].lower()

        if ext in loader_mapping:
            try:
                loader = loader_mapping[ext](file_path)
                docs = loader.load()

                # Add document tag to metadata if provided
                if document_tag:
                    tag_bitmask = get_tag_bitmask(document_tag)
                    for doc in docs:
                        if doc.metadata is None:
                            doc.metadata = {}
                        doc.metadata["category"] = document_tag
                        doc.metadata["category_bitmask"] = tag_bitmask

                documents.extend(docs)
            except Exception as e:
                tqdm.write(f"[FEHLER] {file_path}: {str(e)}")

    print("\n Gesamtzahl geladener Dokumente:", len(documents))
    if document_tag:
        print(f" Kategorie zugewiesen: {document_tag}")

    return documents
