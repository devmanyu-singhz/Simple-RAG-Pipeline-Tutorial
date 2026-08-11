from pathlib import Path
from typing import List, Any
from langchain_community.document_loaders import PyPDFLoader

# Now we create a function to load the pdf files in the data directory

def load_all_docs(data_dir:str) -> List[Any]:
    """Load all PDF documents and convert to langchain document structure"""

    # use project root data folder
    data_path = Path(data_dir).resolve()
    print(f"[DEBUG] data path : {data_path}")
    documents = []

    # for pdf files
    pdf_files = list(data_path.glob('**/*.pdf'))
    print(f"[DEBUG] found {len(pdf_files)} pdf files : {[str(f) for f in pdf_files]}")
    for pdf_file in pdf_files:
        print(f"[DEBUG] loading pdf : {pdf_file}")

        try:
            loader = PyPDFLoader(str(pdf_file))
            loaded = loader.load()
            print(f"[DEBUG] loaded {len(loaded)} pdf docs from {pdf_file}")
            documents.extend(loaded)

        except Exception as e:
            print(f"error while loading the file : {e}")
            raise

    return documents

    
# similarly for text, csv, sql files as well