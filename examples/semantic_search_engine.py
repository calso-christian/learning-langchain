from langchain_core import documents
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv

try:
    load_dotenv()
except ImportError:
    pass


file_path = r"docs\data_science_handbook.pdf"

loader = PyPDFLoader(file_path)

docs=loader.load()

print(len(docs))

