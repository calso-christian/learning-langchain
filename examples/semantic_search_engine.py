from langchain_core import documents
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

try:
    load_dotenv()
except ImportError:
    pass


file_path = r"docs\data_science_handbook.pdf"

# --------------------------------------------------------
# DOCUMENT LOADING
# --------------------------------------------------------

loader = PyPDFLoader(file_path)

docs=loader.load()

print(f"Total Document Objects: {len(docs)}\n\n")

print(f"Page Content:\n{docs[100].page_content[:400]}\n")
print(f"Metadata:\n{docs[100].metadata}\n\n")

# --------------------------------------------------------
# SPLITTING
# --------------------------------------------------------

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    add_start_index=True
)

all_splits=text_splitter.split_documents(docs)

print(f"Total Splits: {len(all_splits)}")

# --------------------------------------------------------
# SPLITTING
# --------------------------------------------------------

embeddings = OpenAIEmbeddings(model='text-embedding-3-large')

vector_1 = embeddings.embed_query(all_splits[0].page_content)
print(f"Generated vectors of length {len(vector_1)}\n")
print(vector_1[:10])
