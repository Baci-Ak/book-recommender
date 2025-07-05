
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
#from langchain_chroma import Chroma
from langchain.vectorstores import Chroma


# Load environment variables
load_dotenv()

# Load the tagged descriptions
raw_documents = TextLoader("tagged_descriptions.txt").load()

#  Split into chunks
text_splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=0, separator="\n")
documents = text_splitter.split_documents(raw_documents)

# Create embeddings
embedding_model = OpenAIEmbeddings()

# Persist vector store to disk
db = Chroma.from_documents(
    documents,
    embedding=embedding_model,
    persist_directory="chroma_store"
)

# Save the DB
db.persist()
print("Vector store has been successfully built and saved to 'chroma_store/'")
