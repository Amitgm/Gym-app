from langchain_community.document_loaders import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import chroma
from langchain_community.llms import openai
from dotenv import load_dotenv
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_models import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
import os

import pinecone

# pinecone.init(api_key="40945abd-da2c-4bf4-909b-25f1dc2c95a1",environment="gcp-starter")

from dotenv import load_dotenv

load_dotenv()


os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

loader = CSVLoader(file_path="megaGymDataset.csv",encoding="utf-8",csv_args={"delimiter":','})

data = loader.load()


def split_text(data):

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)

    chunks = text_splitter.split_documents(data)

    return chunks

def generate_vector_store(text_chunks):

    # embedding = OpenAIEmbeddings()
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

    PineconeVectorStore.from_documents(
                documents=text_chunks,
                index_name="pinconedb",
                embedding=OpenAIEmbeddings(api_key=os.environ["OPENAI_API_KEY"]), 
                namespace="wondervector5000" 
            )
    



text_chunks = split_text(data)

vectors = generate_vector_store(text_chunks)