import pandas as pd


import os

from dotenv import load_dotenv


import os
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
import pinecone




# pc = Pinecone(
#         api_key=os.environ.get("PINECONE_API_KEY")
#     )


load_dotenv()

os.environ["PINECONE_API_KEY"] = os.getenv("PINECONE_API_KEY")

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

vectors = OpenAIEmbeddings(api_key=os.environ["OPENAI_API_KEY"] ).embed_query("how are you?")


# pinecone.init(api_key=os.environ["PINECONE_API_KEY"],environment="gcp-starter")

# index_name = "lanch2"



print(len(vectors))


pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

pc.create_index(
    name="pinconedb",
    dimension=1536, # Replace with your model dimensions
    metric="cosine", # Replace with your model metric
    spec=ServerlessSpec(
        cloud="aws",
        region="us-east-1"
    ) 
)

# docsearch = PineconeVectorStore.from_documents(
#     documents=md_header_splits,
#     index_name="docs-rag-chatbot",
#     embedding=OpenAIEmbeddings(api_key=os.environ["OPENAI_API_KEY"]), 
#     namespace="wondervector5000" 
# )

# print(pc)




# from langchain_community.utilities import SQLDatabase
# from sqlalchemy import create_engine
# from langchain_community.agent_toolkits import create_sql_agent
# from langchain_openai import ChatOpenAI
# import os

# from dotenv import load_dotenv
# from langchain.agents.agent_types import AgentType
# load_dotenv()
# os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")


# # print(len(pd.read_csv("megaGymDataset.csv")))

# # df = pd.read_csv("megaGymDataset.csv")


# engine = create_engine("sqlite:///megaGymDataset.db")

# # df.to_sql("megaGymDataset", engine, index=False)

# llm = ChatOpenAI(model="gpt-3.5-turbo",temperature=0)



# db = SQLDatabase(engine=engine)

# agent_executor = create_sql_agent(llm=llm,
#                                   db=db,
#                                   agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
#                                   handle_parsing_errors=True, 
#                                   verbose=True)


# print(agent_executor.invoke({"input": "give me an intermediate exercise for abdominal?"}))


