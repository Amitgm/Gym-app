from langchain_community.document_loaders import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import chroma
from langchain_community.llms import openai
from langchain.chains import LLMChain
from dotenv import load_dotenv
from langchain.chains import ConversationalRetrievalChain
from langchain_core.prompts import ChatPromptTemplate,PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_models import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
import os
from dotenv import load_dotenv
import streamlit as st
import streamlit_chat 
from langchain_groq import ChatGroq
global seed 
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_models import ChatOpenAI
from langchain.docstore.document import Document
import pandas as pd

load_dotenv()


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# groq_api_key = os.getenv("GROQ_API_KEY")


class prompts:

    prompt = PromptTemplate.from_template("""
                                          
            You are a helpful fitness assistant. Use the following context to answer the question The Level is provided for you to get a better idea on how to answer the question
                                           .
            If you don't know the answer, just say that you don't know, don't try to make up an answer.Also make sure to mention the level passed for the user.
            Context:
            {context}

            Chat History:
            {history}

            Question:
            {question}
        
            Level:
            {level}

            Answer:
            """)

# Data Filteration
def filter_transform_data(dataframe):

    dataframe.drop("RatingDesc",axis=1,inplace=True)

    dataframe.dropna(subset=["Desc","Equipment"],inplace=True)

    dataframe.drop("Rating",inplace=True,axis=1)

    # transform data

    document_data = dataframe.to_dict(orient="records")

    return document_data


def get_context(vector_store,query,level):

    results = vector_store.max_marginal_relevance_search(
        
            query=query,
            k=5,
            filter={"Level": level},
        )
    
    # Creating the LLM Chain

        # Pass your context manually from retrieved documents
    context = "\n\n".join([doc.page_content for doc in results])

    return context

def generate_vector_store():

    # embedding = OpenAIEmbeddings(

    if "vector_store" not in st.session_state:

        langchain_documents = []

        dataframe = pd.read_csv("megaGymDataset.csv",index_col=0)

        document_data = filter_transform_data(dataframe)

        # Iterate through the sample data and create Document objects
        for item in document_data:
            # Formulate the page_content string
            page_content = (
                f"Title: {item['Title']}\n"
                f"Type:{item['Type']}\n"
                f"BodyPart: {item['BodyPart']}\n"
                f"Desc: {item['Desc']}\n"
                f"Equipment: {item['Equipment']}\n"
            )
            
            # Create the metadata dictionary
            metadata = {"Level": item['Level']}
            
            # Create the Document object
            doc = Document(page_content=page_content, metadata=metadata)
            
            # Add the Document to our list
            langchain_documents.append(doc)

        # creating the session_state for vector_store

        embedding = OpenAIEmbeddings(openai_api_key=os.environ["OPENAI_API_KEY"])
        # if path not exist
        if not os.path.exists("db"):

            st.session_state.vector_store = Chroma.from_documents(langchain_documents,embedding=embedding,collection_name="gym-queries-data",persist_directory = "db")
            st.session_state.vector_store.persist()

        else:

            st.session_state.vector_store  = Chroma(

                persist_directory="db",
                embedding_function=embedding
            )

    return st.session_state.vector_store

def get_conversational_chain(vector_store,query,level):

    llm = ChatOpenAI(temperature=0.5,model_name="gpt-4o")

    # llama3-70b-8192
    # llm = ChatGroq(
    #         temperature=1, 
    #         groq_api_key = groq_api_key, 
    #         model_name="llama3-70b-8192",
    #         max_tokens=150,
    #     #     top_p=0.95,
    #     #     frequency_penalty=1,
    #     #     presence_penalty=1,
    #     )
    # llm_chain = LLMChain(llm=llm, prompt=prompts.prompt)

    if "memory" not in st.session_state:

        st.session_state.memory = ConversationBufferMemory(memory_key="history", input_key="question", return_messages=True)

        st.session_state.conversational_chain = LLMChain(
            llm=llm,
            # taking the prompt template
            prompt=prompts.prompt,
            memory=st.session_state.memory
        )


    return st.session_state.conversational_chain,st.session_state.memory

def stick_it_good():

    # make header sticky.
    st.markdown(
        """
            <div class='fixed-header'/>
            <style>
                div[data-testid="stVerticalBlock"] div:has(div.fixed-header) {
                    position: sticky;
                    top: 2.875rem;
                    background-color: ##393939;
                    z-index: 999;
                }
                .fixed-header {
                    border-bottom: 1px solid black;
                }
            </style>
        """,
        unsafe_allow_html=True
    )


def show_privacy_policy():
    st.title("Privacy Policy")
   

def show_terms_of_service():
    st.title("Terms of Service")
   
seed = 0

def main():

    global seed

    page = st.sidebar.selectbox("Choose a page", ["Home", "Privacy Policy", "Terms of Service"])

    if page == "Privacy Policy":

        show_privacy_policy()

    elif page == "Terms of Service":

        show_terms_of_service()

    else:

        st.write("Welcome to the Home Page")

        with st.container():

            st.title("Workout Wizard")
            stick_it_good()
            

        with st.sidebar:

            if "seed" not in st.session_state:
                
                st.session_state.seed = 0
                
            # Display the image using the URL

            choose_mode = st.selectbox('Choose Workout Level',["Beginner","Intermediate","Expert"])


            st.markdown("<h2 style='text-align: center;'>Choose Your Avatar</h2>", unsafe_allow_html=True)

            # st.markdown(f"<h2 style='text-align: center;'>{st.button("Back")}</h2>", unsafe_allow_html=True)

            # Center the buttons using HTML and CSS
            col1, col2, col3 = st.columns([1, 1, 1])

    
            with col1:
                
                st.write("")  # Empty column for spacing

            with col2:

                print(st.session_state.seed)

                choose_Avatar = st.button("Next")

                choose_Avatar_second = st.button("Back")


                if choose_Avatar:

                    st.session_state.seed += 1

                if choose_Avatar_second:

                    st.session_state.seed -= 1

                avatar_url = f"https://api.dicebear.com/9.x/adventurer/svg?seed={st.session_state.seed}"

                st.image(avatar_url, caption=f"Avatar {st.session_state.seed }")

            with col3:

                st.write("")  # Empty column for spacing

        
        streamlit_chat.message("Hi. I'm your friendly Gym Assistant Bot.")
        streamlit_chat.message("Ask me anything about the gym! Just don’t ask me to do any push-ups... I'm already *up* and running!")
        streamlit_chat.message("If you want to change your workout level and avatar, press the top left arrow and you will have options to make changes")


        question = st.chat_input("Ask a question related to your GYM queries")


        if "conversation_chain" not in st.session_state:

            st.session_state.conversation_chain = None


        # if question:

        # Converstion chain
        if st.session_state.conversation_chain == None:
            # st.session_state.vectors

            print("the vector store generated")

            st.session_state.vector_store = generate_vector_store()

            st.session_state.conversation_chain, st.session_state.memory = get_conversational_chain(st.session_state.vector_store,question,choose_mode)

        # the session state memory
        if st.session_state.memory != None:

            for i,message in enumerate(st.session_state.memory.chat_memory.messages):

                if i%2 == 0:

                    suffix = f" for {choose_mode} level"

                    # Check if the message ends with the suffix and strip it
                    if message.content.endswith(suffix):

                        message.content = message.content[:-len(suffix)]

                    # message.content = message.content.strip(f" for {choose_mode} level")

                    print("this is the message content",message.content)

                    streamlit_chat.message(message.content,is_user=True, avatar_style="adventurer",seed=st.session_state.seed, key=f"user_msg_{i}")

                else:

                    streamlit_chat.message(message.content,key=f"bot_msg_{i}")

                    st.write("--------------------------------------------------")

        if question:
            
                streamlit_chat.message(question,is_user=True, avatar_style="adventurer",seed=st.session_state.seed)

                print(question)

                print("------------------------")

                # GETTING THE CONTEXT AND ANSWER FROM THE MODEL

                context = get_context(st.session_state.vector_store,question,choose_mode)

                print("context::",context)
                print("the choose mode:",choose_mode)

                response = st.session_state.conversational_chain.run({"context": context, "question": question,"level":choose_mode})

                streamlit_chat.message(response)

        
if __name__ == "__main__":

    main()
