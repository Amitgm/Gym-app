from langchain_community.document_loaders import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import chroma
from langchain_community.llms import openai
from langchain.chains import LLMChain
from dotenv import load_dotenv
from langchain.chains import ConversationalRetrievalChain
from langchain_core.prompts import ChatPromptTemplate,PromptTemplate,SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_models import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
import os
import pinecone
from dotenv import load_dotenv
import streamlit as st
import streamlit_chat 
from langchain_groq import ChatGroq
global seed 



load_dotenv()


os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
groq_api_key = os.getenv("GROQ_API_KEY")


class prompts:

    
    general_system_template='''You are a bot that answers various Gym queries. Based on the {context} type level of exercise chosen , you have to answer gym questions,
    based on the users experience, if you don't have the answer for it, say I don't know instead of making up one.
        '''
    
    user_template = ''' Question :{question} '''


    messages = [

        SystemMessagePromptTemplate.from_template(general_system_template),
        HumanMessagePromptTemplate.from_template(user_template)
    ]

    qa_prompt = ChatPromptTemplate.from_messages(messages)

def generate_vector_store():

    # embedding = OpenAIEmbeddings(


    st.session_state.vector_store = PineconeVectorStore.from_existing_index(

            index_name="pinconedb",
            namespace="wondervector5000",
            embedding=OpenAIEmbeddings(openai_api_key=os.environ["OPENAI_API_KEY"])
    )

    return st.session_state.vector_store


def get_conversational_chain(vector_store):

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

    memory = ConversationBufferMemory(memory_key="chat_history",return_messages=True)

    # print(prompts.prompt)

    # formatted_prompt = prompts.qa_prompt.format(level="Beginner")

    st.session_state.conversational_chain = ConversationalRetrievalChain.from_llm(llm=llm,retriever=vector_store.as_retriever(),memory=memory)

    return st.session_state.conversational_chain

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

seed = 0

def main():

    global seed

    with st.container():

        st.title("Workout Wizard")
        stick_it_good()

    with st.sidebar:


        if "seed" not in st.session_state:
             
             st.session_state.seed = 0
             
        
        # Display the image using the URL

        choose_mode = st.selectbox('Choose Workout Level',["Beginner","Intermediate","Advanced"])


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


    # st.chat_message("assistant", avatar=r"robot.png").write("Hi. I'm your friendly Gym Assistant Bot.")
    # st.chat_message("assistant", avatar=r"robot.png").write("Ask me any Gym related queries and I will do my best to assist you")


    question = st.chat_input("Ask a question related to your GYM queries")

    # st.chat_input("Type your question here.")

    if "chathistory" not in st.session_state:

        st.session_state.chathistory = None

    if "conversation_chain" not in st.session_state:

        st.session_state.conversation_chain = None


    # if question:
   
    if st.session_state.conversation_chain == None:

            vectors = generate_vector_store()

            st.session_state.conversation_chain = get_conversational_chain(vectors)

    if st.session_state.chathistory != None:

            for i,message in enumerate(st.session_state.chathistory):

        
                if i%2 == 0:

                    # streamlit_chat.message(message.content,is_user=True,avatar="😂")
                    # st.chat_message("user", avatar=r"cool.png").write(message.content)

                    suffix = f" for {choose_mode} level"

                    # Check if the message ends with the suffix and strip it
                    if message.content.endswith(suffix):

                        message.content = message.content[:-len(suffix)]

                    # message.content = message.content.strip(f" for {choose_mode} level")

                    print("this is the message content",message.content)

                    streamlit_chat.message(message.content,is_user=True, avatar_style="adventurer",seed=st.session_state.seed)


                else:

                    # streamlit_chat.message(message.content,is_user=False)

                    # st.chat_message("assistant", avatar=r"robot.png").write(message.content)
                    streamlit_chat.message(message.content)


                    st.write("--------------------------------------------------")

    if question:
            
            # st.chat_message("user", avatar=r"cool.png").write(question)

            streamlit_chat.message(question,is_user=True, avatar_style="adventurer",seed=st.session_state.seed)


            # streamlit_chat.message(f"{question}",is_user=True)


            print(question)

            print("------------------------")

            print(question +" for Beginner level")

            question = question + f' for {choose_mode} level'

            response = st.session_state.conversational_chain({"question": question})

            st.session_state.chathistory = response["chat_history"]

            streamlit_chat.message(response['answer'])

           
            # st.chat_message("assistant", avatar=r"robot.png").write(response['answer'])

            # streamlit_chat.message(f"{response['answer']}",is_user=False)



if __name__ == "__main__":

    main()







