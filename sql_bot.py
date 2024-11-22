from dotenv import load_dotenv   
from langchain_core.prompts import ChatPromptTemplate
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_google_genai import ChatGoogleGenerativeAI
import streamlit as st
from langchain.schema import HumanMessage,AIMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter    

# Load PDF and split text
pdf_path="temel-oracle-sqlleri.pdf"
loader = PyPDFLoader(pdf_path)  # Load doc
data = loader.load()


text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)  # Text split
docs = text_splitter.split_documents(data)
print(f"Total chunks: {len(docs)}")

# Load environment variables
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

# Configure Gemini
if not api_key:
    raise ValueError("GEMINI_API_KEY not found in environment variables")
genai.configure(api_key=api_key)

# Initialize embeddings
embedding = GoogleGenerativeAIEmbeddings(  # Embedding
    model="models/embedding-001",
    google_api_key=api_key
)

# Create vectorstore and retriever
vectorstore = FAISS.from_documents(docs, embedding)  # Vectorstore
vectorstore.save_local("faiss_index")

# Load vectorstore and create retriever
vectorstore = FAISS.load_local("faiss_index", embeddings=embedding, allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={
        "k": 3,  # Max number of chunk size
        "score_threshold": 0.7  # Threshold for similarity_search
    }
)

# Create prompt template
prompt_template = """ 
Sen yardımsever bir yapay zeka asistanısın.
Sana verilen metinle alakalı soruları cevaplayabilirsin. 
Eğer metinle alakasız bir soru sorulursa 'Ben de bu bilgi yok nerden bileyim ben. git google'a sor 😎 😎' de.
Cevaplarının sonuna cümlenle alakalı emojiler koy.

Bağlam: {context}
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", prompt_template),
    ("human", "{input}")
])


llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.2)


document_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)

# Create the retrieval chain
retrieval_chain = create_retrieval_chain(retriever, document_chain)

def respond_to_question(question: str):
    response = retrieval_chain.invoke({"input": question})
    return response["answer"]

st.sidebar.title("SQLBot  Klavuzu")
st.sidebar.write("👨🏼‍💻 **Özellikler:**")
st.sidebar.write("- Sql ile ilgili istediğin soruyu✅.")
st.sidebar.write("- İstediğin herhangi bir anda sohbeti indirebilirsin✅.")
st.sidebar.write("- Sql ile ilgili istediğin soruyu sorabilirsin✅.")
st.sidebar.write("💡 **İpucu:**")
st.sidebar.write("Açık ve detaylı sorular  daha net cevaplar almanı sağlar")
st.title("SQL CHATBOT 🤖👨🏼‍💻")

question = st.chat_input("yazınız")

if "chat_history" not in st.session_state: #Keep conversation history
    st.session_state.chat_history = []

# Display chat history
for messages in st.session_state.chat_history:
    if isinstance(messages, HumanMessage):   # control if this message is written by human
        with st.chat_message("Human",avatar="👨🏼‍💻"):
            st.markdown(messages.content)    #display the message supported by html on streamlit page
    elif isinstance(messages, AIMessage):
        with st.chat_message("AI",avatar="🤖"):
            st.markdown(messages.content)


if question is not None and question != "":  # prevent sending  empty message

    # Add user question to chat history as HumanMessage
    st.session_state.chat_history.append(HumanMessage(content=question)) 
    # Display user question
    with st.chat_message("Human",avatar="👨🏼‍💻"):
        st.markdown(question)

    
    with st.chat_message("AI",avatar="🤖"):
        answer = respond_to_question(question)  # generate answer based on user question
        st.markdown(answer)

        # Add AI response to chat history as AImessage
        st.session_state.chat_history.append(AIMessage(content=answer))

# Download chat history as a text file
print(st.session_state.chat_history)
chat_history_text = "\n".join(
    f"{type(message).__name__}: {message.content}" for message in st.session_state.chat_history
)
st.download_button(
    label="Sohbeti İndir",
    data=chat_history_text,
    file_name="chat_history.txt",
    mime="text/plain"
)
