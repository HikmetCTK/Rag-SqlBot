# Simple Rag-SqlBot
Sql Chatbot for answering sql questions based on sql pdf file

![sqlbottest](https://github.com/user-attachments/assets/5621cc68-7644-4fa1-b925-3440f724d819)


SQL Chatbot is an AI-powered chat application that can answer your questions on SQL. It aims to help users learn, understand and improve their SQL knowledge. Chatbot has the ability to extract information from PDF files and present this information in a way that is relevant to users' questions.

🚀 Project Features:

🔍 PDF Supported Knowledge Extraction:
Chatbot extracts text from a PDF file containing basic SQL commands and uses this text as a knowledge base.

🧠Natural Language Processing:
Integrated with Google's Gemini model, it provides fast and accurate answers to user questions.

📖 Text Snipping and Search:
Breaks text into meaningful chunks using RecursiveCharacterTextSplitter and answers user questions based on these chunks.

💬 Personal Chat Experience:
It stores the messages written by users and the AI's responses in the chat history, thus providing a seamless conversation.

⏬ Chat History Download:
Users can download chat history as a text file and review it later.

⚡ User Friendly Interface:
A clean and simple interface designed with Streamlit. Users can easily ask SQL related questions.

🔧 Technologies Used:

Streamlit: Used to provide a modern and interactive web interface.

LangChain: Used to facilitate natural language processing and document processing.

Google Gemini AI: To provide accurate and fast answers to questions.

PyPDFLoader: For uploading PDF files and extracting text.

FAISS: Vector-based search mechanism for quick access to information.

dotenv: For securely managing API keys.
