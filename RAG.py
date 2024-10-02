import gradio as gr
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
import ollama

# Function to load, split, and retrieve documents from PDF
def load_and_retrieve_docs_from_pdf(pdf_path):
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    embeddings = OllamaEmbeddings(model="mxbai-embed-large")
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    return vectorstore.as_retriever()

# Function to format documents
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Function that defines the RAG chain using PDF
def rag_chain_pdf(pdf_path, question):
    retriever = load_and_retrieve_docs_from_pdf(pdf_path)
    retrieved_docs = retriever.invoke(question)
    formatted_context = format_docs(retrieved_docs)
    formatted_prompt = f"Question: {question}\n\nContext: {formatted_context}"
    response = ollama.chat(model='llama3-korean', 
                           messages=[
                                {"role": "system",
                                 "content": """
                                 You are a helpful AI assistant. Please answer the user's questions kindly. 
                                 당신은 유능한 AI 어시스턴트입니다. 사용자의 질문에 대해 어느 정보를 근거로 답변해주는지와 함께 친절하게 한국어로 답변해주세요.
                                 """
                                },
                                {"role": "user", "content": formatted_prompt}])
    return response['message']['content']

# Gradio interface for RAG with PDF
iface = gr.Interface(
    fn=rag_chain_pdf,
    inputs=["file", "text"],
    outputs="text",
    title="LLAMA 3: RAG Chain with PDF Reference",
    description="Upload a PDF and enter a query to get answers from the RAG chain."
)

# Launch the app
iface.launch()