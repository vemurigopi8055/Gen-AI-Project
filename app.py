from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.embeddings import OllamaEmbeddings
from langchain_ollama import OllamaLLM
from langchain.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import streamlit as st

pdf_loader = PyPDFLoader("gullivers-travels.pdf")
pdf_loaded = pdf_loader.load()


text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 100)
final_documents = text_splitter.split_documents(pdf_loaded)

embeddings = (
    OllamaEmbeddings(model = "gemma:2b")
)

chroma_db = Chroma.from_documents(
    final_documents, embedding=embeddings
)

retirever = chroma_db.as_retriever()

llm = OllamaLLM(model = "gemma:2b")

prompt = ChatPromptTemplate([
    ("system", "You are a helpful assistant. Please respond to the questions asked basing on the given context"),
    ("user", {"Context:{context}, Question : {question}"})
])

outparser = StrOutputParser()

rag_chain = (
    {"Context": retirever}|prompt|llm|outparser
)

st.title("Langchain RAG Demo With Gemma Model ask only Questions related to Gullivers-Travels")
input_text = st.text_input("What question do you have in mind?")

if input_text:
    response = rag_chain.invoke(input_text)
    st.write(response)
