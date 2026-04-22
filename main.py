# Install dependencies
!pip install gradio faiss-cpu pypdf python-dotenv google-generativeai \
 langchain langchain-google-genai langdetect deep-translator sentence-transformers -q

import os
import gradio as gr
import tempfile
import shutil
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from deep_translator import GoogleTranslator
import langdetect

# Set Google Gemini API key here
GOOGLE_API_KEY = "YOUR_API_KEY"
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

llm = ChatGoogleGenerativeAI(
    model="models/gemini-1.5-flash",
    temperature=0.3,
    google_api_key=GOOGLE_API_KEY,
)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
)

# Expanded set of ISO language codes to explicitly support via translation
SUPPORTED_LANGUAGES = [
    'en', 'hi', 'te', 'fr', 'de', 'ar', 'bn', 'kn', 'ml', 'mr', 'pa', 'ta', 'ur', 'sw',
    'vi', 'fa', 'id', 'el', 'uk', 'tl', 'eu', 'is', 'cy', 'af', 'pl', 'ru', 'es', 'zh',
    'ja', 'ko'
]

def load_vector_store(pdf_path):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(documents)
    vectordb = FAISS.from_documents(chunks, embedding=embeddings)
    return vectordb

def build_qa_chain(vectordb):
    retriever = vectordb.as_retriever()
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )
    return chain

def multilingual_query(qa_chain, query):
    try:
        query_lang = langdetect.detect(query)
    except:
        query_lang = "en"

    result = qa_chain(query)
    answer = result['result']

    # Translate answer back only if detected language is in supported languages and not English
    if query_lang != 'en' and query_lang in SUPPORTED_LANGUAGES:
        try:
            answer = GoogleTranslator(source="auto", target=query_lang).translate(answer)
        except Exception:
            pass

    return {
        "answer": answer,
        "source_documents": result['source_documents']
    }

# Path to your pre-uploaded PDF file in Colab
PDF_PATH = "/content/Health_Diseases_Info.pdf"

def gradio_mediguru_fixed_pdf(question):
    try:
        vectordb = load_vector_store(PDF_PATH)
        qa_chain = build_qa_chain(vectordb)
        result = multilingual_query(qa_chain, question)

        answer = result.get("answer", "No answer generated.")
        sources = "\n\n---\n\n".join([doc.page_content[:500] + "..." for doc in result.get("source_documents", [])])
        return answer, sources
    except Exception as e:
        return f"Error: {e}", ""

iface = gr.Interface(
    fn=gradio_mediguru_fixed_pdf,
    inputs=gr.Textbox(label="Ask your medical question (any language)", lines=2),
    outputs=[gr.Textbox(label="Answer", lines=10), gr.Textbox(label="Source excerpts", lines=10)],
    title="🌍 MediGuru - Medical Q&A Assistant (Expanded Multilingual Support)",
    description=f"Using preloaded PDF document at: {PDF_PATH}",
    allow_flagging="never",
)

iface.launch(share=True)

