# MediGuru — Multilingual Medical Q&A Assistant

> AI-powered medical chatbot that answers questions from a PDF in 30+ languages,
> grounded in your document via a RAG pipeline — not hallucination.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![LangChain](https://img.shields.io/badge/LangChain-RAG-green)
![Gemini](https://img.shields.io/badge/Gemini-1.5%20Flash-purple)
![Gradio](https://img.shields.io/badge/UI-Gradio-orange)
![Languages](https://img.shields.io/badge/Languages-30%2B-teal)

## What is MediGuru?

MediGuru is an AI-powered medical Q&A chatbot that answers health questions 
from a preloaded medical PDF in over 30 languages. Built on a 
Retrieval-Augmented Generation (RAG) pipeline, it grounds every answer 
in your document — not hallucination — and automatically responds in the 
user's own language.

## How It Works

1. **PDF Ingestion** — Loads and chunks the medical PDF into a FAISS vector store
2. **Semantic Search** — Finds the most relevant chunks for each query
3. **LLM Answer** — Google Gemini 1.5 Flash generates a grounded response
4. **Auto-Translation** — Detects the user's language and translates the reply back

## Tech Stack

| Layer | Tool |
|---|---|
| LLM | Google Gemini 1.5 Flash |
| Embeddings | HuggingFace multilingual Sentence Transformers |
| Vector Store | FAISS |
| RAG Framework | LangChain |
| Translation | deep-translator (Google Translate) |
| UI | Gradio |
| Runtime | Google Colab |

## Supported Languages

English, Hindi, Telugu, Tamil, Kannada, Malayalam, Bengali, Marathi, Punjabi,
Urdu, French, German, Spanish, Arabic, Russian, Chinese, Japanese, Korean,
Polish, Ukrainian, Vietnamese, Indonesian, Swahili, and more.

## Disclaimer

> MediGuru is for informational purposes only. It is not a substitute for 
> professional medical advice, diagnosis, or treatment.
