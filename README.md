# 🎥 YouTube RAG Chatbot

**YouTube RAG Chatbot** is a **Retrieval-Augmented Generation (RAG)** system that allows you to ask questions about any YouTube video transcript.  
It uses **LangChain**, **FAISS**, **HuggingFace embeddings**, and **Google Generative AI (Gemini LLM)** to provide accurate, context-aware answers.

Built with **Python** and **Streamlit** for an interactive, user-friendly interface.

---

## 🔹 Features

- Input any **YouTube Video ID** to fetch the transcript dynamically.
- Ask questions based on the **video transcript**.
- **Vector embeddings** via HuggingFace + **FAISS** for fast similarity search.
- **Chain-based RAG pipeline** using `RunnableParallel` + prompt + **Google Generative AI** + output parser.
- **Streamlit UI** for smooth interaction.
- Handles videos **without transcripts** gracefully.

---
