# 💬 RAG Chat Assistant (Pinecone v3 + NVIDIA LLaMa3 + Tools)

An intelligent, multi-modal assistant that:
- 📚 Answers questions based on uploaded PDFs using RAG (Retrieval-Augmented Generation)
- 🤖 Falls back to LLaMa3-70B when document context isn't available
- 🧮 Supports math questions with a calculator tool
- 🖼️ Clean Streamlit chat UI with document chunk source highlighting

---

## 🔧 Features

- 🔍 PDF chunking and vector embedding using NVIDIAEmbeddings
- 🧠 Pinecone vector store (v3 SDK)
- 🤝 Seamless integration with `meta/llama3-70b-instruct`
- 📦 Context-aware RAG with fallback to model-only answers
- ✨ Smart UI with context expander & chat memory

---

## 📁 Setup

### 1. Clone the project
```bash
git clone https://github.com/yourname/rag-chat-assistant.git
cd rag-chat-assistant
