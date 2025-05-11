# 💬 RAG-Powered Multi-Agent Q&A Assistant (Streamlit + Pinecone + LLaMa3)

This application is a document-aware chatbot built with **LangChain**, **NVIDIA LLaMa3**, and **Pinecone**, capable of answering questions by referencing uploaded PDFs or using its own language model intelligence when documents are irrelevant.

---

## 🚀 Live App

👉 [Click here to try it on Streamlit Cloud](https://anchitchourasia-rag-chat.streamlit.app/)

---

## 📂 Features

✅ Upload one or more PDF documents  
✅ Embed them into Pinecone vector DB with chunking  
✅ Ask questions — get context-based answers from the documents  
✅ If the answer isn't in the PDFs, fallback to LLaMa3's own knowledge  
✅ Built-in calculator agent for math queries  
✅ Clean, chat-style Streamlit UI  
✅ Sample PDFs are included for demo testing  

---

## 📄 Example PDFs

These documents are available directly in the app for testing:
- `FAQ.pdf`
- `Privacy Policy.pdf`
- `Terms.pdf`

Users can download and re-upload them for live embedding.

---

## 🤖 Example Questions

- What payment methods do you accept?  
- What is your return policy?  
- How do you use my personal data?  
- What are the terms of service for cancellations?  
- Calculate 43 * 17  
- Define artificial intelligence  
- What is quantum computing?

---

## 🛠️ Technologies Used

- 🧠 LLM: `meta/llama3-70b-instruct` via NVIDIA NIM
- 🔍 Vector DB: Pinecone v3 (serverless, cosine)
- 🔗 LangChain & NVIDIA Embeddings
- 🌐 Deployed with Streamlit Cloud
- 🐍 Python 3.10+

---

## 📦 How to Run Locally
Install dependencies

bash
Copy
Edit
pip install -r requirements.txt
Set environment variables

Create a .env file:

ini
Copy
Edit
NVIDIA_API_KEY=your_nvidia_api_key
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_ENV=your_pinecone_env
Run the app

bash
Copy
Edit
streamlit run main.py
📤 Submit Assignment
Public App URL: https://anchitchourasia-rag-chat.streamlit.app

yaml
Copy
Edit


1. **Clone the repository**

```bash
git clone https://github.com/anchitchourasia/RAG-Powered-Multi-Agent-Q-A-Assistant.git
cd RAG-Powered-Multi-Agent-Q-A-Assistant
