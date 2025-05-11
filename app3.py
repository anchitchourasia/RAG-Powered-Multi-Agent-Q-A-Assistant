import streamlit as st
import os
import time
import uuid
from dotenv import load_dotenv
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, ChatNVIDIA
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pinecone import Pinecone

# Load environment variables
load_dotenv()

# Pinecone setup
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_env = os.getenv("PINECONE_ENV")
index_name = "rag"
namespace = "us_census_namespace"

# Initialize Pinecone v3
pc = Pinecone(api_key=pinecone_api_key)
index = pc.Index(index_name)

# Streamlit setup
st.set_page_config(page_title="RAG Chat Assistant", layout="wide")
st.title("💬 Chat with Your Documents (Pinecone v3 + LLaMa3 + Tools)")

# Chat history init
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Upload PDFs
uploaded_files = st.file_uploader("📄 Upload PDF files", type="pdf", accept_multiple_files=True)

# Document Embedding
def vector_embedding():
    if not uploaded_files:
        st.warning("Please upload one or more PDF files.")
        return

    st.session_state.embeddings = NVIDIAEmbeddings()
    docs = []

    for uploaded_file in uploaded_files:
        loader = PyPDFLoader(uploaded_file)
        docs.extend(loader.load())

    if not docs:
        st.error("No documents could be parsed.")
        return

    st.success(f"📄 Loaded {len(docs)} PDF(s).")

    splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    texts = [doc.page_content for doc in chunks]
    embeds = st.session_state.embeddings.embed_documents(texts)

    vectors = []
    for i, doc in enumerate(chunks):
        uid = str(uuid.uuid4())
        vectors.append({
            "id": uid,
            "values": embeds[i],
            "metadata": {
                "source": doc.metadata.get("source", "uploaded_file"),
                "content": doc.page_content,
                "chunk_number": i + 1
            }
        })

    index.upsert(vectors=vectors, namespace=namespace)
    st.success("✅ Vectors stored in Pinecone.")

# Embed button
if st.button("📄 Embed Documents"):
    vector_embedding()

# LLM setup
llm = ChatNVIDIA(model="meta/llama3-70b-instruct")

# Tool routing
def route_query(query):
    query_lower = query.lower().strip()
    if any(op in query_lower for op in ["+", "-", "*", "/", "%", "calculate"]):
        try:
            expr = query_lower.replace("calculate", "").strip()
            result = eval(expr)
            return "calculator", str(result)
        except:
            return "calculator", "⚠️ Invalid math expression."
    return "rag", None

# Search vector DB
def search_similar_chunks(query, top_k=3):
    embed = st.session_state.embeddings.embed_query(query)
    return index.query(
        vector=embed,
        top_k=top_k,
        namespace=namespace,
        include_metadata=True
    )

# Chat input
user_query = st.chat_input("Ask a question about your documents or general topics...")

if user_query:
    route, result = route_query(user_query)
    st.session_state.chat_history.append(("user", user_query))

    if route == "calculator":
        st.session_state.chat_history.append(("assistant", result))

    elif route == "rag":
        try:
            results = search_similar_chunks(user_query)
            matches = results.get("matches", [])
            contexts = [m.get("metadata", {}).get("content", "") for m in matches]
            combined_context = "\n".join(contexts)

            prompt = f"""
Use the context below to help answer the question. If the context is not relevant or missing, answer the question using your own knowledge.

<context>
{combined_context}
</context>

Question: {user_query}
"""
            response = llm.invoke(prompt)
            final_answer = response.content

            st.session_state.chat_history.append((
                "assistant_with_context",
                {
                    "answer": final_answer,
                    "context": matches
                }
            ))

        except Exception as e:
            st.session_state.chat_history.append(("assistant", f"❌ RAG failed: {e}"))

# Chat history rendering
for speaker, message in st.session_state.chat_history:
    if speaker == "assistant_with_context":
        with st.chat_message("assistant"):
            st.markdown(f"💡 According to the context, the answer is:\n\n{message['answer']}")
            with st.expander("📚 Retrieved Document Chunks"):
                for match in message["context"]:
                    meta = match.get("metadata", {})
                    st.markdown(f"**📄 Source:** `{meta.get('source', 'uploaded_file')}`")
                    st.markdown(f"**🔢 Chunk:** {meta.get('chunk_number', 'N/A')}`")
                    st.markdown(meta.get("content", "⚠️ No content available."))
                    st.markdown("---")
    else:
        st.chat_message(speaker).write(message)
