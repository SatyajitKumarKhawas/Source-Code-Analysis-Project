import os
import sys
import stat
import time
import shutil
import subprocess
import streamlit as st
from langchain.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv
from src.helper import repo_ingestion

load_dotenv()

# Works both locally and on Streamlit Cloud
try:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
except Exception:
    GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

st.set_page_config(page_title="GitHub Repo Chatbot", page_icon="💬", layout="wide")
st.title("💬 GitHub Repo Chatbot")


# ------------------------------------------------------------------ #
#  Session State Initialization                                        #
# ------------------------------------------------------------------ #

if "messages" not in st.session_state:
    st.session_state.messages = []

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "db_version" not in st.session_state:
    st.session_state.db_version = 0          # incremented on every new ingestion

if "current_repo" not in st.session_state:
    st.session_state.current_repo = ""       # track which repo is loaded


# ------------------------------------------------------------------ #
#  Utility Helpers                                                     #
# ------------------------------------------------------------------ #

def force_remove_readonly(func, path, excinfo):
    os.chmod(path, stat.S_IWRITE)
    func(path)


def safe_remove_dir(path):
    """Remove directory safely on Windows and Linux."""
    if not os.path.exists(path):
        return True

    # On Linux (Streamlit Cloud) — simple rmtree
    if os.name != "nt":
        try:
            shutil.rmtree(path)
            return True
        except Exception as e:
            st.warning(f"⚠️ Could not delete `{path}`: {e}")
            return False

    # On Windows — retry with readonly fix
    for attempt in range(5):
        try:
            shutil.rmtree(path, onexc=force_remove_readonly)
            return True
        except Exception:
            time.sleep(1 + attempt)

    # Windows last resort: file by file
    for root, dirs, files in os.walk(path, topdown=False):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                os.chmod(file_path, stat.S_IWRITE)
                os.remove(file_path)
            except Exception:
                pass
        for d in dirs:
            try:
                os.rmdir(os.path.join(root, d))
            except Exception:
                pass
    try:
        os.rmdir(path)
        return True
    except Exception:
        st.warning(
            f"⚠️ Could not fully delete `{path}`. "
            "Stop the app, delete it manually, then restart."
        )
        return False


def run_store_index():
    """Run store_index.py in a separate subprocess so ChromaDB
    connection is fully isolated from the Streamlit process."""
    result = subprocess.run(
        [sys.executable, "store_index.py", "--clean"],
        capture_output=True,
        text=True,
        timeout=300
    )
    return result


# ------------------------------------------------------------------ #
#  Load RAG Chain — versioned cache key forces reload on new ingestion #
# ------------------------------------------------------------------ #

@st.cache_resource(show_spinner=False)
def load_chain(db_version: int):
    """
    db_version is the cache key.
    Incrementing it forces st.cache_resource to treat this
    as a completely new function call and rebuild the chain.
    """
    if not os.path.exists("db"):
        return None

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    vectordb = Chroma(
        persist_directory="db",
        embedding_function=embeddings
    )

    llm = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name="openai/gpt-oss-120b",
        temperature=0.5,
    )

    retriever = vectordb.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 3}
    )

    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Given the chat history and the latest user question, "
         "reformulate the question to be standalone and clear. "
         "Do NOT answer it, just reformulate if needed."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])

    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are an expert code assistant. Use the retrieved code context "
         "below to answer the user's question clearly and concisely.\n\n"
         "Context:\n{context}"),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    return rag_chain


# ------------------------------------------------------------------ #
#  Sidebar                                                             #
# ------------------------------------------------------------------ #

with st.sidebar:
    st.header("📂 Load a GitHub Repo")
    repo_url = st.text_input("Enter GitHub Repo URL")

    # Show currently loaded repo
    if st.session_state.current_repo:
        st.success(f"✅ Loaded: `{st.session_state.current_repo.split('/')[-1]}`")

    if st.button("⚡ Ingest Repo"):
        if repo_url.strip() == "":
            st.warning("Please enter a valid GitHub repo URL.")

        elif repo_url.strip() == st.session_state.current_repo:
            st.info("This repo is already loaded!")

        else:
            with st.spinner("Step 1/3: Releasing old connections..."):
                st.cache_resource.clear()
                time.sleep(3 if os.name == "nt" else 1)

            with st.spinner("Step 2/3: Cloning repository..."):
                try:
                    safe_remove_dir("repo")
                    repo_ingestion(repo_url.strip())
                except Exception as e:
                    st.error(f"⚠️ Clone failed: {e}")
                    st.stop()

            with st.spinner("Step 3/3: Building vector database..."):
                try:
                    result = run_store_index()
                    if result.returncode != 0:
                        st.error(f"⚠️ store_index.py failed:\n{result.stderr}")
                        st.stop()
                except subprocess.TimeoutExpired:
                    st.error("⚠️ Timed out building vector DB.")
                    st.stop()
                except Exception as e:
                    st.error(f"⚠️ Error: {e}")
                    st.stop()

            # ✅ Increment version — this forces load_chain() to reload
            st.session_state.db_version += 1
            st.session_state.current_repo = repo_url.strip()

            # Reset chat for new repo
            st.session_state.messages = []
            st.session_state.chat_history = []

            st.success("✅ Repo ingested! Ask your question below.")
            st.rerun()

    st.markdown("---")

    if st.button("🗑️ Clear Repo & DB"):
        with st.spinner("Clearing..."):
            st.cache_resource.clear()
            time.sleep(3 if os.name == "nt" else 1)
            safe_remove_dir("repo")
            safe_remove_dir("db")

        st.session_state.messages = []
        st.session_state.chat_history = []
        st.session_state.current_repo = ""
        st.session_state.db_version = 0
        st.success("✅ Cleared!")
        st.rerun()

    if st.button("🧹 Clear Chat History"):
        st.session_state.messages = []
        st.session_state.chat_history = []
        st.rerun()

    st.markdown("---")
    st.markdown("### ℹ️ Model Info")
    st.markdown("**LLM:** Groq `qwen-qwq-32b`")
    st.markdown("**Embeddings:** `all-MiniLM-L6-v2` (local)")

    if os.name == "nt":
        st.info("💡 If delete fails, stop the app, delete `db/` manually, and restart.")


# ------------------------------------------------------------------ #
#  Main Chat Interface                                                 #
# ------------------------------------------------------------------ #

if not os.path.exists("db"):
    st.info("👈 Please ingest a GitHub repo from the sidebar to get started.")
    st.stop()

# Load chain using db_version as cache key
try:
    rag_chain = load_chain(st.session_state.db_version)
    if rag_chain is None:
        st.info("👈 Please ingest a GitHub repo from the sidebar to get started.")
        st.stop()
except Exception as e:
    st.error(f"⚠️ Error loading chain: {e}")
    st.stop()

# Display existing messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask something about the repo..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("🤔 Thinking..."):
            try:
                result = rag_chain.invoke({
                    "input": prompt,
                    "chat_history": st.session_state.chat_history
                })
                answer = result["answer"]
            except Exception as e:
                answer = f"⚠️ Error: {e}"
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.session_state.chat_history.extend([
        HumanMessage(content=prompt),
        AIMessage(content=answer)
    ])