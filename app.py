import os
import tempfile
import hashlib

import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader

import chromadb
from sentence_transformers import SentenceTransformer

from google import genai


# ============================================================
# CONFIGURATION
# ============================================================

load_dotenv()

st.set_page_config(
    page_title="AI Student Study Assistant",
    page_icon="🎓",
    layout="wide"
)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    st.error("GEMINI_API_KEY is missing. Add it to your .env file.")
    st.stop()


# ============================================================
# GEMINI
# ============================================================

client = genai.Client(api_key=GEMINI_API_KEY)


# ============================================================
# EMBEDDING MODEL
# ============================================================

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer(
        "all-MiniLM-L6-v2"
    )


embedding_model = load_embedding_model()


# ============================================================
# CHROMA DATABASE
# ============================================================

@st.cache_resource
def load_chroma():
    chroma_client = chromadb.PersistentClient(
        path="./chroma_db"
    )

    collection = chroma_client.get_or_create_collection(
        name="student_study_material"
    )

    return collection


collection = load_chroma()


# ============================================================
# SESSION STATE
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []


# ============================================================
# FUNCTIONS
# ============================================================

def extract_text_from_pdf(uploaded_file):
    """Extract text from uploaded PDF."""

    reader = PdfReader(uploaded_file)

    pages = []

    for page in reader.pages:
        text = page.extract_text()

        if text:
            pages.append(text)

    return "\n".join(pages)


def split_text(text, chunk_size=1000, overlap=150):
    """Split large text into smaller chunks."""

    chunks = []

    start = 0

    while start < len(text):

        end = start + chunk_size

        chunk = text[start:end]

        if chunk.strip():
            chunks.append(chunk.strip())

        start += chunk_size - overlap

    return chunks


def create_embeddings(texts):
    """Convert text into embeddings."""

    embeddings = embedding_model.encode(
        texts,
        normalize_embeddings=True
    )

    return embeddings.tolist()


def add_pdf_to_database(uploaded_file):
    """Process PDF and store chunks in ChromaDB."""

    pdf_text = extract_text_from_pdf(uploaded_file)

    if not pdf_text.strip():
        return 0

    chunks = split_text(pdf_text)

    embeddings = create_embeddings(chunks)

    file_hash = hashlib.md5(
        uploaded_file.getvalue()
    ).hexdigest()

    ids = [
        f"{file_hash}_{i}"
        for i in range(len(chunks))
    ]

    metadatas = [
        {
            "source": uploaded_file.name,
            "chunk": i
        }
        for i in range(len(chunks))
    ]

    collection.upsert(
        ids=ids,
        documents=chunks,
        embeddings=embeddings,
        metadatas=metadatas
    )

    return len(chunks)


def search_documents(question, number_of_results=5):
    """Find relevant study material."""

    question_embedding = create_embeddings(
        [question]
    )[0]

    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=number_of_results
    )

    documents = results.get("documents", [[]])[0]

    metadatas = results.get("metadatas", [[]])[0]

    return documents, metadatas


def ask_gemini(question, context):
    """Generate answer using Gemini."""

    prompt = f"""
You are an AI Student Study Assistant.

Your job is to help students understand their study material.

Use the provided study material whenever possible.

STUDY MATERIAL:
{context}

STUDENT QUESTION:
{question}

Instructions:

1. Explain the answer clearly.
2. Use simple student-friendly language.
3. Give examples when useful.
4. Use bullet points for important information.
5. If the question is related to an exam, provide an exam-friendly answer.
6. Do not invent information that is not supported by the study material.
7. If the answer cannot be found in the study material, clearly say that
   and then provide a general explanation if possible.
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    return response.text


def generate_summary(text):
    """Generate summary of study material."""

    prompt = f"""
Summarize the following study material for a college student.

Give:

- Main concepts
- Important definitions
- Key points
- Important formulas if present
- Exam points
- A short revision summary

STUDY MATERIAL:

{text}
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    return response.text


def generate_mcqs(text, count=5):
    """Generate multiple choice questions."""

    prompt = f"""
Create {count} multiple-choice questions from the following study material.

For every question provide:

Question:
A)
B)
C)
D)

Correct Answer:
Explanation:

Study material:

{text}
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    return response.text


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("🎓 Study Assistant")

    st.write(
        "Upload your study materials and ask questions "
        "from them."
    )

    uploaded_files = st.file_uploader(
        "📚 Upload PDF Notes",
        type=["pdf"],
        accept_multiple_files=True
    )

    if uploaded_files:

        for uploaded_file in uploaded_files:

            if uploaded_file.name not in st.session_state.uploaded_files:

                with st.spinner(
                    f"Processing {uploaded_file.name}..."
                ):

                    chunk_count = add_pdf_to_database(
                        uploaded_file
                    )

                st.session_state.uploaded_files.append(
                    uploaded_file.name
                )

                st.success(
                    f"{uploaded_file.name} added "
                    f"({chunk_count} chunks)"
                )

    st.divider()

    st.subheader("🛠️ Study Tools")

    tool = st.radio(
        "Choose a tool",
        [
            "AI Chat",
            "Summarize Notes",
            "Generate MCQs"
        ]
    )

    st.divider()

    st.caption(
        "AI Student Study Assistant"
    )


# ============================================================
# MAIN UI
# ============================================================

st.title("🎓 AI Student Study Assistant")

st.write(
    "Your personal AI assistant for learning, "
    "revision and exam preparation."
)


# ============================================================
# AI CHAT
# ============================================================

if tool == "AI Chat":

    st.subheader("💬 Ask Your Study Assistant")

    if not st.session_state.uploaded_files:

        st.info(
            "Upload a PDF from the sidebar to start "
            "asking questions."
        )

    for message in st.session_state.messages:

        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    question = st.chat_input(
        "Ask something about your study material..."
    )

    if question:

        st.session_state.messages.append(
            {
                "role": "user",
                "content": question
            }
        )

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):

            with st.spinner("Thinking..."):

                documents, metadatas = search_documents(
                    question
                )

                if documents:

                    context = "\n\n---\n\n".join(
                        documents
                    )

                    answer = ask_gemini(
                        question,
                        context
                    )

                else:

                    answer = (
                        "I couldn't find relevant "
                        "study material."
                    )

                st.markdown(answer)

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer
            }
        )


# ============================================================
# SUMMARY
# ============================================================

elif tool == "Summarize Notes":

    st.subheader("📝 Summarize Your Notes")

    if not st.session_state.uploaded_files:

        st.info(
            "Upload a PDF first."
        )

    else:

        if st.button(
            "✨ Generate Summary",
            use_container_width=True
        ):

            results = collection.get(
                limit=100
            )

            documents = results.get(
                "documents",
                []
            )

            if documents:

                text = "\n\n".join(
                    documents
                )

                with st.spinner(
                    "Creating summary..."
                ):

                    summary = generate_summary(
                        text
                    )

                st.markdown(summary)

            else:

                st.warning(
                    "No study material found."
                )


# ============================================================
# MCQ GENERATOR
# ============================================================

elif tool == "Generate MCQs":

    st.subheader("❓ Generate Practice Questions")

    if not st.session_state.uploaded_files:

        st.info(
            "Upload a PDF first."
        )

    else:

        number = st.slider(
            "Number of questions",
            min_value=5,
            max_value=20,
            value=5
        )

        if st.button(
            "🧠 Generate MCQs",
            use_container_width=True
        ):

            results = collection.get(
                limit=100
            )

            documents = results.get(
                "documents",
                []
            )

            if documents:

                text = "\n\n".join(
                    documents
                )

                with st.spinner(
                    "Generating questions..."
                ):

                    questions = generate_mcqs(
                        text,
                        number
                    )

                st.markdown(
                    questions
                )

            else:

                st.warning(
                    "No study material found."
                )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "🎓 AI Student Study Assistant • "
    "Learn smarter, revise faster."
)