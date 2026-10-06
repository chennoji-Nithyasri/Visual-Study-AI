import os
import io
import time
import base64
import mimetypes
import hashlib

import streamlit as st
from dotenv import load_dotenv
from PIL import Image
from pypdf import PdfReader
from docx import Document

import chromadb
from sentence_transformers import SentenceTransformer
from huggingface_hub import InferenceClient


# ============================================================
# 1. LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN", "").strip()

VISION_MODEL = os.getenv(
    "VISION_MODEL",
    "Qwen/Qwen2.5-VL-32B-Instruct"
)

HF_VISION_PROVIDER = os.getenv(
    "HF_VISION_PROVIDER",
    "featherless-ai"
)

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "all-MiniLM-L6-v2"
)

DB_PATH = "./chroma_db"


# ============================================================
# 2. STREAMLIT PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Visual Study Material Explainer",
    page_icon="📚",
    layout="wide"
)


# ============================================================
# 3. TITLE
# ============================================================

st.title("📚 Visual Study Material Explainer")

st.write(
    """
    Upload a diagram/image and your study material.
    The AI will understand the diagram, search the study material,
    and generate a detailed student-friendly explanation.
    """
)


# ============================================================
# 4. CHECK HUGGING FACE TOKEN
# ============================================================

if not HF_TOKEN:
    st.error(
        "❌ Hugging Face API token is missing.\n\n"
        "Create a .env file and add:\n\n"
        "HF_TOKEN=your_huggingface_token"
    )
    st.stop()


# ============================================================
# 5. HUGGING FACE CLIENT
# ============================================================

@st.cache_resource
def load_huggingface_client():

    try:
        client = InferenceClient(
            api_key=HF_TOKEN,
            provider=HF_VISION_PROVIDER
        )

        return client

    except Exception as e:

        st.error(
            f"❌ Could not create Hugging Face client.\n\n{e}"
        )

        return None


hf_client = load_huggingface_client()


# ============================================================
# 6. EMBEDDING MODEL
# ============================================================

@st.cache_resource
def load_embedding_model():

    try:

        model = SentenceTransformer(
            EMBEDDING_MODEL
        )

        return model

    except Exception as e:

        st.error(
            f"❌ Could not load embedding model:\n\n{e}"
        )

        return None


embedding_model = load_embedding_model()


# ============================================================
# 7. CHROMADB
# ============================================================

@st.cache_resource
def load_chroma_collection():

    try:

        client = chromadb.PersistentClient(
            path=DB_PATH
        )

        collection = client.get_or_create_collection(
            name="study_material"
        )

        return collection

    except Exception as e:

        st.error(
            f"❌ ChromaDB error:\n\n{e}"
        )

        return None


collection = load_chroma_collection()


# ============================================================
# 8. PDF TEXT EXTRACTION
# ============================================================

def extract_pdf_text(uploaded_file):

    try:

        pdf_bytes = uploaded_file.getvalue()

        reader = PdfReader(
            io.BytesIO(pdf_bytes)
        )

        pages = []

        for page in reader.pages:

            text = page.extract_text()

            if text:
                pages.append(text)

        return "\n".join(pages)

    except Exception as e:

        st.error(
            f"❌ PDF extraction error:\n\n{e}"
        )

        return ""


# ============================================================
# 9. DOCX TEXT EXTRACTION
# ============================================================

def extract_docx_text(uploaded_file):

    try:

        docx_bytes = uploaded_file.getvalue()

        document = Document(
            io.BytesIO(docx_bytes)
        )

        paragraphs = []

        for paragraph in document.paragraphs:

            text = paragraph.text.strip()

            if text:
                paragraphs.append(text)

        return "\n".join(paragraphs)

    except Exception as e:

        st.error(
            f"❌ DOCX extraction error:\n\n{e}"
        )

        return ""


# ============================================================
# 10. TEXT CHUNKING
# ============================================================

def split_text_into_chunks(
    text,
    chunk_size=800,
    overlap=100
):

    text = text.strip()

    if not text:
        return []

    chunks = []

    start = 0

    while start < len(text):

        end = start + chunk_size

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start = end - overlap

        if start < 0:
            start = 0

        if end >= len(text):
            break

    return chunks


# ============================================================
# 11. CREATE UNIQUE ID
# ============================================================

def create_chunk_id(text, index):

    hash_value = hashlib.md5(
        text.encode("utf-8")
    ).hexdigest()

    return f"{hash_value}_{index}"


# ============================================================
# 12. STORE STUDY MATERIAL IN CHROMADB
# ============================================================

def store_study_material(text):

    if collection is None:
        return 0

    if embedding_model is None:
        return 0

    chunks = split_text_into_chunks(text)

    if not chunks:
        return 0

    embeddings = embedding_model.encode(
        chunks,
        show_progress_bar=False
    )

    ids = []
    documents = []
    vectors = []

    for index, chunk in enumerate(chunks):

        ids.append(
            create_chunk_id(chunk, index)
        )

        documents.append(chunk)

        vectors.append(
            embeddings[index].tolist()
        )

    try:

        collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=vectors
        )

        return len(chunks)

    except Exception as e:

        st.error(
            f"❌ Could not store study material:\n\n{e}"
        )

        return 0


# ============================================================
# 13. SEARCH RELEVANT STUDY MATERIAL
# ============================================================

def search_study_material(
    query,
    number_of_results=5
):

    if collection is None:
        return []

    if embedding_model is None:
        return []

    try:

        query_embedding = embedding_model.encode(
            [query]
        )[0].tolist()

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=number_of_results
        )

        documents = results.get(
            "documents",
            []
        )

        if not documents:
            return []

        return documents[0]

    except Exception as e:

        st.warning(
            f"⚠️ Study material search failed:\n\n{e}"
        )

        return []


# ============================================================
# 14. IMAGE → BASE64 DATA URL
# ============================================================

def image_to_data_url(uploaded_image):

    image_bytes = uploaded_image.getvalue()

    mime_type = uploaded_image.type

    if not mime_type:

        mime_type = mimetypes.guess_type(
            uploaded_image.name
        )[0]

    if not mime_type:

        mime_type = "image/png"

    encoded_image = base64.b64encode(
        image_bytes
    ).decode("utf-8")

    return (
        f"data:{mime_type};base64,"
        f"{encoded_image}"
    )


# ============================================================
# 15. GET TEXT FROM HF RESPONSE
# ============================================================

def extract_response_text(response):

    try:

        if hasattr(response, "choices"):

            if len(response.choices) > 0:

                message = response.choices[0].message

                content = message.content

                if isinstance(content, str):
                    return content

                return str(content)

        return str(response)

    except Exception:

        return str(response)


# ============================================================
# 16. HUGGING FACE CHAT FUNCTION
# ============================================================

def call_huggingface_chat(
    messages,
    max_tokens=1800,
    temperature=0.2,
    max_retries=4
):

    if hf_client is None:

        return None, "Hugging Face client is not available."

    last_error = ""

    for attempt in range(max_retries):

        try:

            response = hf_client.chat.completions.create(
                model=VISION_MODEL,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature
            )

            answer = extract_response_text(
                response
            )

            return answer, None

        except Exception as e:

            last_error = str(e)

            error_lower = last_error.lower()

            # --------------------------------------------
            # 503 SERVER BUSY
            # --------------------------------------------

            if (
                "503" in error_lower
                or "service unavailable" in error_lower
                or "temporarily unavailable" in error_lower
            ):

                if attempt < max_retries - 1:

                    wait_time = 2 ** attempt

                    st.warning(
                        f"⚠️ Hugging Face server is busy. "
                        f"Retrying in {wait_time} seconds..."
                    )

                    time.sleep(wait_time)

                    continue

            # --------------------------------------------
            # 429 RATE LIMIT
            # --------------------------------------------

            if (
                "429" in error_lower
                or "rate limit" in error_lower
                or "too many requests" in error_lower
            ):

                if attempt < max_retries - 1:

                    wait_time = 3 * (attempt + 1)

                    st.warning(
                        f"⚠️ API rate limit reached. "
                        f"Retrying in {wait_time} seconds..."
                    )

                    time.sleep(wait_time)

                    continue

            # --------------------------------------------
            # TIMEOUT
            # --------------------------------------------

            if (
                "timeout" in error_lower
                or "timed out" in error_lower
            ):

                if attempt < max_retries - 1:

                    wait_time = 2 ** attempt

                    st.warning(
                        f"⚠️ Request timed out. "
                        f"Retrying in {wait_time} seconds..."
                    )

                    time.sleep(wait_time)

                    continue

            break

    return None, last_error


# ============================================================
# 17. ANALYZE DIAGRAM
# ============================================================

def analyze_diagram(
    uploaded_image
):

    image_data_url = image_to_data_url(
        uploaded_image
    )

    prompt = """
You are an expert educational AI.

Analyze the uploaded diagram/image carefully.

Do NOT give only a short description.

Provide a detailed educational analysis using this structure:

1. DIAGRAM IDENTIFICATION
   - What is the diagram?
   - What topic does it represent?

2. MAIN CONCEPT
   - Explain the main concept in simple student-friendly language.

3. DIAGRAM COMPONENTS
   - Identify every important box, label, symbol, component,
     arrow, or section visible in the diagram.
   - Explain the purpose of each component.

4. STEP-BY-STEP WORKING
   - Follow the arrows or logical flow.
   - Explain what happens first, second, third, etc.

5. RELATION BETWEEN COMPONENTS
   - Explain how the components are connected.

6. SIMPLE EXPLANATION
   - Explain the complete diagram as if teaching a beginner.

7. REAL-WORLD EXAMPLE
   - Give a simple practical example if applicable.

8. IMPORTANT POINTS
   - Give the important points a student should remember.

9. EXAM QUESTIONS
   - Generate 5 useful exam questions based on this diagram.

10. VIVA QUESTIONS AND ANSWERS
   - Generate 5 short viva questions with answers.

Be accurate.
Do not invent labels or components that cannot be seen.
If something is unclear in the image, explicitly say that it is unclear.
"""

    messages = [

        {
            "role": "system",
            "content": (
                "You are a multimodal educational tutor. "
                "Your job is to understand diagrams and "
                "explain them clearly to students."
            )
        },

        {
            "role": "user",
            "content": [

                {
                    "type": "text",
                    "text": prompt
                },

                {
                    "type": "image_url",
                    "image_url": {
                        "url": image_data_url
                    }
                }

            ]
        }

    ]

    answer, error = call_huggingface_chat(
        messages,
        max_tokens=2200,
        temperature=0.2
    )

    if error:

        return None, error

    return answer, None


# ============================================================
# 18. GENERATE FINAL STUDY EXPLANATION
# ============================================================

def generate_final_explanation(
    diagram_explanation,
    study_material
):

    prompt = f"""
You are an expert AI study tutor.

The student uploaded a diagram and study material.

Your task is to create ONE clear, detailed explanation by combining
the diagram understanding with the relevant study material.

IMPORTANT RULES:

1. Use the study material when it contains relevant information.
2. Do not invent facts that are not supported by the diagram
   or study material.
3. If the study material does not contain information about
   something, say that clearly.
4. Explain difficult technical terms in simple language.
5. Follow the actual flow of the diagram.
6. Make the answer useful for exams and viva.

==============================
DIAGRAM ANALYSIS
==============================

{diagram_explanation}


==============================
RELEVANT STUDY MATERIAL
==============================

{study_material}


==============================
REQUIRED OUTPUT
==============================

# Detailed Study Explanation

## 1. What is the Diagram?

Identify the diagram and explain what topic it represents.

## 2. Main Concept

Explain the main concept in simple language.

## 3. Components of the Diagram

List the important components and explain the purpose of each one.

## 4. Step-by-Step Working

Explain the complete flow from beginning to end.

Use numbered steps.

## 5. Connection With Study Material

Explain how the diagram is related to the uploaded study material.

Mention important concepts from the study material.

## 6. Easy Explanation

Explain the complete topic as if teaching a beginner.

## 7. Example

Give one simple real-world or technical example.

## 8. Important Points for Exam

Give 5-10 important points.

## 9. Short Exam Answer

Give a short exam-ready answer that a student can write in an exam.

## 10. Viva Questions and Answers

Give 5 important viva questions with short answers.

## 11. Quick Revision

Finish with 5 very short points for last-minute revision.
"""

    messages = [

        {
            "role": "system",
            "content": (
                "You are a professional educational tutor. "
                "Your explanations must be accurate, structured, "
                "simple, and useful for engineering students."
            )
        },

        {
            "role": "user",
            "content": prompt
        }

    ]

    answer, error = call_huggingface_chat(
        messages,
        max_tokens=3000,
        temperature=0.2
    )

    if error:

        return None, error

    return answer, None


# ============================================================
# 19. SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ Configuration")

    st.write(
        f"**Vision Model:** `{VISION_MODEL}`"
    )

    st.write(
        f"**Provider:** `{HF_VISION_PROVIDER}`"
    )

    st.write(
        f"**Embedding Model:** `{EMBEDDING_MODEL}`"
    )

    st.divider()

    st.info(
        """
        Workflow:

        📷 Upload Diagram
        ↓
        👁️ Vision AI
        ↓
        📄 Study Material
        ↓
        🔎 RAG Search
        ↓
        🧠 AI Explanation
        ↓
        📚 Exam + Viva
        """
    )


# ============================================================
# 20. STUDY MATERIAL UPLOAD
# ============================================================

st.header("📄 Step 1: Upload Study Material")

study_file = st.file_uploader(
    "Upload your PDF or DOCX study material",
    type=["pdf", "docx"]
)


if study_file:

    file_name = study_file.name

    st.success(
        f"✅ Uploaded: {file_name}"
    )

    if file_name.lower().endswith(".pdf"):

        study_text = extract_pdf_text(
            study_file
        )

    elif file_name.lower().endswith(".docx"):

        study_text = extract_docx_text(
            study_file
        )

    else:

        study_text = ""

    if study_text:

        st.info(
            f"📄 Extracted {len(study_text):,} characters."
        )

        with st.expander(
            "👀 Preview Study Material"
        ):

            st.write(
                study_text[:5000]
            )

        if st.button(
            "💾 Process & Store Study Material",
            type="primary"
        ):

            with st.spinner(
                "Processing study material..."
            ):

                chunk_count = store_study_material(
                    study_text
                )

            if chunk_count > 0:

                st.success(
                    f"✅ Study material processed successfully!"
                    f" Created {chunk_count} chunks."
                )

            else:

                st.error(
                    "❌ Could not process the study material."
                )

    else:

        st.warning(
            "⚠️ No readable text was found."
        )


# ============================================================
# 21. DIAGRAM UPLOAD
# ============================================================

st.divider()

st.header("🖼️ Step 2: Upload Diagram / Image")

diagram_file = st.file_uploader(
    "Upload your diagram, flowchart, graph, architecture, or study image",
    type=[
        "png",
        "jpg",
        "jpeg",
        "webp"
    ],
    key="diagram"
)


if diagram_file:

    st.success(
        f"✅ Diagram uploaded: {diagram_file.name}"
    )

    try:

        image = Image.open(
            diagram_file
        )

        st.image(
            image,
            caption="Uploaded Diagram",
            use_container_width=True
        )

    except Exception as e:

        st.error(
            f"❌ Could not open image:\n\n{e}"
        )


# ============================================================
# 22. ANALYZE BUTTON
# ============================================================

if diagram_file:

    st.divider()

    analyze_button = st.button(
        "🔍 Analyze Diagram & Generate Detailed Explanation",
        type="primary",
        use_container_width=True
    )

    if analyze_button:

        # ----------------------------------------------------
        # STEP 1: VISION ANALYSIS
        # ----------------------------------------------------

        with st.spinner(
            "👁️ AI is understanding the diagram..."
        ):

            diagram_result, diagram_error = analyze_diagram(
                diagram_file
            )

        if diagram_error:

            error_text = diagram_error.lower()

            if (
                "model_not_supported" in error_text
                or "not supported by any provider" in error_text
            ):

                st.error(
                    "❌ The selected Hugging Face model/provider "
                    "does not currently support this request."
                )

                st.code(
                    f"VISION_MODEL={VISION_MODEL}\n"
                    f"HF_VISION_PROVIDER={HF_VISION_PROVIDER}"
                )

            elif (
                "401" in error_text
                or "unauthorized" in error_text
            ):

                st.error(
                    "❌ Hugging Face token is invalid "
                    "or does not have the required access."
                )

            elif "403" in error_text:

                st.error(
                    "❌ Access to the selected model/provider "
                    "was denied."
                )

            elif "503" in error_text:

                st.error(
                    "❌ Hugging Face service is temporarily unavailable "
                    "after several retries."
                )

                st.info(
                    "Please wait a little and try again."
                )

            else:

                st.error(
                    f"❌ Vision AI Error:\n\n{diagram_error}"
                )

            st.stop()

        # ----------------------------------------------------
        # SHOW DIAGRAM ANALYSIS
        # ----------------------------------------------------

        st.subheader(
            "👁️ AI Understanding of the Diagram"
        )

        st.markdown(
            diagram_result
        )

        # ----------------------------------------------------
        # STEP 2: SEARCH STUDY MATERIAL
        # ----------------------------------------------------

        with st.spinner(
            "🔎 Searching relevant study material..."
        ):

            relevant_material = search_study_material(
                diagram_result,
                number_of_results=5
            )

        if relevant_material:

            combined_material = "\n\n".join(
                relevant_material
            )

            st.subheader(
                "📄 Relevant Study Material Found"
            )

            with st.expander(
                "View Retrieved Material"
            ):

                st.write(
                    combined_material
                )

        else:

            combined_material = (
                "No relevant study material was found "
                "in the ChromaDB database."
            )

            st.warning(
                "⚠️ No matching study material was found."
            )

        # ----------------------------------------------------
        # STEP 3: FINAL EXPLANATION
        # ----------------------------------------------------

        with st.spinner(
            "🧠 Generating detailed study explanation..."
        ):

            final_answer, final_error = generate_final_explanation(
                diagram_result,
                combined_material
            )

        if final_error:

            st.error(
                f"❌ Could not generate final explanation:\n\n"
                f"{final_error}"
            )

        else:

            st.divider()

            st.header(
                "📚 Detailed Study Explanation"
            )

            st.markdown(
                final_answer
            )

            # ------------------------------------------------
            # DOWNLOAD EXPLANATION
            # ------------------------------------------------

            st.download_button(
                label="⬇️ Download Explanation",
                data=final_answer,
                file_name="study_explanation.txt",
                mime="text/plain"
            )


# ============================================================
# 23. FOOTER
# ============================================================

st.divider()

st.caption(
    "📚 Visual Study Material Explainer | "
    "Multimodal AI + RAG + Embeddings + ChromaDB + Streamlit"
)