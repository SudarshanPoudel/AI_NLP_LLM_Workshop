import streamlit as st
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
import faiss
import json
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import re
from PIL import Image
import io
import tempfile

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="RAG PDF Chatbot",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'texts' not in st.session_state:
    st.session_state.texts = []
if 'index' not in st.session_state:
    st.session_state.index = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'llm' not in st.session_state:
    st.session_state.llm = None

# Constants
DOCS_FOLDER = "docs"
INDEX_FILE = "my_index.faiss"
TEXTS_FILE = "texts.json"

# Create docs folder if it doesn't exist
os.makedirs(DOCS_FOLDER, exist_ok=True)

def load_and_chunk_pdf(pdf_path: str, chunk_size=1000, chunk_overlap=200):
    doc = fitz.open(pdf_path)
    chunks = []

    for page_num, page in enumerate(doc, start=1):
        text = page.get_text()
        if text.strip():
            chunks.append({
                "filename": pdf_path.split("/")[-1],
                "text": text,
                "page_no": page_num
            })

    doc.close()
    return chunks


@st.cache_resource
def load_model():
    """Load the sentence transformer model"""
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource
def load_llm():
    """Load the LLM"""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        st.error("GEMINI_API_KEY not found in environment variables!")
        return None
    return ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        api_key=api_key
    )

def load_or_create_index():
    """Load existing index or create new one"""
    model = load_model()
    
    if os.path.exists(INDEX_FILE):
        index = faiss.read_index(INDEX_FILE)
    else:
        index = faiss.IndexFlatIP(384)  # 384 for MiniLM-L6-v2

    # Load existing texts
    if os.path.exists(TEXTS_FILE):
        with open(TEXTS_FILE, "r") as f:
            texts = json.load(f)
    else:
        texts = []
    
    return index, texts, model

def add_text_to_index(new_items: list[dict], index, texts, model):
    """Add new text chunks to the index"""
    # Extract texts to embed
    raw_texts = [item["text"] for item in new_items]
    embeddings = model.encode(raw_texts, normalize_embeddings=True).astype("float32")

    index.add(embeddings)
    texts.extend(new_items)
    
    # Save index and texts
    faiss.write_index(index, INDEX_FILE)
    with open(TEXTS_FILE, "w") as f:
        json.dump(texts, f, ensure_ascii=False, indent=2)
    
    return index, texts

def clear_index_and_data():
    """Clear the FAISS index, data files, and uploaded documents"""
    files_deleted = []

    # Delete FAISS index file
    if os.path.exists(INDEX_FILE):
        os.remove(INDEX_FILE)
        files_deleted.append(INDEX_FILE)

    # Delete texts JSON file
    if os.path.exists(TEXTS_FILE):
        os.remove(TEXTS_FILE)
        files_deleted.append(TEXTS_FILE)

    # Delete files inside docs folder
    if os.path.isdir(DOCS_FOLDER):
        for filename in os.listdir(DOCS_FOLDER):
            file_path = os.path.join(DOCS_FOLDER, filename)
            try:
                os.remove(file_path)
                files_deleted.append(file_path)
            except Exception as e:
                print(f"Failed to delete {file_path}: {e}")

    # Reset session state
    st.session_state.texts = []
    st.session_state.index = None

    # Clear any cached results
    for key in ['last_result', 'last_question']:
        if key in st.session_state:
            del st.session_state[key]

    return files_deleted


def retrieve(query, index, texts, model, top_k=5):
    """Retrieve relevant chunks for a query"""
    if index.ntotal == 0:
        return []
    
    query_emb = model.encode([query], normalize_embeddings=True).astype('float32')
    D, I = index.search(query_emb, top_k)
    return [texts[i] for i in I[0] if i != -1 and i < len(texts)]

def json_to_obj(json_str: str) -> dict:
    """Parse JSON response from LLM"""
    cleaned = re.sub(r"^```(?:json)?\s*|```$", "", json_str.strip(), flags=re.IGNORECASE)
    
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        st.error(f"Failed to parse JSON: {e}")
        return {}

def get_page_image(pdf_path: str, page_no: int):
    """Extract page as image from PDF"""
    try:
        doc = fitz.open(pdf_path)
        page = doc[page_no - 1]  # Page numbers are 0-indexed in PyMuPDF
        
        # Render page as image
        mat = fitz.Matrix(2, 2)  # Increase resolution
        pix = page.get_pixmap(matrix=mat)
        img_data = pix.tobytes("png")
        doc.close()
        
        # Convert to PIL Image
        image = Image.open(io.BytesIO(img_data))
        return image
    except Exception as e:
        st.error(f"Error extracting page image: {e}")
        return None

def rag_answer(question, index, texts, model, llm):
    """Generate answer using RAG"""
    chunks = retrieve(question, index, texts, model)
    
    if not chunks:
        return {
            "answer": "I'm sorry, but I don't have enough information to answer that.",
            "filename": None,
            "page_no": None
        }
    
    context = json.dumps(chunks)
    
    prompt_template = """
You are a helpful and intelligent study assistant. You are given a list of JSON objects as context, each representing extracted text from a PDF.

Each object contains:
- 'text': the actual content
- 'filename': the PDF file name
- 'page_no': the page number

Your task is to answer the student's question based **primarily** on the 'text' fields in the context. You may reason and infer the answer if the exact wording is not available, as long as your answer is clearly supported by the content.

# Context:
{context}

# Question:
{question}

# Instructions:
- Use only the 'text' field from the context entries for answering the question, but include the most relevant 'filename' and 'page_no' you used.
- Even if the original text is technical or unclear, rewrite the answer in a simple, **student-friendly** way that is easy to understand.
- You can use **Markdown formatting** (headings, bullet points, code blocks, tables, etc.) to make the answer more readable and structured.
- You **may infer** or **summarize** answers from the content to help students understand, even if the answer is not a perfect match.
- Avoid saying you don't know unless the question is entirely unrelated to the context.
- Return a JSON object with:
  - "answer": your helpful, clear, Markdown-formatted answer
  - "filename": the filename of the most relevant entry you used
  - "page_no": the corresponding page number

- If the context contains **nothing relevant at all** to the question, return:
  ```json
  {{
    "answer": "I'm sorry, but I don't have enough information to answer that.",
    "filename": null,
    "page_no": null
  }}
  ```
Respond ONLY with a valid JSON object. Do not include any explanation, formatting, or extra text.
"""
    
    prompt = prompt_template.replace("{context}", context).replace("{question}", question)
    response = llm.invoke(prompt)
    obj = json_to_obj(response.content)
    return obj

# Main Streamlit App
def main():
    st.title("📚 RAG PDF Chatbot")
    st.markdown("Upload PDF documents and ask questions about their content!")

    # Initialize components
    if st.session_state.index is None:
        with st.spinner("Loading models and index..."):
            st.session_state.index, st.session_state.texts, st.session_state.model = load_or_create_index()
            st.session_state.llm = load_llm()

    # Sidebar for PDF upload and management
    with st.sidebar:
        st.header("📁 Document Management")
        
        # Display current documents
        uploaded_docs = [f for f in os.listdir(DOCS_FOLDER) if f.endswith('.pdf')]
        if uploaded_docs:
            st.subheader("Uploaded Documents")
            for doc in uploaded_docs:
                st.write(f"📄 {doc}")
        
        st.subheader("Upload New PDF")
        uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
        
        if uploaded_file is not None:
            if st.button("Process PDF"):
                with st.spinner("Processing PDF..."):
                    # Save uploaded file
                    file_path = os.path.join(DOCS_FOLDER, uploaded_file.name)
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    # Process PDF
                    chunks = load_and_chunk_pdf(file_path)
                    
                    # Add to index
                    st.session_state.index, st.session_state.texts = add_text_to_index(
                        chunks, st.session_state.index, st.session_state.texts, st.session_state.model
                    )
                    
                    st.success(f"Successfully processed {uploaded_file.name}!")
                    st.info(f"Added {len(chunks)} chunks to the index.")
                    st.rerun()

        # Clear Index Section
        st.markdown("---")
        st.subheader("🗑️ Clear Index")
        st.warning("This will delete all indexed data and reset the system!")
        
        if st.button("Clear All Index Data", type="secondary"):
            with st.spinner("Clearing index and data..."):
                deleted_files = clear_index_and_data()
                
                if deleted_files:
                    st.success(f"Successfully cleared index! Deleted: {', '.join(deleted_files)}")
                else:
                    st.info("No index files found to delete.")
                
                # Reinitialize the index
                st.session_state.index, st.session_state.texts, st.session_state.model = load_or_create_index()
                st.rerun()

    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("💬 Ask a Question")
        
        if not st.session_state.texts:
            st.warning("Please upload and process a PDF document first!")
            return
        
        question = st.text_area("Enter your question:", height=100, placeholder="What would you like to know about your documents?")
        
        if st.button("Ask Question", type="primary"):
            if question.strip():
                with st.spinner("Generating answer..."):
                    if st.session_state.llm is None:
                        st.error("LLM not available. Please check your GEMINI_API_KEY.")
                        return
                    
                    result = rag_answer(
                        question, 
                        st.session_state.index, 
                        st.session_state.texts, 
                        st.session_state.model, 
                        st.session_state.llm
                    )
                    
                    # Store result in session state for display
                    st.session_state.last_result = result
                    st.session_state.last_question = question
            else:
                st.warning("Please enter a question!")

    with col2:
        st.header("📖 Answer & Source")
        
        if hasattr(st.session_state, 'last_result') and st.session_state.last_result:
            result = st.session_state.last_result
            
            # Display the answer
            st.subheader("Answer:")
            st.markdown(result.get("answer", "No answer available"))
            
            # Display source information
            if result.get("filename") and result.get("page_no"):
                # Display page image if available
                pdf_path = os.path.join(DOCS_FOLDER, result['filename'])
                if os.path.exists(pdf_path):
                    st.subheader("Source Page:")
                    page_image = get_page_image(pdf_path, result['page_no'])
                    if page_image:
                        st.image(page_image, caption=f"Page {result['page_no']} from {result['filename']}", use_container_width=True)
                    else:
                        st.error("Could not extract page image")
                else:
                    st.error("Source PDF file not found")
        else:
            st.info("Ask a question to see the answer and source page here!")

    # Footer with stats
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Documents", len(uploaded_docs))
    with col2:
        st.metric("Total Chunks", len(st.session_state.texts))
    with col3:
        if hasattr(st.session_state, 'last_question'):
            st.metric("Last Query", "✓")
        else:
            st.metric("Last Query", "None")

if __name__ == "__main__":
    main()