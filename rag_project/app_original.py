import fitz  # PyMuPDF

def load_and_chunk_pdf(pdf_path: str):
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


from sentence_transformers import SentenceTransformer
import faiss
import json
import os

model = SentenceTransformer("all-MiniLM-L6-v2")
index_file = "my_index.faiss"
texts_file = "texts.json"

# Load or create index
if os.path.exists(index_file):
    index = faiss.read_index(index_file)
else:
    index = faiss.IndexFlatIP(384)  # 384 for MiniLM-L6-v2

# Load or init texts
if os.path.exists(texts_file):
    with open(texts_file, "r") as f:
        texts = json.load(f)
else:
    texts = []

def add_text(new_items: list[dict]):
    global texts, index

    # Extract texts to embed
    raw_texts = [item["text"] for item in new_items]
    embeddings = model.encode(raw_texts, normalize_embeddings=True).astype("float32")

    index.add(embeddings)
    texts.extend(new_items)
    
    faiss.write_index(index, index_file)
    with open(texts_file, "w") as f:
        json.dump(texts, f, ensure_ascii=False, indent=2)


def retrieve(query, top_k=5):
    query_emb = model.encode([query], normalize_embeddings=True).astype('float32')
    D, I = index.search(query_emb, top_k)
    return [texts[i] for i in I[0] if i != -1]


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
- Avoid saying you don’t know unless the question is entirely unrelated to the context.
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


from langchain_google_genai import ChatGoogleGenerativeAI
import os
from dotenv import load_dotenv

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    api_key=os.getenv("GEMINI_API_KEY")
)

import json
import json
import re

def json_to_obj(json_str: str) -> dict:
    cleaned = re.sub(r"^```(?:json)?\s*|```$", "", json_str.strip(), flags=re.IGNORECASE)
    
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        print("Failed to parse JSON:", e)
        return {}


def rag_answer(question):
    chunks = retrieve(question)
    context = json.dumps(chunks)
    prompt = prompt_template.replace("{context}", context).replace("{question}", question)
    response = llm.invoke(prompt)
    obj = json_to_obj(response.content)
    return obj




