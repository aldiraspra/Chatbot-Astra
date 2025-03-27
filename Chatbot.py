import os
import io
import logging
import requests
import streamlit as st
import google.generativeai as genai
from dotenv import load_dotenv
from PIL import Image
from PyPDF2 import PdfReader

# Langchain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

# Load API
load_dotenv()
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_GEMINI_API_KEY = os.getenv("GOOGLE_GEMINI_API_KEY")
genai.configure(api_key=GOOGLE_GEMINI_API_KEY)

DOCS_FOLDER = "docs/"
INDEX_PATH = "faiss_index"

#UI
st.set_page_config(page_title="IsuzuBOT", page_icon="🚗")

with st.sidebar:
    st.title("Settings")
    select_model = st.selectbox("Select feature", ["IsuzuBOT", "Generate Gambar Produk"])

# --- IMAGE SEARCH  ---
def search_google_images(query):
    search_url = "https://www.googleapis.com/customsearch/v1"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    params = {
        "q": query,
        "cx": GOOGLE_CSE_ID,
        "key": GOOGLE_API_KEY,
        "searchType": "image",
        "num": 1
    }
    try:
        response = requests.get(search_url, params=params, headers=headers)
        response.raise_for_status()
        results = response.json()
        return results["items"][0]["link"] if "items" in results else None
    except requests.exceptions.RequestException as e:
        logging.error(f"Error fetching image: {e}")
        return None

# --- PDF PROCESSING  ---
def get_documents():
    documents = []
    for filename in os.listdir(DOCS_FOLDER):
        if filename.lower().endswith(".pdf"):
            filepath = os.path.join(DOCS_FOLDER, filename)
            try:
                pdf_reader = PdfReader(filepath)
                text = ""
                for page in pdf_reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n\n"
                if text:
                    documents.append({"text": text, "source": filename, "filepath": filepath})
            except Exception as e:
                logging.error(f"Error reading {filename}: {e}")
    return documents

#split to chunks
def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return splitter.split_text(text)

#vector store 
def create_vector_store(documents):
    texts = []
    metadatas = []
    for doc in documents:
        chunks = get_text_chunks(doc["text"])
        texts.extend(chunks)
        metadatas.extend([{"source": doc["source"], "filepath": doc["filepath"]}] * len(chunks))
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(texts, embedding=embeddings, metadatas=metadatas)
    vector_store.save_local(INDEX_PATH)
    return vector_store

@st.cache_resource
def load_faiss_index():
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    if os.path.exists(f"{INDEX_PATH}/index.faiss"):
        return FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
    else:
        with st.spinner("Processing documents..."):
            documents = get_documents()
            vector_store = create_vector_store(documents)
            return FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)

vector_store = load_faiss_index()

# system prompt
def get_conversational_chain():
    prompt_template = """
    Anda adalah asisten chatbot bernama 'IsuzuBOT' yang ahli dalam menjawab pertanyaan terkait produk-produk Isuzu.
    Gunakan informasi dari brosur yang telah diberikan untuk memberikan jawaban yang akurat dan terperinci.
    Jika informasi tidak ditemukan dalam data, beri tahu pengguna bahwa informasi tidak tersedia.
    
    - Jika pengguna bertanya tentang spesifikasi produk, berikan penjelasan yang lengkap dan detail.
    - Jika pengguna meminta perbandingan atau comparison antar produk, tampilkan informasi dalam bentuk tabel yang rapi, terstruktur, detail, mudah dipahami, dan sertakan summary di akhir.
    - Jangan memberikan jawaban di luar konteks dari data yang tersedia.
    
    Konteks:
    {context}
    
    Pertanyaan:
    {question}
    
    Jawaban (tidak dalam format markdown khusus):
    """
    
    model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", client=genai, temperature=0.9)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(llm=model, chain_type="stuff", prompt=prompt)

#get answers
def user_input(user_question):
    #similarity search
    docs = vector_store.similarity_search(user_question)
    if not docs:
        return "Sorry, I couldn't find the answer.", None
    #dokumen pertama sebagai referensi (score tertinggi)
    top_doc = docs[0]
    context = top_doc.page_content
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    answer_text = response.get("output_text", "Sorry, I couldn't find the answer.")
    # Ambil metadata dari dokumen
    source = top_doc.metadata.get("source", "Unknown")
    filepath = top_doc.metadata.get("filepath", None)
    return answer_text, (source, filepath)

# --- STREAMLIT UI ---
st.title("IsuzuBOT 🚗")

if select_model == "Generate Gambar Produk":
    st.subheader("Ketik produk isuzu yang ingin dicari 🏎️")
    image_prompt = st.text_input("Tipe Produk:")
    if st.button("Search 🚀"):
        with st.spinner("Searching..."):
            image_url = search_google_images(image_prompt)
            if image_url:
                st.image(image_url, caption="", use_container_width=True)
            else:
                st.error("No image found.")

elif select_model == "IsuzuBOT":
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Tanyakan apapun mengenai Isuzu D-Max, MuX, Giga FVZ, Traga, dan Elf NLR!"}]
    #menyimpan referensi dokumen dari jawaban
    if "last_reference" not in st.session_state:
        st.session_state.last_reference = None

    #chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    if prompt := st.chat_input("Ask about Isuzu products..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer_text, source_info = user_input(prompt)
                st.write(answer_text)
                # Simpan referensi dokumen ke session state
                st.session_state.last_reference = source_info
        
        st.session_state.messages.append({"role": "assistant", "content": answer_text})
    
    if st.session_state.last_reference:
        source, filepath = st.session_state.last_reference
        st.markdown(f"Sources:")
        if filepath and os.path.exists(filepath):
            with open(filepath, "rb") as f:
                file_bytes = f.read()
            st.download_button(
                label=f"Download {source}",
                data=file_bytes,
                file_name=source,
                mime="application/pdf"
            )
