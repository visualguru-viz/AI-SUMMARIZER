from PyPDF2 import PdfReader
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import chromadb
from chromadb.config import Settings
from urllib.request import Request, urlopen

import streamlit as st


# Load embedding model
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Load summarization model
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Initialize Chroma DB client
chroma_client = chromadb.Client(Settings(anonymized_telemetry=False))
collection = chroma_client.get_or_create_collection(name="text_collection")


###------- Extractors --------####

def extract_pdf_text(pdf_file):
    pdf_reader = PdfReader(pdf_file)
    return " ".join([page.extract_text() for page in pdf_reader.pages if page.extract_text()])

def extract_text_from_web(url):
    req = Request(
        url=url,
        headers={'User-Agent': 'Mozilla/5.0'}
    )
    webpage = urlopen(req).read()    
    soup = BeautifulSoup(webpage, features="html.parser")

    # kill all script and style elements
    for script in soup(["script", "style"]):
        script.extract()    # rip it out

    # get text
    text = soup.get_text()
    

    # break into lines and remove leading and trailing space on each
    lines = (line.strip() for line in text.splitlines())
    # break multi-headlines into a line each
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    # drop blank lines
    text = '\n'.join(chunk for chunk in chunks if chunk)
    
    return text

def extract_text_from_raw(raw_text):
    return raw_text.strip()

# -------- Store in Chroma DB --------

def store_text_in_chroma(text, doc_id):
    chunks = [text[i:i+500] for i in range(0, len(text), 500)]
    embeddings = embedder.encode(chunks).tolist()
    ids = [f"{doc_id}_{i}" for i in range(len(chunks))]
    collection.add(documents=chunks, embeddings=embeddings, ids=ids)


# -------- Summarization --------

def summarize_text(text):
    # HuggingFace models usually have 1024 token limit
    return summarizer(text, max_length=150, min_length=40, do_sample=False)[0]['summary_text']




# -------- Pipeline Usage Example --------

if __name__ == "__main__":

    # --- Streamlit UI ---
    st.title("📄 Text Summarizer App")
    option = st.radio("Choose input type:", ["PDF File", "Web URL", "Raw Text"])

    input_text = ""

    if option == "PDF File":
        uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
        if uploaded_file:
            input_text = extract_pdf_text(uploaded_file)
            # st.text_area("Extracted Text", input_text, height=200)

    elif option == "Web URL":
        url = st.text_input("Enter a webpage URL")
        if url:
            input_text = extract_text_from_web(url)
            # st.text_area("Extracted Web Text", input_text, height=200)

    elif option == "Raw Text":
        input_text = st.text_area("Enter your text here", height=200)

    if input_text:
        if st.button("Generate Summary"):
            with st.spinner("Summarizing..."):
                # summary = summarize_text(input_text)
                summary = summarize_text(input_text[:2048]) 
                st.success("Summary:")
                st.write(summary)

    # Store the text in Chroma DB
    if st.button("Store in ChromaDB"):
        if input_text:
            doc_id = st.text_input("Enter Document ID", value="doc_1")
            store_text_in_chroma(input_text, doc_id)
            st.success(f"Text stored in ChromaDB with ID: {doc_id}")
        else:
            st.error("Please provide text to store.")



# # -------- Example Usage --------
# if __name__ == "__main__":
#     # Example URL and text inputs
#     # Replace with your actual URL and text inputs
#     # Note: The following lines are for demonstration purposes and should be replaced with actual inputs.
#     # You can use the Streamlit UI above to input these values interactively.
#     # Initialize Chroma DB
#     chroma_client = chromadb.Client(Settings(anonymized_telemetry=False))
#     collection = chroma_client.get_or_create_collection(name="text_collection")
#     # Initialize models
#     embedder = SentenceTransformer("all-MiniLM-L6-v2")
#     summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
#     # Example usage of the functions
#     # Note: Replace the following lines with actual inputs or use the Streamlit UI above.
#     # Initialize Chroma DB
#     # Initialize models
#     # Initialize Chroma DB



#     # Example input
#     url = "https://example.com"  # Replace with your URL
#     input_text = "This is some raw text input for summarization."  # Replace with your raw text
#     uploaded_file = "path/to/your/file.pdf"  # Replace with your PDF file path
#     # Example usage
#     uploaded_file = ""
#     pdf_text = extract_pdf_text(uploaded_file)  # Replace with your PDF file path
#     html_text = extract_text_from_web(url)
#     raw_text = extract_text_from_raw(input_text)

#     all_text = pdf_text + "\n" + html_text + "\n" + raw_text
#     store_text_in_chroma(all_text, doc_id="combined_doc")
    
#     summary = summarize_text(all_text[:2048])  # Truncate to fit model input
#     print("Summary:\n", summary)






# import streamlit as st
# from PyPDF2 import PdfReader
# import requests
# from bs4 import BeautifulSoup
# from transformers import pipeline
# import chromadb
# from chromadb.config import Settings
# import uuid

# # ------------------- Initialize Chroma DB -------------------
# chroma_client = chromadb.Client(Settings(anonymized_telemetry=False))
# collection_name = "text_collection"
# if collection_name not in [c.name for c in chroma_client.list_collections()]:
#     collection = chroma_client.create_collection(name=collection_name)
# else:
#     collection = chroma_client.get_collection(name=collection_name)

# # ------------------- Hugging Face Summarizer -------------------
# summarizer = pipeline("summarization")

# # ------------------- Helper Functions -------------------
# def extract_pdf_text(pdf_file):
#     pdf_reader = PdfReader(pdf_file)
#     return " ".join([page.extract_text() for page in pdf_reader.pages if page.extract_text()])

# def extract_url_text(url):
#     response = requests.get(url)
#     soup = BeautifulSoup(response.text, 'html.parser')
#     paragraphs = soup.find_all('p')
#     return " ".join([para.get_text() for para in paragraphs])

# def store_in_chroma(text, source="unknown"):
#     doc_id = str(uuid.uuid4())
#     collection.add(
#         documents=[text],
#         metadatas=[{"source": source}],
#         ids=[doc_id]
#     )
#     return doc_id

# def get_all_documents_from_chroma():
#     all_docs = collection.get()
#     return all_docs['documents']

# def summarize_texts(texts):
#     combined_text = " ".join(texts)
#     # Huggingface models have a token limit (~1024 for many)
#     max_chunk_size = 1000
#     summaries = []
#     for i in range(0, len(combined_text), max_chunk_size):
#         chunk = combined_text[i:i+max_chunk_size]
#         summary = summarizer(chunk, max_length=150, min_length=30, do_sample=False)
#         summaries.append(summary[0]['summary_text'])
#     return " ".join(summaries)

# # ------------------- Streamlit UI -------------------
# st.title("📄 Text Summarizer with ChromaDB")

# input_method = st.radio("Select Input Method", ("Upload PDF", "Enter URL", "Enter Raw Text"))

# text = ""
# source = ""

# if input_method == "Upload PDF":
#     uploaded_pdf = st.file_uploader("Upload PDF File", type="pdf")
#     if uploaded_pdf:
#         text = extract_pdf_text(uploaded_pdf)
#         source = uploaded_pdf.name
#         st.text_area("Extracted Text", text, height=200)

# elif input_method == "Enter URL":
#     url = st.text_input("Enter Web URL")
#     if url:
#         text = extract_url_text(url)
#         source = url
#         st.text_area("Extracted Text", text, height=200)

# elif input_method == "Enter Raw Text":
#     raw_text = st.text_area("Paste Raw Text", height=200)
#     if raw_text:
#         text = raw_text
#         source = "raw_text"

# # Store & Summarize
# if text:
#     if st.button("Store in Chroma & Summarize All"):
#         store_in_chroma(text, source)
#         all_texts = get_all_documents_from_chroma()
#         summary = summarize_texts(all_texts)
#         st.subheader("📝 Summary from All Stored Texts:")
#         st.write(summary)

# # Optional: Clear database
# if st.button("Clear ChromaDB"):
#     chroma_client.delete_collection(name=collection_name)
#     st.success("ChromaDB Collection Cleared!")
