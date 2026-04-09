# import libraries
import streamlit as st
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import PyPDF2

# STEP 1: extract text from PDF
def extract_text_from_pdf(uploaded_file):
    text = ""
    
    # create PDF reader object
    pdf_reader = PyPDF2.PdfReader(uploaded_file)
    
    # loop through all pages
    for page in pdf_reader.pages:
        text += page.extract_text()
    
    return text

# STEP 2: split text into chunks
def split_text(text):
    sentences = text.split(".")  # simple splitting
    sentences = [s.strip() for s in sentences if s.strip() != ""]
    return sentences

# STEP 3: create vectors
def create_vectors(chunks):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(chunks)
    return vectorizer, vectors

# STEP 4: find best match
def get_best_match(query, vectorizer, vectors, chunks):
    query_vector = vectorizer.transform([query])
    similarity = cosine_similarity(query_vector, vectors)
    best_index = np.argmax(similarity)
    return chunks[best_index]

# STEP 5: generate answer
def generate_answer(context):
    return f"Answer: {context}"

# STREAMLIT UI
st.title(" Simple RAG App (Beginner Project)")

# upload PDF
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file is not None:
    st.success("PDF uploaded successfully!")

    # extract text
    text = extract_text_from_pdf(uploaded_file)
    
    # split into chunks
    chunks = split_text(text)
    
    # create vectors
    vectorizer, vectors = create_vectors(chunks)
    
    # user query
    query = st.text_input("Ask a question from the document:")
    
    if query:
        # get relevant context
        context = get_best_match(query, vectorizer, vectors, chunks)
        
        # generate answer
        answer = generate_answer(context)
        
        # display result
        st.subheader(" Answer")
        st.write(answer)