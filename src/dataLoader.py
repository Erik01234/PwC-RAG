import PyPDF2
from sentence_transformers import SentenceTransformer
import numpy as np
import torch
import faiss
import json


##ADJUSTABLE PARAMETERS##
filename = "1706.03762v7.pdf"
#In this case, it can be replaced with any PDF
embeddingModel = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")
#Load the embedding model
chunkS = 500
#Preferred chunk size
overl = 50
#Preferred overlap



#Function that turns PDF into text using the PyPDF2 library
def turnToText(path):
    read = PyPDF2.PdfReader(path)
    text = ""
    for page in read.pages:
        #Extract page by page
        text += page.extract_text() + "\n"
    return text


#Function that splits the extracted text into chunks
def splitter(text):
    chunkSize = chunkS
    overlap = overl
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunkSize - overlap):
        chunk = " ".join(words[i:i + chunkSize])
        chunks.append(chunk)
    return chunks


chunks = [] 
#To ensure it will not be ran (causing high latency) during import
if __name__ == "__main__": 
    pdfText = turnToText(filename)
    chunks = splitter(pdfText)
    #Create chunks of text of a given file

    chunk_vectors = []
    for chunk in chunks:
        #Vectorize chunks using the embedding model
        vector = embeddingModel.encode(chunk, convert_to_numpy=True)
        chunk_vectors.append(vector)
    chunk_vectors = np.array(chunk_vectors).astype("float32") 
    #FAISS needs float32

    with open("faiss_index/chunks.json", "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2) 
        #Save chunks as texts too

    dimensions = chunk_vectors.shape[1]
    index = faiss.IndexFlatL2(dimensions)
    #Grab vector size (dimensions) and index with it
    index.add(chunk_vectors)

    faiss.write_index(index, "faiss_index/chunk_index.faiss")
    print("FAISS saved")

    #Write FAISS to disk
