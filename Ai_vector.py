# --------------------------------------------
# Imports
# --------------------------------------------

import os
import glob
from dotenv import load_dotenv
import gradio as gr

# LangChain, Chroma, and Plotly-related imports
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

import numpy as np
from sklearn.manifold import TSNE
import plotly.graph_objects as go


# --------------------------------------------
# Model and Database Setup
# --------------------------------------------

# Use a cost-effective model (gpt-4o-mini)
MODEL = "gpt-4o-mini"

# Set database directory name
db_name = "vector_db"


# --------------------------------------------
# Load Environment Variables
# --------------------------------------------

load_dotenv(override=True)
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY', 'your-key-if-not-using-env')

print("✅ Environment variables loaded.")


# --------------------------------------------
# Load Markdown Documents from Knowledge Base
# --------------------------------------------

# Get all subfolders inside "knowledge-base"
folders = glob.glob("knowledge-base/*")

# Set loader encoding - common fix for Windows and Unicode issues
text_loader_kwargs = {'encoding': 'utf-8'}

documents = []

print(f"📁 Found folders: {folders}")

# Loop through each folder and load .md documents
for folder in folders:
    doc_type = os.path.basename(folder)
    loader = DirectoryLoader(folder, glob="**/*.md", loader_cls=TextLoader, loader_kwargs=text_loader_kwargs)
    folder_docs = loader.load()
    for doc in folder_docs:
        doc.metadata["doc_type"] = doc_type
        documents.append(doc)

print(f"📄 Loaded {len(documents)} documents.")


# --------------------------------------------
# Split Documents into Chunks
# --------------------------------------------

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(documents)

print(f"✂️ Split into {len(chunks)} document chunks.")

doc_types = set(chunk.metadata['doc_type'] for chunk in chunks)
print(f"📦 Document types found: {', '.join(doc_types)}")


# --------------------------------------------
# Generate Embeddings
# --------------------------------------------

embeddings = OpenAIEmbeddings()

# If using HuggingFace instead (free option), use this:
# from langchain.embeddings import HuggingFaceEmbeddings
# embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


# --------------------------------------------
# Clear Existing Chroma Collection (If Any)
# --------------------------------------------

if os.path.exists(db_name):
    print(f"🧹 Clearing existing Chroma DB: {db_name}")
    Chroma(persist_directory=db_name, embedding_function=embeddings).delete_collection()
else:
    print("ℹ️ No existing DB found. Creating new one.")


# --------------------------------------------
# Create Chroma Vector Store
# --------------------------------------------

vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=db_name)

print(f"✅ Vector store created with {vectorstore._collection.count()} document chunks.")


# --------------------------------------------
# Inspect Sample Vector Dimensions
# --------------------------------------------

collection = vectorstore._collection
sample_embedding = collection.get(limit=1, include=["embeddings"])["embeddings"][0]
dimensions = len(sample_embedding)

print(f"🔢 Vector dimensions: {dimensions}")


# --------------------------------------------
# Retrieve All Vectors for Visualization
# --------------------------------------------

result = collection.get(include=['embeddings', 'documents', 'metadatas'])
vectors = np.array(result['embeddings'])
documents = result['documents']
doc_types = [metadata['doc_type'] for metadata in result['metadatas']]

# Assign colors based on doc_type (customize as needed)
color_map = {'products': 'blue', 'employees': 'green', 'contracts': 'red', 'company': 'orange'}
colors = [color_map.get(t, 'grey') for t in doc_types]


# --------------------------------------------
# 2D t-SNE Visualization
# --------------------------------------------

print("📉 Running 2D t-SNE...")
tsne_2d = TSNE(n_components=2, random_state=42)
reduced_vectors_2d = tsne_2d.fit_transform(vectors)

fig_2d = go.Figure(data=[go.Scatter(
    x=reduced_vectors_2d[:, 0],
    y=reduced_vectors_2d[:, 1],
    mode='markers',
    marker=dict(size=5, color=colors, opacity=0.8),
    text=[f"Type: {t}<br>Text: {d[:100]}..." for t, d in zip(doc_types, documents)],
    hoverinfo='text'
)])

fig_2d.update_layout(
    title='2D Chroma Vector Store Visualization',
    scene=dict(xaxis_title='x', yaxis_title='y'),
    width=800,
    height=600,
    margin=dict(r=20, b=10, l=10, t=40)
)

fig_2d.show()
print("✅ 2D t-SNE plot displayed.")


# --------------------------------------------
# 3D t-SNE Visualization
# --------------------------------------------

print("📈 Running 3D t-SNE...")
tsne_3d = TSNE(n_components=3, random_state=42)
reduced_vectors_3d = tsne_3d.fit_transform(vectors)

fig_3d = go.Figure(data=[go.Scatter3d(
    x=reduced_vectors_3d[:, 0],
    y=reduced_vectors_3d[:, 1],
    z=reduced_vectors_3d[:, 2],
    mode='markers',
    marker=dict(size=5, color=colors, opacity=0.8),
    text=[f"Type: {t}<br>Text: {d[:100]}..." for t, d in zip(doc_types, documents)],
    hoverinfo='text'
)])

fig_3d.update_layout(
    title='3D Chroma Vector Store Visualization',
    scene=dict(xaxis_title='x', yaxis_title='y', zaxis_title='z'),
    width=900,
    height=700,
    margin=dict(r=20, b=10, l=10, t=40)
)

fig_3d.show()
print("✅ 3D t-SNE plot displayed.")
