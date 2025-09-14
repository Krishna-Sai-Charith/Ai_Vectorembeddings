# 🧠 Visualizing Vector Embeddings using LangChain, Chroma, and Plotly

This project shows how to transform markdown documents into vector embeddings using OpenAI (or HuggingFace) models, store them in a Chroma vector database, and visualize them using t-SNE in 2D and 3D with Plotly. It’s designed to help you understand how Retrieval-Augmented Generation (RAG) works under the hood — by combining document embeddings, vector search, and language models. With LangChain, we split and process documents, generate semantic vectors, and store them in a retrievable format. This allows LLMs to generate accurate, grounded responses based on real document content, not just pretraining. The visualizations reveal how similar documents group together, providing an intuitive understanding of AI-based semantic search.

---

## 🔍 What You’ll Learn

- How vector embeddings represent text meaning 🔢  
- How to split and preprocess documents for AI ✂️  
- How Chroma stores and retrieves semantic data 💾  
- Visualizing embeddings with t-SNE and Plotly 📈  
- How RAG pipelines combine search + generation 🔁  
- Understanding LLMs: auto-regressive vs auto-encoding 🧠  


## 🧭 Program Workflow


flowchart TD
    A[📂 Load Markdown Docs] --> B[✂️ Split into Chunks]
    B --> C[🔢 Generate Embeddings]
    C --> D[💾 Store in Chroma DB]
    D --> E[📉 Reduce Dimensions with t-SNE]
    E --> F[📊 Visualize in 2D and 3D]


## 🧠 Key Concepts

### 📌 Vector Embeddings  
Embeddings are numerical vectors that capture the **semantic meaning** of text.  
Similar texts have similar vectors, allowing us to perform **semantic search**, not just keyword search.

### 📚 LangChain  
LangChain connects language models to tools like document loaders, vector databases, and APIs.  
In this project, it helps load, chunk, embed, and manage documents for building a retrieval-augmented system.

### 💾 Chroma Vector DB  
Chroma is a database built specifically for storing and searching **high-dimensional vector embeddings**.  
Unlike traditional SQL/NoSQL databases, it supports **fast similarity search**, critical for finding relevant chunks in RAG.

---

## 🔁 What is RAG (Retrieval-Augmented Generation)?

RAG is a method where an AI model retrieves relevant knowledge from a custom database and uses it to generate more accurate, grounded answers.


flowchart LR
    Q[User Query] --> R[🔍 Retrieve from Chroma]
    R --> A[🤖 Answer via LLM using Retrieved Docs]

Why it's useful:
- Keeps LLM responses **factual and up-to-date**
- Allows LLMs to answer questions based on your own documents
- Bridges the gap between static LLM knowledge and dynamic enterprise data


## 🧠 Auto-regressive vs Auto-encoding LLMs

| Type              | Purpose                              | Example Models   |
|-------------------|---------------------------------------|------------------|
| 🔁 Auto-regressive | Predict next token in a sequence      | GPT, LLaMA       |
| 🧩 Auto-encoding   | Understand input by reconstructing it | BERT, RoBERTa    |

- **Auto-regressive** models (like GPT) are great for generating natural language responses.  
- **Auto-encoding** models (like BERT) are better at understanding and embedding text for semantic tasks.

Both are used in a RAG pipeline:
- Auto-encoding models for **embedding and retrieval**
- Auto-regressive models for **response generation**

---

## 📈 Why t-SNE for Visualization?

- Embeddings are high-dimensional (often 768+ dimensions).
- t-SNE helps reduce them to **2D or 3D** while keeping similar documents close together.
- Makes it easier to **see clusters, patterns, and relationships** between documents.
- Adds transparency and helps in debugging embedding quality visually.

---

## 🌟 Significance of This Project

This is more than just a data visualization task — it's a mini end-to-end RAG pipeline:
- Load → Chunk → Embed → Store → Visualize  
- Gives hands-on understanding of how AI “understands” your content  
- Shows how enterprise AI tools like ChatGPT plugins, Notion AI, and Microsoft Copilot retrieve and reason over private documents

You'll walk away understanding:
- How modern LLM-based systems interact with real data  
- How embeddings power intelligent document search  
- Why databases like Chroma are essential in AI search  
- The visual intuition behind "semantic similarity"
