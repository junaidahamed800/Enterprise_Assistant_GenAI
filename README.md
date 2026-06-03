# Enterprise Assistant - RAG-Based Knowledge Assistant

## Overview

Enterprise Assistant is a Retrieval-Augmented Generation (RAG) application designed to answer user questions based on enterprise documents and knowledge sources.

Instead of relying only on a Large Language Model's pre-trained knowledge, the application retrieves relevant information from a private knowledge base and provides accurate, context-aware responses.

This project demonstrates the complete RAG pipeline including document ingestion, chunking, embedding generation, vector storage, semantic retrieval, and response generation.

---

## Problem Statement

Organizations store large amounts of information in:

- Policies
- Technical documentation
- SOPs
- Knowledge base articles
- Internal manuals
- Training documents

Finding the correct information manually can be time-consuming.

Traditional keyword search often returns irrelevant results because it searches for exact words instead of understanding meaning.

This project solves that problem by:

1. Understanding the semantic meaning of user queries.
2. Retrieving the most relevant information from enterprise documents.
3. Providing contextual answers based on retrieved knowledge.

---

## Objectives

- Build an end-to-end RAG application.
- Learn how modern AI assistants retrieve information.
- Understand vector databases and semantic search.
- Implement document ingestion and chunking.
- Generate embeddings for enterprise knowledge.
- Retrieve relevant context using similarity search.
- Integrate retrieved context with an LLM.

---

## Architecture

```text
Documents
    │
    ▼
Document Loader
    │
    ▼
Chunking
    │
    ▼
Embeddings
    │
    ▼
Vector Database (ChromaDB)
    │
    ▼
User Query
    │
    ▼
Query Embedding
    │
    ▼
Similarity Search
    │
    ▼
Relevant Chunks
    │
    ▼
Prompt Augmentation
    │
    ▼
LLM
    │
    ▼
Answer
```

---

## Features

- Document ingestion
- Text chunking
- Embedding generation
- Vector database storage
- Semantic similarity search
- Context retrieval
- Retrieval-Augmented Generation (RAG)
- Extensible architecture for enterprise use cases

---

## Technologies Used

### Programming Language

- Python 3.12

### AI / Generative AI

- Retrieval-Augmented Generation (RAG)
- Embeddings
- Semantic Search
- Vector Databases

### Frameworks & Libraries

- LangChain
- LangChain Community
- Sentence Transformers
- ChromaDB

### Embedding Model

- all-MiniLM-L6-v2

### Vector Database

- ChromaDB

### Development Tools

- Visual Studio Code
- Git
- GitHub

---

## Skills Demonstrated

### Generative AI

- RAG Architecture
- Embeddings
- Semantic Search
- Prompt Augmentation
- Context Retrieval

### Python Development

- Virtual Environments
- Dependency Management
- File Handling
- Modular Project Structure

### AI Engineering

- Document Processing
- Chunking Strategies
- Vector Similarity Search
- Knowledge Retrieval Systems

---

## Project Structure

```text
Enterprise_Assistant/
│
├── data/
│   └── company_policy.txt
│
├── src/
│
├── vectorstore/
│
├── app.py
│
├── requirements.txt
│
└── README.md
```

---

## Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd Enterprise_Assistant
```

### 2. Create Virtual Environment

```bash
python -m venv venv
```

If virtual environment creation hangs during pip installation:

```bash
python -m venv venv --without-pip
python -m ensurepip
```

### 3. Activate Virtual Environment

#### Windows

```bash
venv\Scripts\Activate.ps1
```

#### Linux / Mac

```bash
source venv/bin/activate
```

### 4. Upgrade pip

```bash
python -m pip install --upgrade pip
```

### 5. Install Dependencies

```bash
python -m pip install langchain
python -m pip install langchain-community
python -m pip install chromadb
python -m pip install sentence-transformers
```

---

## Future Enhancements

- PDF document support
- Multiple document ingestion
- Metadata filtering
- Hybrid search
- Re-ranking
- OpenAI integration
- Ollama integration
- LangGraph workflows
- Conversational memory
- Agentic AI capabilities

---

## Learning Outcomes

Through this project, you will gain hands-on experience with:

- Retrieval-Augmented Generation (RAG)
- Embedding Models
- Vector Databases
- Semantic Search
- Enterprise Knowledge Assistants
- LangChain
- ChromaDB
- Generative AI Application Development

---

## Author

Junaid Shaik

Trainee Software Engineer | Generative AI Enthusiast

Skills:
- Python
- RAG
- LangChain
- LangGraph
- Vector Databases
- Prompt Engineering
- Java
- SQL
