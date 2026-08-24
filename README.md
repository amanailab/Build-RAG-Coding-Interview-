# RAG Coding Interview: Build RAG From Scratch With LangChain

> **AmanAI Lab**

---

## Step 1 — Create the Environment

First, show your terminal and create a project folder.

**Windows**

```bash
mkdir rag-langchain-interview
cd rag-langchain-interview
```

Then create a virtual environment:

```bash
python -m venv venv
```

Activate it:

```bash
venv\Scripts\activate
```

You should now see something like:

```
(venv) C:\...\rag-langchain-interview>
```

---

## Step 2 — Open the Project in VS Code

```bash
code .
```

Create this structure:

```
rag-langchain-interview/
│
├── venv/
├── data/
├── .env
├── .gitignore
└── rag.py
```

---

## Step 3 — Create `.gitignore`

Add:

```
venv/
.env
__pycache__/
*.pyc
```

**AmanAI Lab — Explain:**

> "I'm adding `.env` to `.gitignore` because it will contain my API key, and API keys should never be pushed to GitHub."

---

## Step 4 — Install the Libraries

Now, while the virtual environment is activated:

```bash
pip install -U langchain langchain-groq langchain-community langchain-huggingface langchain-text-splitters faiss-cpu pypdf python-dotenv sentence-transformers
```

Then verify:

```bash
pip list
```

For the video, don't spend too much time explaining every package. Just explain:

| Package | Purpose |
|---|---|
| **LangChain** | RAG orchestration |
| **langchain-groq** | Groq LLM integration |
| **langchain-community** | PDF loader / FAISS |
| **langchain-huggingface** | Hugging Face embeddings |
| **langchain-text-splitters** | Chunking |
| **FAISS** | Vector search |
| **pypdf** | Read PDF |
| **python-dotenv** | Environment variables |
| **sentence-transformers** | Local embeddings |

---

## Step 5 — Test the Environment

Before building RAG, create `rag.py`:

```python
print("RAG environment is ready!")
```

Run:

```bash
python rag.py
```

Expected:

```
RAG environment is ready!
```

---

*All following steps — loading the PDF, chunking, embeddings, vector store, retriever, prompt, chain, and querying — are implemented directly inside `rag.py`. See the file for the full, working implementation.*
