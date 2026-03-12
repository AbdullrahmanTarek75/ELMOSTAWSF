# 🏥 El-Mostawsaf — Agentic Medical RAG Chatbot

An AI-powered medical assistant built for **El-Mostawsaf Healthcare Platform** (المستوصف للرعاية الصحية). Uses Retrieval-Augmented Generation (RAG) with agentic reasoning to answer medical questions, analyze lab reports, and book appointments — in both **Arabic and English**.

---

## ✨ Features

- 🤖 **Agentic RAG** — Multi-step reasoning with tool use for complex medical queries
- 🧪 **Lab Report Analysis** — Interprets blood test results with reference ranges and clinical notes
- 🩻 **Medical Image Analysis** — Preliminary assessment of burns, fractures, and wounds
- 📅 **Appointment Booking** — Collects patient info and books consultations automatically
- 🌐 **Bilingual** — Full Arabic (RTL) and English support
- 🔍 **Hybrid Retrieval** — Combines vector search (FAISS) and BM25 for accurate company knowledge lookup
- 🚨 **Emergency Protocol** — Detects critical symptoms and directs to emergency services (123) immediately

---

## 🗂️ Project Structure

```
El-Mostawsaf-Chatbot/
│
├── app.py                      # Gradio UI entry point
├── src/
│   ├── agent.py                # Agent logic, streaming, memory & error handling
│   ├── config.py               # LLM, embeddings & path configuration
│   ├── data_loaders.py         # FAQ, company info & uploaded file loaders
│   ├── retriever.py            # Hybrid retriever (FAISS + BM25)
│   ├── text_processor.py       # Text chunking strategies
│   ├── tools.py                # All agent tools (RAG, search, booking, lab, image)
│   ├── utils.py                # Vector store & knowledge base utilities
│   └── vector_store.py         # FAISS vector store management
│
├── data/
│   ├── raw_company_info/       # FAQ.csv + info.md
│   └── processed/              # Cached vector store & chunks
│
├── assets/                     # Screenshots and demo images
├── .env                        # API keys (not committed)
├── requirements.txt
├── Dockerfile
├── README.md
└── LICENSE
```

---

## ⚙️ Installation

### Prerequisites

- Python 3.10+
- [Groq API key](https://console.groq.com/) (free tier available)
- [Tavily API key](https://tavily.com/) for web search
- [HuggingFace token](https://huggingface.co/settings/tokens) for embeddings

### 1. Clone the repository

```bash
git clone https://github.com/
cd Agentic-RAG-Medical-Chatbot
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate       # Linux / macOS
venv\Scripts\activate          # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_groq_api_key
base_url=https://api.groq.com/openai/v1
TAVILY_API_KEY=your_tavily_api_key
HUGGINGFACE_HUB_TOKEN=your_hf_token
```

### 5. Add company knowledge

Place your data files in `data/raw_company_info/`:
- `FAQ.csv` — columns: `Question`, `Answer`
- `info.md` — general company information in Markdown

### 6. Run the app

```bash
python app.py
```

The Gradio interface will open at `http://localhost:7860`

---

## 🐳 Docker

**Build and run:**

```bash
docker build -t el-mostawsaf-chatbot .
docker run --name el-mostawsaf-chatbot --env-file .env -p 7860:7860 el-mostawsaf-chatbot
```

**Pull from Docker Hub:**

```bash
docker pull moazz/agentic-medical-rag-chatbot:latest
docker run --name el-mostawsaf-chatbot --env-file .env -p 7860:7860 moazz/agentic-medical-rag-chatbot:latest
```

---

## 🛠️ Available Agent Tools

| Tool | Description |
|------|-------------|
| `company_knowledge_tool` | Retrieves answers from company FAQ and info using hybrid RAG |
| `Tavily_Search_Tool` | Web search for up-to-date medical information |
| `get_current_datetime_tool` | Returns current date/time (used for appointment validation) |
| `book_consultation_tool` | Collects patient data and books a medical consultation |
| `analyze_lab_report` | Interprets blood test results against clinical reference ranges |
| `analyze_medical_image` | Provides preliminary assessment of burns, X-rays, and wounds |

---

## 🧩 Customization

- **Add tools** — Implement in `src/tools.py` and register in `AVAILABLE_TOOLS` in `src/agent.py`
- **Change LLM** — Update `model` in `src/config.py` (any Groq-supported model with tool calling)
- **Change retriever weights** — Adjust BM25/FAISS weights in `src/retriever.py`
- **Update knowledge base** — Edit `data/raw_company_info/FAQ.csv` or `info.md`, then delete `data/processed/` to rebuild

---

## 🧠 Tech Stack

| Layer | Technology |
|-------|-----------|
| LLM | Groq (`llama-3.3-70b-versatile`) via OpenAI-compatible API |
| Embeddings | `intfloat/multilingual-e5-small` (HuggingFace) |
| Vector Store | FAISS (CPU) |
| Sparse Retrieval | BM25 (`rank-bm25`) |
| Agent Framework | LangChain (`create_openai_tools_agent`) |
| UI | Gradio 5 |
| PDF Parsing | pdfplumber |

---

## 📋 Notes

- In case of **medical emergencies**, the assistant immediately directs users to call **123**
- Lab report analysis is for **educational purposes only** and does not replace professional medical advice
- The assistant strictly refuses non-medical topics (sports, cooking, entertainment, etc.)

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

## 🙏 Acknowledgements

- [FAISS](https://github.com/facebookresearch/faiss) — Vector similarity search
- [Groq](https://groq.com/) — Ultra-fast LLM inference
- [LangChain](https://github.com/hwchase17/langchain) — Agent framework
- [Gradio](https://gradio.app/) — UI framework

---

## 📬 Contact

For questions or support, please open an issue on GitHub.