<div align="center">

# 🏥 El-Mostawsaf (المستوصَف)

### AI-Assisted Healthcare Platform

A graduation project that combines a patient mobile app, a doctor dashboard, backend services, and AI-powered medical features: a RAG-based medical assistant, lab report assistance, and medical image analysis.

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat&logo=fastapi&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-Agents-1C3C3C?style=flat)
![RAG](https://img.shields.io/badge/RAG-Medical%20Knowledge-purple)
![Computer Vision](https://img.shields.io/badge/Computer%20Vision-Medical%20Images-orange)
![Flutter](https://img.shields.io/badge/Flutter-Mobile-02569B?style=flat&logo=flutter&logoColor=white)
![React](https://img.shields.io/badge/React-Dashboard-61DAFB?style=flat&logo=react&logoColor=black)
![Supabase](https://img.shields.io/badge/Supabase-PostgreSQL-3ECF8E?style=flat&logo=supabase&logoColor=white)

[🌐 Live Web App](https://el-mostawsaf-frontend.vercel.app/) · [🤖 AI Chatbot Demo](https://abdoabdotarek7523--el-mostawsaf-run-app.modal.run) · [💻 Source Code](https://github.com/AbdullrahmanTarek75/ELMOSTAWSF_RAG)

</div>

---

## 📑 Table of Contents

- [Overview](#-overview)
- [Live Links](#-live-links)
- [Key Features](#-key-features)
- [System Architecture](#-system-architecture)
- [AI & Machine Learning](#-ai--machine-learning)
- [Agentic Medical RAG Chatbot](#-agentic-medical-rag-chatbot)
- [Technology Stack](#-technology-stack)
- [Getting Started](#-getting-started)
- [Medical Safety](#-medical-safety)
- [Team & Contributions](#-team--contributions)
- [Limitations](#-limitations)
- [Roadmap](#-roadmap)

---

## 📖 Overview

**El-Mostawsaf** is a healthcare platform designed to make medical information and healthcare services easier to access. It brings together:

- 📱 a **patient mobile application**,
- 🩺 a **doctor dashboard**,
- ⚙️ **backend services and APIs**, and
- 🧠 **AI-assisted features** for medical questions, laboratory reports, and medical images.

> **Disclaimer:** The AI features support health education and preliminary review only. They do **not** replace a qualified healthcare professional and do **not** provide a definitive diagnosis.

---

## 🔗 Live Links

| Resource | Link |
|---|---|
| 🌐 Web Application (Frontend) | [el-mostawsaf-frontend.vercel.app](https://el-mostawsaf-frontend.vercel.app/) |
| 🤖 AI Chatbot Demo | [El-Mostawsaf Chatbot](https://abdoabdotarek7523--el-mostawsaf-run-app.modal.run) |
| 💻 Chatbot Source Repository | [ELMOSTAWSF_RAG](https://github.com/AbdullrahmanTarek75/ELMOSTAWSF_RAG) |

---

## ✨ Key Features

| Feature | Description |
|---|---|
| 🤖 **AI Medical Assistant** | Answers healthcare questions using a retrieval-augmented generation (RAG) workflow in Arabic and English. |
| 🔍 **Medical Knowledge Retrieval** | Retrieves relevant passages from a medical knowledge base before generating a response. |
| 🧪 **Lab Report Assistance** | Extracts and explains test names, values, units, and reference ranges from uploaded lab reports. |
| 🩻 **Medical Image Analysis** | Experimental support for X-ray and skin/burn image scenarios. |
| 📱 **Patient App** | Access to medical information, records, appointments, and project services. |
| 🩺 **Doctor Dashboard** | Lets healthcare professionals review patient information and related records. |
| ⌚ **Health Monitoring** | Wearable-related readings and alert workflows (ESP32, BLE, health sensors). |

---

## 🏗 System Architecture

```text
   Patient Mobile App          Doctor Dashboard
        (Flutter)             (React / TypeScript)
              \                     /
               \                   /
                v                 v
                  Backend APIs (FastAPI)
                          |
          +---------------+----------------+
          |               |                |
          v               v                v
      Database        AI Services      Health Data
  (PostgreSQL /           |            (ESP32 / BLE)
     Supabase)            |
                 +--------+---------+
                 |                  |
                 v                  v
          RAG / Retrieval     Medical Image
         + Language Model       Analysis
```

> This is a conceptual overview. Deployment details depend on the current project configuration.

### Project Structure

```text
El-Mostawsaf/
├── Mobile Application
├── Doctor Dashboard
├── Backend APIs
├── AI Services
│   ├── Medical Question Answering / RAG
│   ├── Laboratory Report Assistance
│   └── Medical Image Analysis
├── Database and Storage
├── Wearable / IoT Integration
└── Documentation
```

---

## 🧠 AI & Machine Learning

### 1. Medical Question Answering with RAG

1. Receive the user's healthcare question.
2. Retrieve relevant information from the medical knowledge base.
3. Rank and combine the retrieved context.
4. Pass the context to a language model to generate a response.
5. Present the answer as general health information with safety limitations.

**Retrieval components**

| Component | Technology |
|---|---|
| Embedding model | `intfloat/multilingual-e5-small` |
| Vector search | FAISS |
| Keyword search | BM25 |
| Hybrid retrieval | Weighted combination of semantic and keyword search |
| Language models | Gemini and other endpoints (active model depends on deployment configuration) |

### 2. Medical Knowledge Base

The system uses Arabic healthcare Q&A material and a separate collection of laboratory-related documents. The preparation pipeline covers text processing, chunking, embedding, and indexing. The exact size and composition of the knowledge base may differ between versions.

### 3. Medical Image Experiments

- X-ray image understanding
- Burn and skin-image classification or description
- Image-based preliminary observations

Available tasks depend on the configured model and endpoint. These are experimental decision-support aids, not validated diagnostic systems.

### 4. Laboratory Report Assistance

Helps users understand uploaded reports by identifying visible test names and values and explaining them in simple language. Image quality, OCR errors, units, and reference ranges can affect output, so values should always be checked against the original document and interpreted by a healthcare professional.

### 5. Evaluation Guidelines

Quality is assessed separately for retrieval, generated answers, image tasks, latency, and safety. Any published score should state:

- the dataset and evaluation split,
- the metric definition,
- the model and prompt version,
- the number of evaluated examples, and
- known failure cases and limitations.

---

## 🤖 Agentic Medical RAG Chatbot

The El-Mostawsaf chatbot is a dedicated component that combines RAG with tool-based workflows for medical questions and selected healthcare tasks. Available tools depend on the deployed version and environment configuration.

### Capabilities

- **Medical Q&A:** answers grounded in retrieved context.
- **Hybrid retrieval:** FAISS semantic search + BM25 keyword search.
- **Lab report assistance:** explains test names, values, units, and reference ranges.
- **Medical image assistance:** preliminary observations when a vision model is configured.
- **Consultation booking workflow:** collects information to arrange a consultation (requires a connected scheduling backend).
- **Web search:** optional Tavily-based search.
- **Date/time handling:** supports appointment validation.
- **Bilingual:** Arabic and English.

> **Note:** "Agentic" means the chatbot can select and use configured tools as part of a workflow. It does not mean every answer is verified or that the system can independently diagnose disease.

### Agent Tools

| Tool | Purpose |
|---|---|
| `company_knowledge_tool` | Retrieves information from the FAQ and knowledge files using hybrid RAG. |
| `Tavily_Search_Tool` | Searches the web (requires a Tavily API key). |
| `get_current_datetime_tool` | Provides current date and time for date-sensitive workflows. |
| `book_consultation_tool` | Supports the consultation-booking workflow. |
| `analyze_lab_report` | Interprets laboratory report content and reference ranges. |
| `analyze_medical_image` | Provides preliminary image observations via a supported model. |

### Chatbot Source Layout

```text
El-Mostawsaf-Chatbot/
├── app.py
├── src/
│   ├── agent.py
│   ├── config.py
│   ├── data_loaders.py
│   ├── retriever.py
│   ├── text_processor.py
│   ├── tools.py
│   ├── utils.py
│   └── vector_store.py
├── data/
│   ├── raw_company_info/
│   │   ├── FAQ.csv
│   │   └── info.md
│   └── processed/
├── assets/
├── requirements.txt
├── Dockerfile
└── README.md
```

### Chatbot Tech Stack

| Layer | Technology |
|---|---|
| Chat interface | Gradio |
| LLM integration | Groq via OpenAI-compatible API |
| Embeddings | `intfloat/multilingual-e5-small` |
| Vector retrieval | FAISS |
| Keyword retrieval | BM25 (`rank-bm25`) |
| Agent framework | LangChain tool-calling agent |
| PDF extraction | `pdfplumber` |
| Web search (optional) | Tavily |

---

## 🛠 Technology Stack

| Area | Technologies |
|---|---|
| Patient mobile application | Flutter, Dart |
| Doctor dashboard / web frontend | React, TypeScript (deployed on Vercel) |
| Backend APIs | Python, FastAPI |
| Language & retrieval AI | NLP, RAG, embeddings, FAISS, BM25, LangChain |
| Medical image experiments | MedGemma / configured vision models |
| Database & cloud | PostgreSQL, Supabase, Modal |
| Connected health devices | ESP32, BLE, health sensors |

---

## 🚀 Getting Started

### Run the Chatbot Locally

**1. Clone the repository**

```bash
git clone https://github.com/AbdullrahmanTarek75/ELMOSTAWSF_RAG.git
cd ELMOSTAWSF_RAG
```

**2. Create and activate a virtual environment**

```bash
python -m venv venv
```

Linux / macOS:

```bash
source venv/bin/activate
```

Windows (PowerShell):

```powershell
.\venv\Scripts\Activate.ps1
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

**4. Configure environment variables**

Create a local `.env` file:

```dotenv
GROQ_API_KEY=your_groq_api_key
TAVILY_API_KEY=your_tavily_api_key
HUGGINGFACE_HUB_TOKEN=your_huggingface_token
```

> Confirm the exact variable names in `src/config.py`. Never commit `.env` or real credentials; provide a sanitized `.env.example` with variable names only.

**5. Add knowledge files**

Place the expected `FAQ.csv` and `info.md` in `data/raw_company_info/`. Check `src/data_loaders.py` for required column names and paths.

**6. Start the app**

```bash
python app.py
```

By default Gradio serves at `http://localhost:7860`; check the terminal for the actual address.

### Docker

A `Dockerfile` is included. Verify the image name, exposed port, and required environment variables before use, and never bake API keys into the image.

### Customization

| Goal | Where to look |
|---|---|
| Add or update tools | `src/tools.py`, `src/agent.py` |
| Change the model | `src/config.py` |
| Tune hybrid retrieval | `src/retriever.py` |
| Update the knowledge base | Edit source files and rebuild the index |

---

## ⚕️ Medical Safety

- AI answers may be incomplete, outdated, or incorrect, and are not a diagnosis or treatment plan.
- Verify lab values, units, and reference ranges against the original report with a qualified clinician.
- Medical image analysis is preliminary and must not delay professional evaluation.
- In Egypt, users with a possible emergency should contact emergency services (**123**). The chatbot is designed to direct such users accordingly, but this guidance is not a substitute for calling emergency services.
- The assistant may be configured to restrict responses to healthcare-related topics.

---

## 👥 Team & Contributions

El-Mostawsaf was developed as a graduation project at the **Faculty of Engineering at Shoubra, Benha University**, by a student team. The project spans mobile development, web development, backend services, AI, database work, and IoT.

**AI work documented in this repository includes:**

- the medical question-answering and RAG pipeline,
- hybrid retrieval using embeddings and keyword search,
- preparation and indexing of medical text sources,
- laboratory report assistance,
- integration and evaluation of language and vision models, and
- testing of AI responses and documentation of limitations.

---

## ⚠️ Limitations

- This is an academic project, not a certified medical device.
- AI-generated answers can contain errors or omit important context.
- OCR and image-analysis results depend on input quality and model limitations.
- No clinical-performance claim is made without a documented validation study.
- Real patient data must never appear in screenshots, datasets, logs, or examples.

---

## 🗺 Roadmap

- [ ] Reproducible setup and deployment instructions for all components
- [ ] Published AI evaluation protocol with representative test cases
- [ ] Improved Arabic and Egyptian-Arabic question handling
- [ ] Citations and source references for retrieved medical answers
- [ ] Expanded testing for retrieval quality, hallucinations, and unsafe recommendations
- [ ] Better monitoring, error handling, and model-version tracking

---

## 📄 License & Usage

This repository describes an academic healthcare software project. Confirm licensing and obtain permission from the team before redistributing source code, datasets, screenshots, or other project materials.

---

<div align="center">

**El-Mostawsaf — AI-assisted healthcare, designed with safety and accessibility in mind.**

[🌐 Live App](https://el-mostawsaf-frontend.vercel.app/) · [🤖 Chatbot Demo](https://abdoabdotarek7523--el-mostawsaf-run-app.modal.run) · [💻 GitHub](https://github.com/AbdullrahmanTarek75/ELMOSTAWSF_RAG)

</div>
