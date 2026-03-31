# 🏥 V-MedRAG: Vietnamese Healthcare Recommendation System

An intelligent, automated healthcare recommendation and consultation system built with Retrieval-Augmented Generation (RAG) and Vietnamese Large Language Models (LLMs). Designed to mitigate LLM hallucinations, this system grounds its answers in a highly curated, localized medical database.

**🎥 Video Demo:** [Watch the demo](https://drive.google.com/file/d/1HNeb-MCdKrV1TUTJ_UP5LpUCnQlSgYt_/view?usp=drive_link)

## 📊 Knowledge Base Overview

The system's "memory" is built on a comprehensive, localized Vietnamese medical dataset:
- **Medical Articles (77,917 items):** Collected from authoritative Vietnamese healthcare portals to ensure epidemiological alignment, including Long Châu (41,301), Vinmec (30,905), Tâm Anh Hospital (5,105), and Medlatec (606).
- **Q&A Dataset (60,234 items):** Real-world patient-clinician question-answer pairs covering major specialties like Obstetrics & Gynecology (26,458), Pediatrics, Respiratory, and Dermatology.
- **Data Processing:** All data underwent rigorous PII (Personally Identifiable Information) removal and word segmentation using the Underthesea NLP toolkit.

## 🏗️ System Architecture & RAG Pipeline

```text
Healthcare_Recommendation_System/
└── Source/
    ├── api/                 # FastAPI Backend (RAG Core)
    │   ├── main.py          
    │   ├── rag_engine.py    # PhoBERT + FAISS + Vistral integration
    │   ├── models.py        
    │   └── config.py        # System hyperparams (Thresholds, k-values)
    ├── web/                 # Django Web Frontend (Bootstrap 5)
    │   └── chatbot/         
    ├── data/                # Processed chunks & embeddings
    ├── cache/               # HNSW FAISS indices cache
    ├── scripts/             # GPU-optimized index rebuild scripts
    └── requirements.txt
```

### Core Technical Workflow

1. **Input Processing:** Transforms unstructured, natural-language symptom descriptions into normalized text.
2. **Embedding & Retrieval:** Uses PhoBERT (vinai/phobert-base) to generate 768-dimensional semantic embeddings. Retrieves relevant passages using FAISS with HNSW (Hierarchical Navigable Small World) indexing for optimal low-latency ANN (Approximate Nearest Neighbor) search.
3. **Intermediate Processing & Re-ranking:**
   - **Action Extraction:** Prioritizes passages containing actionable verbs/imperatives to ensure practical advice.
   - **Hybrid Re-ranking:** Combines semantic cosine similarity with lexical overlap to maintain both intent matching and medical terminological precision.
4. **Generation:** Injects the refined top-k context into Vistral-7B-Chat (a 7-billion-parameter model optimized for Vietnamese) to synthesize the final response.

## ✨ Key Features & Optimal Configurations

- **Dual-Target Output:** Automatically predicts the appropriate medical specialty (Triage) and provides reference health advice (First-aid, lifestyle).
- **Hallucination Control:** Generation is strictly constrained by a low LLM Temperature (0.2) and max token limit (512) to prioritize factual consistency over creative text.
- **Optimal Chunking Strategy:** Text is split into 256-token chunks with a 50-token overlap to preserve medical discourse flow across sentence boundaries.
- **Fast Startup:** Auto-detects GPU/CPU and caches indices, reducing deployment loading times significantly.

## 📈 Experimental Performance

Evaluations on the test set reveal strong system reliability under the optimal similarity threshold of 0.65 and k=5:
- **Semantic Recall:** Reaches ~90% (87.2% at k=5, 89.6% at k=10), successfully surfacing relevant medical knowledge.
- **Action Found Rate:** >94.4%, validating the effectiveness of the Action Extraction module for delivering practical recommendations.
- **Specialty Accuracy:** Peaks at 52.0% for correct medical routing.

## 🚀 Quick Start

### 1. Installation

```bash
git clone <repository-url>
cd Healthcare_Recommendation_System/Source

# Install dependencies for both Backend and Frontend
pip install -r requirements.txt
cd web && pip install -r requirements.txt && cd ..

# Setup environment variables
cp .env.example .env
```
*(Tip: Set `ENABLE_CACHE=1` and `ENABLE_LLM_GENERATION=1` in your `.env` file).*

### 2. Run the System

**Terminal 1: FastAPI Backend (RAG Engine)**
```bash
cd Source
uvicorn api.main:app --host 0.0.0.0 --port 8000
```
*API Docs available at http://localhost:8000/docs*

**Terminal 2: Django Frontend**
```bash
cd Source/web
python manage.py migrate
python manage.py runserver 0.0.0.0:8080
```
*Access the Web UI at http://localhost:8080/ai-advisor/*

## 📖 API Usage Example

```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"query":"Bé 3 tuổi nhà tôi bị sốt 39 độ thì nên làm gì?","include_sources":true}'
```

## ⚠️ Disclaimer

This system is strictly an informational support tool intended for triage and reference. It does not provide definitive medical diagnoses and does not replace the role of licensed clinicians. All outputs contain a mandatory disclaimer advising users to consult healthcare professionals. In emergencies, please contact local medical facilities immediately.

