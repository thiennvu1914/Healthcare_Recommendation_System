# 🏥 Healthcare Recommendation System

Intelligent healthcare consultation system using RAG (Retrieval-Augmented Generation) technology with AI.

## 📊 Overview

- **88,590** medical articles
- **60,234** answered questions
- **15** medical specialties
- Technologies: **PhoBERT** + **Vistral-7B-Chat** + **FAISS**
- GPU: Optimized for **NVIDIA H200** (supports CPU fallback)

## 🏗️ System Architecture

`
Healthcare_Recommendation_System/
├── api/                    # FastAPI Backend
│   ├── main.py            # API endpoints
│   ├── rag_engine.py      # RAG core logic
│   ├── models.py          # Pydantic schemas
│   └── config.py          # Settings
├── web/                   # Django Web Frontend
│   ├── chatbot/          # Main app
│   ├── templates/        # HTML templates
│   └── manage.py
├── data/                 # Datasets
│   ├── articles.csv      # 88,590 articles
│   └── QAs.csv          # 60,234 Q&As
├── cache/               # FAISS indices cache
├── scripts/             # Utility scripts
│   ├── rebuild_fast.py  # Rebuild FAISS with GPU
│   └── test_api.py      # API testing
└── requirements.txt
`

## ✨ Key Features

### 🤖 API Backend (FastAPI)
- ✅ Semantic search with **PhoBERT embeddings**
- ✅ FAISS index with **HNSW** (fast retrieval)
- ✅ **LLM generation** with Vistral-7B-Chat
- ✅ Index caching - startup in **5 seconds** instead of 8 minutes
- ✅ GPU/CPU auto-detection
- ✅ CORS enabled for web frontend
- ✅ Auto documentation at /docs

### 🌐 Web Interface (Django)
- ✅ Modern chat UI with **Bootstrap 5**
- ✅ Displays **4 sections**: Specialty, Answer, Sources, Disclaimer
- ✅ Source citations with **badges** (Q&A/Article)
- ✅ Reference details modal
- ✅ Direct links to original articles
- ✅ Permanent disclaimer outside chat
- ✅ Responsive design

## 🚀 Installation & Startup

### 1. Clone repository

`ash
git clone <repository-url>
cd Healthcare_Recommendation_System
`

### 2. Install dependencies

`ash
# Backend API
pip install -r requirements.txt

# Web frontend
cd web
pip install -r requirements.txt
cd ..
`

### 3. Environment configuration

`ash
cp .env.example .env
# Edit .env as needed
`

**Recommended .env settings:**
`env
# Cache for fast startup (5s instead of 8 mins)
ENABLE_CACHE=1

# LLM generation with Vistral-7B
ENABLE_LLM_GENERATION=1

# Sample size (0 = use all data)
SAMPLE_SIZE=0

# HuggingFace token (if using gated models)
HUGGINGFACE_HUB_TOKEN=your_token_here
`

### 4. Start API Backend

`ash
# Terminal 1: FastAPI (port 8000)
uvicorn api.main:app --host 0.0.0.0 --port 8000

# First run: ~8 mins (build + cache FAISS indices)
# Subsequent runs: ~5 seconds (load from cache) 🚀
`

API will run at: **http://localhost:8000**
- Docs: http://localhost:8000/docs
- Health: http://localhost:8000/api/health

### 5. Start Web Frontend

`ash
# Terminal 2: Django (port 8080)
cd web
python manage.py migrate
python manage.py runserver 0.0.0.0:8080
`

Web UI will run at: **http://localhost:8080**

## 📖 Usage

### Chat with AI

1. Visit http://localhost:8080/ai-advisor/
2. Enter a health-related question
3. Receive answers from AI including:
   - Suggested specialty
   - Natural answer (LLM-generated)
   - Top reference sources (Q&A + Articles)
   - Click [Citation] to view details

### API Endpoints

`ash
# Chat endpoint
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"query":"What should I do if my 3-year-old has a 39-degree fever?","include_sources":true}'

# Health check
curl http://localhost:8000/api/health
`

**Response format:**
`json
{
  "answer": "When your 3-year-old has a 39-degree fever, the first step is to...",
  "specialty": "Pediatrics",
  "confidence": 0.869,
  "sources": [
    {
      "type": "qa",
      "id": "qa_12345",
      "question": "...",
      "full_answer": "...",
      "score": 0.92
    },
    {
      "type": "article",
      "id": "https://...",
      "title": "...",
      "link": "https://...",
      "score": 0.85
    }
  ],
  "disclaimer": "Information is for reference only..."
}
`

## 🛠️ Useful Scripts

### Rebuild FAISS Index (GPU-optimized)

`ash
python scripts/rebuild_fast.py
`

- Uses GPU for acceleration
- Batch processing with fp16
- Rebuilds both QA + Article indices
- Saves cache to cache/

### Test API

`ash
python scripts/test_api.py
`

## 🔧 Advanced Configuration

### API Settings (pi/config.py)

`python
# Retrieval
TOP_K_QA = 5              # Number of Q&As returned
TOP_K_ARTICLES = 3        # Number of Articles returned
QUESTION_SIM_THRESHOLD = 0.3   # Similarity threshold

# LLM Generation
MAX_NEW_TOKENS = 384      # Answer length
TEMPERATURE = 0.8         # Creativity
TOP_P = 0.92             # Nucleus sampling

# Performance
ENABLE_CACHE = 1          # Cache FAISS indices
SAMPLE_SIZE = 0           # 0 = use all data
`

### Django Settings (web/healthcare_web/settings.py)

`python
# API endpoint
API_ENDPOINT = "http://localhost:8000"

# Database
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': BASE_DIR / 'db.sqlite3',
    }
}
`

## 📚 Technical Details

### RAG Pipeline

1. **Query Processing**: Question preprocessing (lowercase, remove noise)
2. **Embedding**: PhoBERT embedding with mean-pooling
3. **Retrieval**: FAISS IndexFlatIP (cosine similarity)
   - Top-5 Q&As
   - Top-3 Articles
4. **Re-ranking**: Combined semantic + lexical overlap
5. **Generation**: Vistral-7B-Chat synthesizes natural answers
6. **Response**: JSON with specialty, answer, sources, disclaimer

### Models

- **PhoBERT** (inai/phobert-base): Vietnamese BERT for embeddings
- **Vistral-7B-Chat** (Viet-Mistral/Vistral-7B-Chat): Vietnamese LLM for generation
- **FAISS**: IndexFlatIP for cosine similarity search

### Performance

- **Startup**: 5s with cache (vs 8 mins without cache)
- **Query latency**: ~2-3s (including LLM generation)
- **VRAM usage**: ~4GB (PhoBERT + Vistral fp16)
- **Index size**: 
  - QA: 185MB (60,234 vectors)
  - Articles: 272MB (88,590 vectors)

## 🐛 Troubleshooting

### API fails to start

`ash
# Check logs
tail -f api.log

# Try disabling cache if indices are corrupted
ENABLE_CACHE=0 uvicorn api.main:app --host 0.0.0.0 --port 8000

# Rebuild indices
python scripts/rebuild_fast.py
`

### Web cannot connect to API

`ash
# Check API health
curl http://localhost:8000/api/health

# Check CORS settings in api/config.py
CORS_ORIGINS = ["http://localhost:8080", "http://127.0.0.1:8080"]
`

### Out of memory (GPU)

`ash
# Switch to CPU mode
FORCE_CPU=1 uvicorn api.main:app --host 0.0.0.0 --port 8000

# Or reduce sample size
SAMPLE_SIZE=5000 uvicorn api.main:app --host 0.0.0.0 --port 8000
`

### Unnatural answers

`ash
# Enable LLM generation
ENABLE_LLM_GENERATION=1 uvicorn api.main:app --host 0.0.0.0 --port 8000

# Increase temperature for more diverse answers
# Edit in api/config.py: TEMPERATURE = 0.8
`

## 📝 Changelog

### v2.0 (Latest)
- ✅ Integrated Vistral-7B-Chat for natural answers
- ✅ 4-section UI: Specialty, Answer, Sources, Disclaimer
- ✅ Source citations with badges + details modal
- ✅ Direct links to original articles
- ✅ Permanent disclaimer outside chat
- ✅ Full answer displayed for Q&A
- ✅ GPU-optimized rebuild script
- ✅ Code cleanup + folder restructure

### v1.0
- ✅ FastAPI backend with PhoBERT
- ✅ FAISS caching
- ✅ Django web interface
- ✅ Basic chat functionality

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License.

## ⚠️ Disclaimer

**Important Note:**
- Information from the AI is for reference only, based on medical databases.
- You should consult directly with a doctor for a more accurate diagnosis.
- This system DOES NOT replace the professional advice of a medical specialist.
- In case of an emergency, call **115** or go to the hospital immediately.

## 📧 Contact

For questions or support, please open an issue on GitHub.

---

**Created by DS300 - Group 12** | Technology: PhoBERT + Vistral-7B-Chat
