"""Core RAG Engine - Healthcare Consultation System"""
import os
import re
import numpy as np
import pandas as pd
import torch
import pickle
import hashlib
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, pipeline
from underthesea import word_tokenize
import faiss
from html import unescape
import logging

logger = logging.getLogger(__name__)

from api.config import settings

class HealthcareRAGEngine:
    """Main RAG Engine for Healthcare Consultation"""
    
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() and settings.USE_GPU else "cpu"
        print(f"[RAG Engine] Initializing on device: {self.device}")
        
        if self.device == "cuda":
            print(f"[RAG Engine] GPU: {torch.cuda.get_device_name(0)}")
            print(f"[RAG Engine] VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            # Set default device to GPU
            torch.cuda.set_device(0)
        
        # Models
        self.tokenizer_phobert = None
        self.model_phobert = None
        self.gen_tokenizer = None
        self.gen_model = None
        self.gen_pipe = None
        
        # Indices
        self.qa_index = None
        self.article_index = None
        
        # Data
        self.df_qa = None
        self.df_articles = None
        self.article_texts = []
        
        # Action verbs for filtering
        self.action_verbs = self._load_action_verbs()
        
        # Text cleaning patterns
        self._init_text_patterns()
        
    def _init_text_patterns(self):
        """Initialize regex patterns for text cleaning"""
        self.re_at_prefix = re.compile(r'^@[^:]{0,60}:\s*', flags=re.IGNORECASE)
        self.name_pattern = re.compile(r'\b([A-ZÀ-Ỹ][a-zà-ỹ]+(?:_[A-ZÀ-Ỹ][a-zà-ỹ]+)+)\b')
        self.doctor_pattern = re.compile(r'\b(BS|Bác sĩ|Lương y|Dr)\.?\s+([A-ZÀ-Ỹ][a-zà-ỹ_]+(\s+[A-ZÀ-Ỹ][a-zà-ỹ_]+)*)', flags=re.IGNORECASE)
        self.pronoun_pattern = re.compile(r'\b(cháu|em|tớ|mình|con|anh|chị)\b', flags=re.IGNORECASE)
        
        self.connectives = [
            r'vì vậy', r'vì thế', r'vậy nên', r'do vậy', r'vì vậy nên', 
            r'vì thế nên', r'cho nên', r'tóm lại', r'tóm tắt', r'nhưng', r'tuy nhiên'
        ]
        self.connective_pattern = re.compile("|".join([re.escape(x) for x in self.connectives]), flags=re.IGNORECASE)
        
    def _load_action_verbs(self) -> set:
        """Load action verbs for filtering actionable sentences"""
        return set([
            # Nhóm dùng thuốc / điều trị
            "uống", "uống thuốc", "dùng", "dùng thuốc", "xịt", "bôi", "thoa", "nhỏ", "ngậm", 
            "tiêm", "chích", "truyền", "phẫu thuật", "mổ", "tiểu phẫu", "kê đơn", "điều trị",
            "chườm", "chườm nóng", "chườm lạnh", "băng bó", "sát trùng", "rửa vết thương",
            "hút rửa", "xông", "khí dung", "châm cứu", "bấm huyệt", "massage", "xoa bóp",
            
            # Nhóm khám / xét nghiệm
            "khám", "đi khám", "tái khám", "thăm khám", "kiểm tra", "xét nghiệm", "lấy mẫu",
            "siêu âm", "chụp", "chụp x-quang", "chụp ct", "chụp mri", "nội soi", "đo huyết áp",
            "đo đường huyết", "theo dõi", "đánh giá", "tầm soát",
            
            # Nhóm sinh hoạt / dinh dưỡng
            "ăn", "ăn kiêng", "kiêng", "tránh", "hạn chế", "bổ sung", "tăng cường", "giảm",
            "uống nước", "ngủ", "nghỉ ngơi", "kê gối", "nằm nghiêng", "tập", "tập luyện", 
            "vận động", "tập vật lý trị liệu", "thể dục", "vệ sinh", "súc miệng", "súc họng",
            "rửa tay", "rửa mũi", "đeo khẩu trang", "cách ly", "nhập viện", "cấp cứu",

            # Bổ sung từ dataset
            'đi', 'siêu', 'nội', 'đặt', 'nhỏ', 'bổ', 'khám và', 'tránh thai', 'khám bệnh', 
            'ăn uống', 'khám bác sĩ', 'khám sức', 'khám sức khỏe', 'đi ngoài', 'kê', 
            'đi siêu âm', 'khám thai', 'đặt lịch', 'khám lại', 'đặt lịch khám', 'tiêm chủng',
            'đi khám bác', 'khám phụ khoa', 'đi khám để', 'khám chuyên khoa', 'khám phụ',
            'đi khám và', 'khám chuyên', 'tiêm ngừa', 'đi lại', 'rửa', 'đi tiểu', 'kiêng',
            'tiêm mũi', 'khám trực', 'chụp x quang', 'tiêm vacxin', 'khám và điều',
            'đi xét nghiệm', 'đi khám thai', 'đi khám chuyên', 'tiêm phòng', 'chụp x',
            'cho bé đi', 'đi tái', 'đưa bé đi', 'đi xét', 'đi phân', 'khám tư',
            'đi cầu', 'đi ngoài phân', 'đi tái khám', 'đến bệnh viện', 'khám thai định',
            'khám với', 'tiêm vaccine', 'đi kiểm tra', 'đi tiêm', 'đi tiêu', 'đưa bé đến',
            'khám trực tiếp', 'đặt khám', 'đi khám ngay', 'tiêm vắc xin', 'đặt tư vấn',
            'khám em', 'đặt thuốc', 'khám với bác', 'tiêm vắc', 'khám để được', 'đi làm',
            'hút thai', 'siêu âm tim', 'đi khám tại', 'tiêm được', 'khám không', 'ăn và',
            'hút thuốc', 'làm gì', 'đi khám bệnh', 'đi kiểm', 'siêu âm lại', 'khám định',
            'đến khám tại', 'đi vệ', 'đi vệ sinh', 'tiêm thuốc', 'dùng biện pháp',
            'tiêm chủng chuyên', 'siêu âm thai', 'khám định kỳ', 'kê đơn thuốc', 'ăn được',
            'ăn không', 'liên hệ với', 'khám để bác', 'kê thuốc', 'uống đủ nước',
            'khám tại khoa', 'đặt vòng', 'khám bệnh chuyên', 'khám hiếm muộn', 'thai',
            'dùng biện', 'đặt câu', 'đặt câu hỏi', 'uống và', 'chụp mri', 'khám sớm',
            'khám cho', 'uống có', 'làm gì để', 'uống bổ sung', 'kê toa', 'siêu âm ở',
            'nội soi bóc', 'siêu âm thấy', 'khám tư vấn', 'khám hiếm', 'dùng cho',
            'đi kèm', 'ăn đủ', 'ăn của', 'khám tổng', 'khám tổng quát', 'khám và siêu'
        ])
    
    def load_models(self):
        """Load PhoBERT and Generation models"""
        print("[RAG Engine] Loading PhoBERT...")
        self.tokenizer_phobert = AutoTokenizer.from_pretrained(settings.RETRIEVAL_MODEL, use_fast=True)
        
        try:
            if self.device == "cuda":
                self.model_phobert = AutoModel.from_pretrained(
                    settings.RETRIEVAL_MODEL,
                    device_map="auto",
                    torch_dtype=torch.float16,
                    low_cpu_mem_usage=True
                )
            else:
                self.model_phobert = AutoModel.from_pretrained(settings.RETRIEVAL_MODEL)
            self.model_phobert.eval()
            print("[RAG Engine] PhoBERT loaded successfully")
        except Exception as e:
            print(f"[RAG Engine] Error loading PhoBERT: {e}")
            raise
        
        # Generation model (optional, load on demand)
        self.generation_model = None
        self.generation_tokenizer = None
        print("[RAG Engine] Generation model will be loaded on first use")
    
    def _load_generation_model(self):
        """Load Vistral-7B-Chat generation model"""
        if self.generation_model is not None:
            return  # Already loaded
        
        try:
            from transformers import AutoModelForCausalLM
            
            model_name = "Viet-Mistral/Vistral-7B-Chat"
            print(f"[RAG Engine] Loading generation model: {model_name}")

            hf_token = (
                (getattr(settings, "HUGGINGFACE_HUB_TOKEN", "") or "").strip()
                or (getattr(settings, "HUGGINGFACE_TOKEN", "") or "").strip()
                or os.getenv("HUGGINGFACE_HUB_TOKEN", "").strip()
                or os.getenv("HUGGINGFACE_TOKEN", "").strip()
                or None
            )
            
            # Prefer `token=...` (newer transformers). Do not pass both.
            tok_kwargs = {"use_fast": True}
            if hf_token:
                tok_kwargs.update({"token": hf_token})
            self.generation_tokenizer = AutoTokenizer.from_pretrained(model_name, **tok_kwargs)
            
            if self.device == "cuda" and torch.cuda.is_available():
                print(f"[RAG Engine] Loading on GPU: {torch.cuda.get_device_name(0)}")
                model_kwargs = {
                    "device_map": "auto",
                    "dtype": torch.float16,
                    "low_cpu_mem_usage": True,
                }
                if hf_token:
                    model_kwargs.update({"token": hf_token})
                self.generation_model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
            else:
                print("[RAG Engine] Loading on CPU")
                model_kwargs = {"dtype": torch.float32}
                if hf_token:
                    model_kwargs.update({"token": hf_token})
                self.generation_model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
                self.generation_model.to(self.device)
            
            self.generation_model.eval()
            print("[RAG Engine] Generation model loaded successfully")
        except Exception as e:
            print(f"[RAG Engine] Failed to load generation model: {e}")
            print("[RAG Engine] Will use template-based generation")
            self.generation_model = None
            self.generation_tokenizer = None
    
    def load_data(self):
        """Load Q&A and Articles data"""
        import sys
        print(f"[RAG Engine] Loading data from {settings.DATA_DIR}")
        sys.stdout.flush()

        # If FAISS cache exists, avoid expensive full-corpus preprocessing at startup.
        # This keeps API boot fast while still loading the full dataset for display/snippets.
        cache_present = (
            settings.ENABLE_CACHE
            and settings.QA_INDEX_CACHE.exists()
            and settings.METADATA_CACHE.exists()
        )
        preprocess_corpus = os.getenv("PREPROCESS_CORPUS", "0") == "1"
        
        # Load QA
        print("[RAG Engine] Reading QA CSV...")
        sys.stdout.flush()
        qa_usecols = None
        if settings.QA_CSV_PATH.exists():
            # Load only the columns we actually use.
            qa_usecols = ["question", "answer", "topic", "topic_original", "advice"]
        self.df_qa = pd.read_csv(settings.QA_CSV_PATH, usecols=lambda c: (qa_usecols is None or c in qa_usecols))
        # Sample if SAMPLE_SIZE > 0, otherwise use all data
        if settings.SAMPLE_SIZE > 0 and len(self.df_qa) > settings.SAMPLE_SIZE:
            self.df_qa = self.df_qa.sample(n=settings.SAMPLE_SIZE, random_state=42).reset_index(drop=True)
        print(f"[RAG Engine] Loaded {len(self.df_qa)} Q&A records")
        sys.stdout.flush()
        
        # Always normalize dtypes; optional preprocessing is VERY expensive for full datasets.
        for col in ["question", "answer", "advice", "topic", "topic_original"]:
            if col in self.df_qa.columns:
                self.df_qa[col] = self.df_qa[col].fillna("").astype(str)

        # Only preprocess if explicitly requested AND cache isn't present.
        if preprocess_corpus and not cache_present:
            for col in ["question", "answer", "advice"]:
                if col in self.df_qa.columns:
                    self.df_qa[col] = self.df_qa[col].apply(self._preprocess_text)
        
        # Load Articles
        if settings.ARTICLES_CSV_PATH.exists():
            article_usecols = ["id", "title", "text"]
            self.df_articles = pd.read_csv(settings.ARTICLES_CSV_PATH, usecols=lambda c: c in article_usecols)
            # Sample if SAMPLE_SIZE > 0, otherwise use all data
            if settings.SAMPLE_SIZE > 0 and len(self.df_articles) > settings.SAMPLE_SIZE:
                self.df_articles = self.df_articles.sample(n=settings.SAMPLE_SIZE, random_state=42).reset_index(drop=True)
            print(f"[RAG Engine] Loaded {len(self.df_articles)} articles")
            
            # Prepare article texts (use itertuples for speed)
            self.article_texts = []
            for row in self.df_articles.itertuples(index=False, name="ArticleRow"):
                title = str(getattr(row, "title", "")).strip()
                text = str(getattr(row, "text", "")).strip()
                self.article_texts.append((
                    str(getattr(row, "id", "")),
                    title,
                    f"{title}\n\n{text}"
                ))
        else:
            print("[RAG Engine] Articles file not found, skipping")
            self.df_articles = pd.DataFrame()
    
    def build_indices(self):
        """Build HNSW indices for fast retrieval with caching support"""
        import sys
        print("[RAG Engine] build_indices() started")
        sys.stdout.flush()
        
        # Try to load from cache first (fix issue #6: avoid rebuild every time)
        if settings.ENABLE_CACHE and self._try_load_indices():
            logger.info("[RAG Engine] Successfully loaded indices from cache")
            print("[RAG Engine] Successfully loaded indices from cache")
            sys.stdout.flush()
            return
        
        logger.info("[RAG Engine] Building HNSW indices from scratch...")
        print("[RAG Engine] Building HNSW indices from scratch...")
        sys.stdout.flush()
        
        # Build QA index
        qa_texts = (self.df_qa["question"].fillna("") + " " + self.df_qa["answer"].fillna("")).tolist()
        qa_embeddings = []
        
        print(f"[RAG Engine] Encoding {len(qa_texts)} Q&A pairs...")
        for i, text in enumerate(qa_texts):
            if i % 500 == 0:
                print(f"  Progress: {i}/{len(qa_texts)}")
            qa_embeddings.append(self._sentence_embedding(text))
        
        qa_embeddings = np.vstack(qa_embeddings).astype("float32")
        self.qa_index = self._build_hnsw_index(qa_embeddings)
        print(f"[RAG Engine] QA index built with {self.qa_index.ntotal} elements")
        
        # Build Article index
        article_embeddings = None
        if self.article_texts:
            print(f"[RAG Engine] Encoding {len(self.article_texts)} articles...")
            article_embeddings = []
            for i, (_, _, content) in enumerate(self.article_texts):
                if i % 500 == 0:
                    print(f"  Progress: {i}/{len(self.article_texts)}")
                article_embeddings.append(self._sentence_embedding(content))
            
            article_embeddings = np.vstack(article_embeddings).astype("float32")
            self.article_index = self._build_hnsw_index(article_embeddings)
            print(f"[RAG Engine] Article index built with {self.article_index.ntotal} elements")
        
        # Save to cache
        if settings.ENABLE_CACHE:
            self._save_indices(qa_embeddings, article_embeddings)
    
    def _save_indices(self, qa_embeddings: np.ndarray, article_embeddings: Optional[np.ndarray] = None):
        """Save HNSW indices and embeddings to cache"""
        try:
            settings.CACHE_DIR.mkdir(parents=True, exist_ok=True)
            
            # Save HNSW indices
            if self.qa_index:
                faiss.write_index(self.qa_index, str(settings.QA_INDEX_CACHE))
                logger.info(f"Saved QA index to {settings.QA_INDEX_CACHE}")
            
            if self.article_index:
                faiss.write_index(self.article_index, str(settings.ARTICLE_INDEX_CACHE))
                logger.info(f"Saved article index to {settings.ARTICLE_INDEX_CACHE}")
            
            # Save embeddings
            if qa_embeddings is not None:
                np.save(settings.QA_EMBEDDINGS_CACHE, qa_embeddings)
                logger.info(f"Saved QA embeddings to {settings.QA_EMBEDDINGS_CACHE}")
            
            if article_embeddings is not None:
                np.save(settings.ARTICLE_EMBEDDINGS_CACHE, article_embeddings)
                logger.info(f"Saved article embeddings to {settings.ARTICLE_EMBEDDINGS_CACHE}")
            
            # Save metadata (data hash for validation)
            metadata = {
                "qa_count": len(self.df_qa),
                "article_count": len(self.article_texts),
                "sample_size": settings.SAMPLE_SIZE,
                "data_hash": self._compute_data_hash()
            }
            with open(settings.METADATA_CACHE, "wb") as f:
                pickle.dump(metadata, f)
            
            logger.info("[RAG Engine] All indices saved to cache successfully")
        except Exception as e:
            logger.warning(f"Failed to save indices to cache: {e}")
    
    def _try_load_indices(self) -> bool:
        """Try to load indices from cache"""
        try:
            # Check if cache files exist
            if not all([
                settings.QA_INDEX_CACHE.exists(),
                settings.METADATA_CACHE.exists()
            ]):
                logger.info("Cache files not found, will build from scratch")
                return False
            
            # Load and validate metadata
            with open(settings.METADATA_CACHE, "rb") as f:
                metadata = pickle.load(f)
            
            # Validate data hasn't changed (older metadata may not include data_hash/sample_size)
            current_hash = self._compute_data_hash()
            meta_sample_size = metadata.get("sample_size")
            if meta_sample_size is not None and meta_sample_size != settings.SAMPLE_SIZE:
                logger.info("Sample size has changed, invalidating cache")
                return False

            meta_hash = metadata.get("data_hash")
            if meta_hash is not None and meta_hash != current_hash:
                logger.info("Data has changed, invalidating cache")
                return False

            # Backfill missing fields for older cache metadata
            if meta_hash is None:
                metadata["data_hash"] = current_hash
            if meta_sample_size is None:
                metadata["sample_size"] = settings.SAMPLE_SIZE
            try:
                with open(settings.METADATA_CACHE, "wb") as f:
                    pickle.dump(metadata, f)
            except Exception:
                pass
            
            # Load QA index
            self.qa_index = faiss.read_index(str(settings.QA_INDEX_CACHE))
            logger.info(f"Loaded QA index with {self.qa_index.ntotal} elements")
            
            # Load article index if exists
            if settings.ARTICLE_INDEX_CACHE.exists() and len(self.article_texts) > 0:
                self.article_index = faiss.read_index(str(settings.ARTICLE_INDEX_CACHE))
                logger.info(f"Loaded article index with {self.article_index.ntotal} elements")
            
            return True
            
        except Exception as e:
            logger.warning(f"Failed to load indices from cache: {e}")
            return False
    
    def _compute_data_hash(self) -> str:
        """Compute hash of data files to detect changes"""
        try:
            hash_obj = hashlib.md5()
            
            # Hash QA file
            if settings.QA_CSV_PATH.exists():
                hash_obj.update(str(settings.QA_CSV_PATH.stat().st_mtime).encode())
                hash_obj.update(str(settings.QA_CSV_PATH.stat().st_size).encode())
            
            # Hash articles file
            if settings.ARTICLES_CSV_PATH.exists():
                hash_obj.update(str(settings.ARTICLES_CSV_PATH.stat().st_mtime).encode())
                hash_obj.update(str(settings.ARTICLES_CSV_PATH.stat().st_size).encode())
            
            # Include sample size in hash
            hash_obj.update(str(settings.SAMPLE_SIZE).encode())
            
            return hash_obj.hexdigest()
        except Exception as e:
            logger.warning(f"Failed to compute data hash: {e}")
            return "unknown"
    
    def _preprocess_text(self, s: str) -> str:
        """Preprocess Vietnamese text"""
        if not isinstance(s, str):
            return ""
        s = re.sub(r'\s+', ' ', s.strip())
        # IMPORTANT: keep preprocessing consistent with how embeddings in cache were built.
        # Our full-corpus GPU rebuild embeds raw text without word segmentation.
        use_word_tokenize = os.getenv("USE_WORD_TOKENIZE", "0") == "1"
        if not use_word_tokenize:
            return s
        try:
            return word_tokenize(s, format="text")
        except Exception:
            return s
    
    def _preprocess_reference_sentence(self, s: str) -> str:
        """Preprocess reference sentence for embedding (remove names, pronouns, etc.)"""
        if not s:
            return ""
        
        s = s.strip()
        
        # Skip questions
        if s.endswith('?'):
            return ""
        
        # Remove @ prefix
        s = self.re_at_prefix.sub("", s)
        
        # Remove "trả lời"
        s = re.sub(r'^trả[_\s]lời\s*[:.]?\s*', '', s, flags=re.IGNORECASE)
        
        # Remove names with underscore
        s = self.name_pattern.sub("", s)
        
        # Remove doctor names
        s = self.doctor_pattern.sub("", s)
        
        # Replace pronouns with "bạn"
        s = self.pronoun_pattern.sub("bạn", s)
        
        # Remove connectives
        s = self.connective_pattern.sub("", s)
        
        # Clean whitespace
        s = re.sub(r'\s+', ' ', s).strip()
        
        return s
    
    def _clean_text(self, t: str) -> str:
        """Clean text - unescape HTML and normalize whitespace"""
        return re.sub(r'\s+', ' ', unescape(t.strip())).strip()
    
    def _sentence_has_action(self, s: str) -> bool:
        """Check if sentence contains action verbs"""
        if not s:
            return False
        
        sl = s.lower()
        for act in self.action_verbs:
            act_norm = act.replace("_", " ").lower()
            if re.search(r'\b' + re.escape(act_norm) + r'\b', sl):
                return True
        return False
    
    def _sentence_embedding(self, text: str) -> np.ndarray:
        """Generate sentence embedding using PhoBERT"""
        if not text:
            return np.zeros(768, dtype=np.float32)
        
        try:
            inputs = self.tokenizer_phobert(text, return_tensors="pt", truncation=True, max_length=256)
            model_device = next(self.model_phobert.parameters()).device
            inputs = {k: v.to(model_device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model_phobert(**inputs)

                # Mean pooling over last hidden state (sentence embedding)
                last_hidden = outputs.last_hidden_state
                attention_mask = inputs.get("attention_mask")

                if attention_mask is not None:
                    attention_mask = attention_mask.unsqueeze(-1)
                    masked = last_hidden * attention_mask
                    summed = masked.sum(dim=1)
                    counts = attention_mask.sum(dim=1).clamp(min=1e-9)
                    mean_pooled = (summed / counts).squeeze().cpu().numpy().astype("float32")
                else:
                    mean_pooled = last_hidden.mean(dim=1).squeeze().cpu().numpy().astype("float32")

                return mean_pooled
        except Exception as e:
            print(f"[Embedding Error] {e}")
            return np.zeros(768, dtype=np.float32)
    
    def _build_hnsw_index(self, vectors: np.ndarray) -> Optional[faiss.Index]:
        """Build HNSW index from vectors using faiss"""
        if vectors is None or vectors.size == 0:
            return None
        
        try:
            dim = vectors.shape[1]
            # Use IndexFlatIP for inner product (similar to cosine similarity)
            index = faiss.IndexFlatIP(dim)
            # Normalize vectors for cosine similarity
            faiss.normalize_L2(vectors)
            index.add(vectors)
            return index
        except Exception as e:
            print(f"[HNSW Error] {e}")
            return None
    
    def retrieve_qa(self, query: str, k: int = 5) -> List[Dict]:
        """Retrieve top-k Q&A pairs"""
        if self.qa_index is None:
            return []
        
        query_processed = self._preprocess_text(query)
        user_emb = self._sentence_embedding(query_processed).reshape(1, -1)
        
        # Normalize query embedding for cosine similarity
        faiss.normalize_L2(user_emb)
        
        raw_k = min(max(k * 10, 30), self.qa_index.ntotal)
        distances, labels = self.qa_index.search(user_emb, k=raw_k)

        q_tokens = set([t.lower() for t in re.findall(r'\w+', query) if len(t) >= 2])

        candidates = []
        for dist, idx in zip(distances[0], labels[0]):
            if idx < 0 or idx >= len(self.df_qa):
                continue

            row = self.df_qa.iloc[int(idx)]
            sim = float(dist)

            # lexical overlap helps avoid off-topic high-embedding matches
            qa_text = f"{row.get('question', '')} {row.get('answer', '')}".lower()
            qa_tokens = set([t.lower() for t in re.findall(r'\w+', qa_text) if len(t) >= 2])
            lex = 0.0
            if q_tokens:
                lex = float(len(q_tokens & qa_tokens)) / max(1, len(q_tokens))

            combined = 0.85 * sim + 0.15 * lex
            candidates.append((combined, sim, lex, int(idx), row))

        # Sort by combined score
        candidates.sort(key=lambda x: x[0], reverse=True)

        results = []
        for combined, sim, lex, idx, row in candidates:
            # Keep a lower gate because combined scoring already filters noise
            if sim < settings.QUESTION_SIM_THRESHOLD and combined < (settings.QUESTION_SIM_THRESHOLD + 0.05):
                continue

            results.append({
                "score": float(combined),
                "sim": float(sim),
                "lex": float(lex),
                "index": idx,
                "question": row.get("question", ""),
                "answer": row.get("answer", ""),
                "topic": row.get("topic", "Khác")
            })
            if len(results) >= k:
                break

        return results
    
    def retrieve_articles(self, query: str, k: int = 1) -> List[Dict]:
        """Retrieve top-k articles with re-ranking"""
        if self.article_index is None or not self.article_texts:
            return []
        
        query_processed = self._preprocess_text(query)
        user_emb = self._sentence_embedding(query_processed).reshape(1, -1)
        q_tokens_set = set([t.lower() for t in re.findall(r'\w+', query) if len(t) >= 2])
        
        # Normalize query embedding for cosine similarity
        faiss.normalize_L2(user_emb)
        
        raw_k = min(k * 3, self.article_index.ntotal)
        distances, labels = self.article_index.search(user_emb, k=raw_k)
        
        # Get raw candidates
        raw_candidates = []
        for dist, idx in zip(distances[0], labels[0]):
            if idx < 0 or idx >= len(self.article_texts):
                continue
            
            raw_candidates.append({
                "index": int(idx),
                # With normalized vectors, inner product == cosine similarity
                "score": float(dist)
            })
        
        # Re-rank with lexical overlap and title boost
        reranked = []
        w_sim = 0.75
        w_lex = 0.20
        w_title_boost = 0.05
        
        for c in raw_candidates:
            idx = int(c['index'])
            link, title, content = self.article_texts[idx]
            baseline_sim = float(c.get('score', 0.0))
            
            # Extract actual URL from content if present
            url_match = re.search(r'https?://[^\s]+', content)
            actual_link = url_match.group(0) if url_match else link
            
            # Lexical overlap with title + snippet
            article_snippet = title + " " + (content[:1000] if content else "")
            art_tokens = set([t.lower() for t in re.findall(r'\w+', article_snippet) if len(t) >= 2])
            
            lex_overlap = 0.0
            if q_tokens_set:
                common = len(q_tokens_set & art_tokens)
                lex_overlap = common / max(1, len(q_tokens_set))
            
            # Title boost
            title_tokens = set([t.lower() for t in re.findall(r'\w+', title) if len(t) >= 2])
            title_boost_flag = 1.0 if (q_tokens_set & title_tokens) else 0.0
            
            # Combined score
            combined = w_sim * baseline_sim + w_lex * lex_overlap + w_title_boost * title_boost_flag
            
            # Find best passage
            passages = self._chunk_text(content, max_chars=600)
            best_passage = ""
            best_passage_sim = -1.0
            
            for p in passages[:6]:
                p_proc = self._preprocess_text(p)
                p_emb = self._sentence_embedding(p_proc).astype("float32")
                sim_p = self._cosine_sim(user_emb.reshape(-1), p_emb)
                
                if sim_p > best_passage_sim:
                    best_passage_sim = sim_p
                    best_passage = p
            
            if not best_passage and content:
                best_passage = content[:600]
            
            reranked.append({
                "index": idx,
                "link": actual_link,
                "title": title,
                "txt": content,
                "baseline_sim": baseline_sim,
                "lex_overlap": lex_overlap,
                "title_boost": title_boost_flag,
                "combined_score": combined,
                "best_passage": best_passage,
                "best_passage_sim": best_passage_sim,
                "score": combined  # For compatibility
            })
        
        # Sort by combined score
        reranked_sorted = sorted(reranked, key=lambda x: x["combined_score"], reverse=True)
        
        # Filter by minimum score
        final = [r for r in reranked_sorted if r["combined_score"] >= settings.COMBINED_SCORE_THRESHOLD]
        
        if not final:
            return []
        
        # Return top-k
        results = []
        for r in final[:k]:
            results.append({
                "score": r["combined_score"],
                "index": r["index"],
                "title": r["title"],
                "link": r["link"],
                "snippet": r["best_passage"]
            })
        
        return results
    
    def _chunk_text(self, text: str, max_chars: int = 500, overlap_ratio: float = 0.15) -> List[str]:
        """Chunk text into overlapping passages"""
        if not text:
            return []
        
        passages = []
        start = 0
        L = len(text)
        overlap_chars = int(max_chars * overlap_ratio)
        
        while start < L:
            end = start + max_chars
            if end >= L:
                passages.append(text[start:L].strip())
                break
            
            # Find sentence boundary
            cut = text.rfind('.', start, end)
            if cut <= start:
                cut = end
            
            passages.append(text[start:cut].strip())
            
            # Move with overlap
            if overlap_chars > 0:
                start = max(cut - overlap_chars, start + 1)
            else:
                start = cut
        
        return [p for p in passages if p]
    
    def _cosine_sim(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors"""
        if a is None or b is None:
            return 0.0
        na = np.linalg.norm(a)
        nb = np.linalg.norm(b)
        if na == 0 or nb == 0:
            return 0.0
        return float(np.dot(a, b) / (na * nb))
    
    def find_best_action_sentence(
        self, 
        user_text: str,
        topk_rows: List[Dict],
        sent_sim_thresh: float = 0.6,
        combined_thresh: float = 0.68,
        alpha: float = 0.75,
        beta: float = 0.2,
        gamma: float = 0.05
    ) -> Tuple[Optional[str], Optional[int], float]:
        """
        Find best action sentence from top-k Q&A results
        
        Args:
            user_text: User query
            topk_rows: List of top-k Q&A results
            sent_sim_thresh: Sentence similarity threshold
            combined_thresh: Combined score threshold
            alpha: Weight for sentence similarity
            beta: Weight for question similarity
            gamma: Weight for lexical overlap
            
        Returns:
            (action_text, ref_index, score)
        """
        if not topk_rows:
            return None, None, 0.0
        
        # Extract and preprocess all sentences
        all_sents = []
        for ref_pos, r in enumerate(topk_rows, start=1):
            question_text = r.get("question", "") or ""
            raw_answer = r.get("answer", "") or ""
            
            # Split answer into sentences
            sents = re.split(r'(?<=[.!?])\s+', raw_answer.strip()) if raw_answer else []
            
            for s in sents:
                s_orig = self._clean_text(s)
                s_proc = self._preprocess_reference_sentence(s_orig)
                
                if len(s_proc) >= 6:
                    all_sents.append((ref_pos, question_text, s_orig, s_proc))
        
        if not all_sents:
            return None, None, 0.0
        
        # Get user embedding
        user_q = self._preprocess_text(user_text)
        user_emb = self._sentence_embedding(user_q).astype("float32")
        user_tokens_set = set([t.lower() for t in re.findall(r'\w+', user_text) if len(t) >= 2])
        
        # Cache question embeddings
        question_emb_cache = {}
        scored = []
        
        for ref_pos, question_text, sent_orig, sent_proc in all_sents:
            # Get question embedding
            if ref_pos not in question_emb_cache:
                q_text_proc = self._preprocess_text(question_text) if question_text else ""
                if q_text_proc:
                    question_emb_cache[ref_pos] = self._sentence_embedding(q_text_proc).astype("float32")
                else:
                    question_emb_cache[ref_pos] = np.zeros(user_emb.shape, dtype=np.float32)
            
            # Get sentence embedding
            s_emb = self._sentence_embedding(sent_proc).astype("float32")
            
            # Calculate similarities
            sim_sent = self._cosine_sim(user_emb, s_emb)
            sim_q = self._cosine_sim(user_emb, question_emb_cache[ref_pos])
            
            # Lexical overlap
            sent_tokens_set = set([t.lower() for t in re.findall(r'\w+', sent_proc) if len(t) >= 2])
            lex_overlap = float(len(user_tokens_set & sent_tokens_set)) / max(1, len(user_tokens_set)) if user_tokens_set else 0.0
            
            # Combined score
            combined = alpha * sim_sent + beta * sim_q + gamma * lex_overlap
            
            scored.append((combined, sim_sent, sim_q, lex_overlap, sent_orig, sent_proc, ref_pos))
        
        # Sort by combined score
        scored_sorted = sorted(scored, key=lambda x: x[0], reverse=True)
        
        # Find best action sentence
        best_ref_pos = None
        best_combined_score = 0.0
        
        for combined, sim_sent, sim_q, lex, sent_orig, sent_proc, ref_pos in scored_sorted:
            if sim_sent >= sent_sim_thresh and combined >= combined_thresh:
                if self._sentence_has_action(sent_proc) or self._sentence_has_action(sent_orig):
                    best_ref_pos = ref_pos
                    best_combined_score = combined
                    break
        
        if best_ref_pos is None:
            return None, None, 0.0
        
        # Collect all action sentences from best reference
        final_sentences_list = []
        for combined, sim_sent, sim_q, lex, sent_orig, sent_proc, ref_pos in scored_sorted:
            if ref_pos == best_ref_pos:
                if sim_sent >= sent_sim_thresh and combined >= combined_thresh:
                    if self._sentence_has_action(sent_proc) or self._sentence_has_action(sent_orig):
                        # Replace pronouns
                        s_final = self.pronoun_pattern.sub("bạn", sent_orig)
                        s_final = re.sub(r'\s+', ' ', s_final).strip()
                        
                        if s_final not in final_sentences_list:
                            final_sentences_list.append(s_final)
        
        if not final_sentences_list:
            return None, None, 0.0
        
        final_paragraph = " ".join(final_sentences_list)
        return final_paragraph, best_ref_pos, best_combined_score

    def _generate_generic_medical_advice(self, query: str, specialty: str) -> str:
        """Safe fallback when retrieval is low-quality (no LLM required)."""
        q = (query or "").strip()
        lines = []
        lines.append("Mình chưa tìm được thông tin tham khảo đủ sát với câu hỏi để đưa ra hướng dẫn cụ thể.")
        if q:
            lines.append(f"Câu hỏi của bạn: {q}")
        lines.append("")
        lines.append("Bạn có thể bổ sung thêm giúp mình:")
        lines.append("- Triệu chứng chính là gì, xuất hiện bao lâu, mức độ tăng/giảm?")
        lines.append("- Có sốt, nôn, tiêu chảy/táo bón, chóng mặt, khó thở, đau ngực, ngất không?")
        lines.append("- Tuổi, bệnh nền, thuốc đang dùng (nếu có).")
        lines.append("")
        lines.append("Nếu có dấu hiệu nặng (đau dữ dội, nôn ra máu, đi ngoài phân đen/ra máu, sốt cao liên tục, ngất, khó thở, đau ngực...), bạn cần đi cấp cứu ngay.")
        return "\n".join(lines)
    
    def generate_answer(self, query: str, qa_results: List[Dict], article_results: List[Dict]) -> Tuple[str, str, float]:
        """Generate natural answer using retrieved context with medical safety checks"""
        # Log retrieval results
        logger.info(f"Query: {query}")
        logger.info(f"QA results: {len(qa_results)} found, best score: {qa_results[0]['score'] if qa_results else 0}")
        logger.info(f"Article results: {len(article_results)} found, best score: {article_results[0]['score'] if article_results else 0}")
        
        # Determine specialty
        specialty = "Y tế tổng quát"
        if qa_results:
            specialty = (qa_results[0].get("topic") or "").strip() or "Y tế tổng quát"
            # Avoid low-signal buckets like "Khác" when possible
            if specialty.lower() in {"khác", "khac", "other"}:
                for qa in qa_results[:5]:
                    cand = (qa.get("topic") or "").strip()
                    if cand and cand.lower() not in {"khác", "khac", "other"}:
                        specialty = cand
                        break
                else:
                    specialty = "Y tế tổng quát"
        
        # Medical safety check (fix issue #9: detect emergencies)
        emergency_detected, emergency_type = self._check_emergency_keywords(query)
        
        # Retrieval quality
        best_qa_score = qa_results[0]['score'] if qa_results else 0.0
        best_article_score = article_results[0]['score'] if article_results else 0.0
        best_score = max(best_qa_score, best_article_score)

        # If nothing retrieved, fall back to generic medical advice.
        if not qa_results and not article_results:
            logger.warning("No retrieval results. Using generic response.")
            answer = self._generate_generic_medical_advice(query, specialty)
            confidence = 0.3
            return answer, specialty, confidence
        
        # Build context for LLM only (UI will show sources separately)
        retrieved_content = ""
        if qa_results or article_results:
            context_parts = []
            for qa in qa_results[:3]:
                q = (qa.get("question") or "").strip()
                a = (qa.get("answer") or "").strip()
                if q or a:
                    context_parts.append(f"Q: {self._truncate_text(q, 180)}\nA: {self._truncate_text(a, 300)}")
            if article_results:
                title = (article_results[0].get("title") or "").strip()
                snip = (article_results[0].get("snippet") or "").strip()
                if title or snip:
                    context_parts.append(f"Article: {self._truncate_text(title, 180)}\n{self._truncate_text(snip, 300)}")
            retrieved_content = "\n\n".join([p for p in context_parts if p])
        
        # Calculate confidence based on the best available source
        confidence = 0.35
        if best_score > 0:
            confidence = float(min(0.95, best_score + 0.1))
        
        # If emergency detected, return urgent warning
        if emergency_detected:
            logger.warning(f"Emergency detected: {emergency_type} in query: {query[:50]}...")
            answer = self._generate_emergency_response(emergency_type, specialty)
            return answer, specialty, 0.95  # High confidence for emergency warnings
        
        # Use template-based answer (faster, more reliable)
        # LLM generation disabled by default to avoid memory issues
        # Set ENABLE_LLM_GENERATION=1 in env to enable
        use_llm = os.getenv('ENABLE_LLM_GENERATION', '0') == '1'
        
        if use_llm and self.generation_model is None:
            try:
                self._load_generation_model()
            except Exception as e:
                logger.warning(f"Failed to load generation model: {e}. Using template.")
        
        if use_llm and self.generation_model is not None:
            print(f"[RAG] ✅ Using LLM generation for query: {query[:50]}...")
            answer = self._generate_with_llm(query, retrieved_content, specialty)
        else:
            print(f"[RAG] ⚠️ Using template-based generation (no LLM) for query: {query[:50]}...")
            answer = self._generate_rag_answer_no_llm(query, qa_results, article_results, specialty)

        return answer, specialty, confidence

    def build_disclaimer(self, specialty: str, confidence: float) -> str:
        """Return medical disclaimer text (kept separate so UI can format consistently)."""
        base = (
            "Thông tin chỉ mang tính chất tham khảo. "
            "Bạn nên đi khám trực tiếp để được tư vấn chính xác hơn."
        )

        if confidence < 0.6:
            base = (
                "Thông tin chỉ mang tính chất tham khảo với độ tin cậy thấp. "
                "Bạn nên đi khám trực tiếp để được tư vấn chính xác hơn."
            )

        if specialty and specialty.strip():
            base += f" (Gợi ý chuyên khoa: {specialty.strip()})."

        base += "\n\nHệ thống này KHÔNG thay thế cho ý kiến của bác sĩ. Trong trường hợp khẩn cấp, hãy gọi 115 hoặc đến bệnh viện ngay."
        return base
    
    def _check_emergency_keywords(self, query: str) -> Tuple[bool, str]:
        """Detect emergency medical situations (fix issue #9: safety guardrails)"""
        query_lower = query.lower()
        
        # Critical emergency keywords
        critical_keywords = {
            "nguy_kịch": ["nguy kịch", "hôn mê", "bất tỉnh", "ngất xỉu", "thở gấp", "khó thở nặng", "co giật"],
            "chảy_máu": ["chảy máu nhiều", "xuất huyết", "máu chảy không ngừng", "máu đỏ tươi"],
            "đau_ngực": ["đau ngực dữ dội", "đau thắt ngực", "đau tim", "nghẹt ngực"],
            "đột_quỵ": ["liệt nửa người", "méo miệng", "nói lắp", "yếu một bên", "đột quỵ"],
            "tai_nạn": ["tai nạn nghiêm trọng", "gãy xương", "chấn thương nặng", "xe đâm"],
            "ngộ_độc": ["ngộ độc", "uống nhầm", "ăn phải", "ngất sau khi ăn"]
        }
        
        for emergency_type, keywords in critical_keywords.items():
            if any(kw in query_lower for kw in keywords):
                return True, emergency_type
        
        return False, ""
    
    def _generate_emergency_response(self, emergency_type: str, specialty: str) -> str:
        """Generate urgent response for emergency situations"""
        responses = {
            "nguy_kịch": "⚠️ TÌNH HUỐNG KHẨN CẤP: Triệu chứng bạn mô tả CÓ THỂ nghiêm trọng. NGAY LẬP TỨC:\n1. GỌI 115 (cấp cứu) hoặc đưa người bệnh đến bệnh viện GẦN NHẤT\n2. Giữ bình tĩnh, theo dõi ý thức và nhịp thở\n3. KHÔNG tự ý cho uống thuốc\n\nĐây KHÔNG phải lời khuyên thay thế cấp cứu y tế chuyên nghiệp.",
            
            "chảy_máu": "⚠️ CẢNH BÁO: Chảy máu nhiều cần XỬ TRÍ NGAY:\n1. Ấn trực tiếp vào vết thương bằng vải sạch\n2. Nâng cao vị trí bị thương (nếu có thể)\n3. GỌI 115 hoặc đến cấp cứu NGAY nếu máu không cầm\n4. KHÔNG bỏ băng ra khi máu đã đông\n\nCần đánh giá y tế KHẨN CẤP.",
            
            "đau_ngực": "⚠️ KHẨN CẤP TIM MẠCH: Đau ngực có thể là dấu hiệu nhồi máu cơ tim.\nHÀNH ĐỘNG NGAY:\n1. GỌI 115 hoặc đến cấp cứu NGAY LẬP TỨC\n2. Ngồi nghỉ, KHÔNG vận động\n3. Nếu có sẵn: nhai 1 viên aspirin 300mg (trừ khi dị ứng)\n4. Theo dõi nhịp tim, hô hấp\n\nThời gian là vàng - MỖI PHÚT trì hoãn làm tăng nguy cơ.",
            
            "đột_quỵ": "⚠️ DẤU HIỆU ĐỘT QUỴ - HÀNH ĐỘNG NGAY:\nTEST NHANH (FAST):\n- Face (mặt): cười có méo miệng?\n- Arms (tay): giơ 2 tay có tay nào yếu?\n- Speech (nói): nói có lắp?\n- Time (thời gian): GỌI 115 NGAY!\n\n✅ ĐƯA BỆNH NHÂN ĐẾN BỆNH VIỆN TRONG 4.5 GIỜ ĐẦU\n❌ KHÔNG cho ăn, uống (nguy cơ sặc)\n\nĐột quỵ là KHẨN CẤP Y TẾ!",
            
            "tai_nạn": "⚠️ TAI NẠN - CẦN HỖ TRỢ Y TẾ:\n1. Đảm bảo an toàn hiện trường\n2. GỌI 115 nếu: chấn thương đầu/cột sống, gãy xương, chảy máu nhiều\n3. KHÔNG di chuyển người bị thương (trừ khi nguy hiểm)\n4. Giữ ấm, theo dõi ý thức\n\nChấn thương cần được bác sĩ ĐÁNH GIÁ CHUYÊN MÔN.",
            
            "ngộ_độc": "⚠️ NGỘ ĐỘC - XỬ TRÍ KHẨN:\n1. GỌI 115 hoặc Trung tâm Chống độc: (028) 3829 2345\n2. Mang theo bao bì/mẫu chất nghi ngờ\n3. KHÔNG tự ý gây nôn (trừ khi bác sĩ chỉ dẫn)\n4. Nếu hóa chất dính da: rửa sạch bằng nước 15-20 phút\n\nNgộ độc CẦN điều trị chuyên khoa NGAY."
        }
        
        return responses.get(emergency_type, 
            "⚠️ Triệu chứng bạn mô tả có thể nghiêm trọng. Vui lòng GỌI 115 hoặc đến cơ sở y tế GẦN NHẤT để được khám và tư vấn chuyên môn.")
    
    def _add_medical_disclaimer(self, answer: str, confidence: float) -> str:
        """Add appropriate medical disclaimer based on confidence (fix issue #9)"""
        if confidence < 0.6:
            disclaimer = "\n\n⚠️ LƯU Ý QUAN TRỌNG: Thông tin trên chỉ mang tính tham khảo với độ tin cậy THẤP. Bạn NÊN đi khám trực tiếp tại cơ sở y tế để được tư vấn chính xác."
        elif confidence < 0.8:
            disclaimer = "\n\n📋 Lưu ý: Thông tin trên mang tính tham khảo. Nếu triệu chứng kéo dài hoặc nặng hơn, vui lòng đến gặp bác sĩ để được khám và điều trị phù hợp."
        else:
            disclaimer = "\n\n💡 Lời khuyên từ hệ thống AI chỉ mang tính chất tham khảo. Để được chẩn đoán chính xác và điều trị an toàn, bạn nên đi khám trực tiếp tại cơ sở y tế."
        
        # Add general disclaimer
        disclaimer += "\n\n🏥 Hệ thống này KHÔNG thay thế cho ý kiến của bác sĩ. Trong trường hợp khẩn cấp, hãy gọi 115 hoặc đến bệnh viện ngay."
        
        return answer + disclaimer
    
    def _truncate_text(self, text: str, max_chars: int) -> str:
        """Truncate text to complete sentences"""
        if len(text) <= max_chars:
            return text
        
        # Find last complete sentence
        truncated = text[:max_chars]
        for sep in ['. ', '! ', '? ', '; ']:
            idx = truncated.rfind(sep)
            if idx > max_chars * 0.5:  # At least 50% of max_chars
                return truncated[:idx + 1].strip()
        
        # No sentence boundary, cut at word
        idx = truncated.rfind(' ')
        if idx > 0:
            return truncated[:idx] + "..."
        
        return truncated + "..."
    
    def _generate_with_llm(self, query: str, context: str, specialty: str) -> str:
        """Generate answer using Vistral-7B-Chat"""
        if self.generation_model is None or self.generation_tokenizer is None:
            return self._generate_rag_answer_no_llm(query, [], specialty)
        
        # Build prompt following Vistral format
        # NOTE: UI/API will display sources + disclaimer separately.
        # The model should ONLY output the main guidance text.
        system_prompt = f"""Bạn là bác sĩ tư vấn AI chuyên khoa {specialty}, tư vấn bằng ngôn ngữ TỰ NHIÊN, GẦN GŨI.

    QUAN TRỌNG - cách viết câu trả lời:
    - KHÔNG liệt kê kiểu "1. 2. 3." hay "- gạch đầu dòng".
    - KHÔNG copy nguyên văn thông tin RAG.
    - Hãy VIẾT LẠI thành đoạn văn tự nhiên, mượt mà, như đang nói chuyện trực tiếp với người hỏi.
    - Giọng văn thân thiện, dễ hiểu, nhưng chuyên nghiệp.

    Nội dung cần có (nhưng viết dạng văn xuôi, không đánh số):
    - Mở đầu: tóm tắt tình huống (1-2 câu).
    - Phần chính: khuyến nghị cụ thể về theo dõi, xử trí, điều cần/không nên làm. Viết thành các câu liền mạch, không tách thành list.
    - Dấu hiệu cảnh báo: nhắc nhở những triệu chứng nặng cần khám ngay. Tích hợp vào đoạn văn tự nhiên.
    - Kết: 1-2 câu hỏi làm rõ nếu cần thêm thông tin.

    RÀNG BUỘC:
    - Bắt buộc dùng ít nhất 2 chi tiết từ "THÔNG TIN RAG" (số liệu, ngưỡng, biện pháp cụ thể).
    - Không bịa thông tin không có trong RAG. Nếu thiếu, ghi "thông tin chưa nêu rõ".
    - Không chào hỏi, không disclaimer (hệ thống tự thêm).
    - Ngôn ngữ thận trọng: "có thể", "thường", "nên".

    VÍ DỤ CÂU TRẢ LỜI TỐT:
    "Khi trẻ 3 tuổi sốt 39 độ, điều đầu tiên là theo dõi thân nhiệt đều đặn mỗi 4 giờ và cho bé uống nhiều nước để tránh mất nước. Có thể dùng paracetamol hoặc ibuprofen theo chỉ định để hạ sốt, kết hợp chườm ấm lên trán. Nếu sốt kéo dài quá 3 ngày, trẻ li bì khó đánh thức, hoặc thở nhanh co rút lồng ngực, cần đưa bé đến cơ sở y tế ngay. Bạn có thể cho biết bé đã uống thuốc hạ sốt chưa và có dấu hiệu nào bất thường khác không?"

    VÍ DỤ CÂU TRẢ LỜI XẤU (TRÁNH):
    "1. Theo dõi thân nhiệt. 2. Cho uống nước. 3. Dùng thuốc hạ sốt..."

    Định dạng: tiếng Việt, văn xuôi tự nhiên, 2–4 đoạn ngắn."""

        user_message = f"""CÂU HỎI: {query}

    THÔNG TIN TRUY XUẤT (RAG) - chỉ dùng làm ngữ cảnh:
    {context}

    Hãy trả lời đúng theo yêu cầu hệ thống."""

        # Format prompt (Vistral uses ChatML format)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]
        
        try:
            # Apply chat template
            prompt = self.generation_tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            
            # Tokenize
            inputs = self.generation_tokenizer(
                prompt, 
                return_tensors="pt",
                truncation=True,
                max_length=2048
            ).to(self.device)
            
            # Generate
            with torch.no_grad():
                outputs = self.generation_model.generate(
                    **inputs,
                    max_new_tokens=settings.MAX_NEW_TOKENS,
                    temperature=settings.TEMPERATURE,
                    top_p=settings.TOP_P,
                    do_sample=True,
                    pad_token_id=self.generation_tokenizer.eos_token_id
                )
            
            # Decode
            response = self.generation_tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:], 
                skip_special_tokens=True
            )
            
            return response.strip()
            
        except Exception as e:
            print(f"[RAG Engine] LLM generation failed: {e}")
            return self._generate_rag_answer_no_llm(query, [], specialty)
    
    def _strip_greeting_noise(self, text: str) -> str:
        if not text:
            return ""
        t = text.strip()
        t = re.sub(r'^\s*(trả\s*lời\s*[:\-]?\s*)', '', t, flags=re.IGNORECASE)
        t = re.sub(r'^\s*(xin\s+chào|chào)\s+(bạn|anh|chị|em|bác)\s*[,!.:;-]*\s*', '', t, flags=re.IGNORECASE)
        t = re.sub(r'^\s*(thưa\s+bác\s+sĩ|thưa\s+bs)\s*[,!.:;-]*\s*', '', t, flags=re.IGNORECASE)
        return t.strip()

    def _extract_action_sentences(self, text: str, max_sentences: int = 5) -> List[str]:
        """Extract short actionable sentences from a passage."""
        if not text:
            return []
        raw = self._clean_text(text)
        sents = re.split(r'(?<=[.!?])\s+', raw)
        out: List[str] = []
        for s in sents:
            s = self._strip_greeting_noise(s)
            if len(s) < 12:
                continue
            s_proc = self._preprocess_reference_sentence(s)
            if not s_proc:
                continue
            if self._sentence_has_action(s_proc) or self._sentence_has_action(s):
                out.append(s)
            if len(out) >= max_sentences:
                break
        return out

    def _generate_rag_answer_no_llm(
        self,
        query: str,
        qa_results: List[Dict],
        article_results: List[Dict],
        specialty: str,
    ) -> str:
        """Generate a readable answer using retrieved QAs + Articles (no LLM)."""
        bullets: List[str] = []

        # 1) From QAs: try extracting actionable sentences
        action_text, _, _ = self.find_best_action_sentence(
            user_text=query,
            topk_rows=qa_results[:8],
            sent_sim_thresh=0.55,
            combined_thresh=0.62,
        )
        if action_text:
            cleaned = self._strip_greeting_noise(action_text)
            if cleaned:
                bullets.extend(self._extract_action_sentences(cleaned, max_sentences=4) or [self._truncate_text(cleaned, 280)])

        # 2) From Articles: use best passage to extract actionable sentences
        if article_results:
            passage = (article_results[0].get("snippet") or "").strip()
            passage = self._strip_greeting_noise(passage)
            if passage:
                bullets.extend(self._extract_action_sentences(passage, max_sentences=4))

        # 3) Fallbacks if still empty: use top QA answer / article passage snippet
        if not bullets and qa_results:
            top_answer = self._strip_greeting_noise((qa_results[0].get("answer") or "").strip())
            if top_answer:
                bullets.append(self._truncate_text(top_answer, 300))
        if not bullets and article_results:
            passage = self._strip_greeting_noise((article_results[0].get("snippet") or "").strip())
            if passage:
                bullets.append(self._truncate_text(passage, 300))

        if not bullets:
            return self._generate_generic_medical_advice(query, specialty)

        # Deduplicate + format
        uniq: List[str] = []
        seen = set()
        for b in bullets:
            b2 = re.sub(r'\s+', ' ', (b or '').strip())
            if len(b2) < 12:
                continue
            key = b2.lower()
            if key in seen:
                continue
            seen.add(key)
            uniq.append(b2)
            if len(uniq) >= 6:
                break

        if not uniq:
            return self._generate_generic_medical_advice(query, specialty)

        lines = ["Gợi ý dựa trên thông tin tham khảo:"]
        for b in uniq:
            lines.append(f"- {b}")

        # Add 1–2 clarifying questions when query is too short
        if len((query or "").strip()) <= 25:
            lines.append("")
            lines.append("Bạn có thể cho mình biết thêm:")
            lines.append("- Bạn bị tiêu chảy bao lâu rồi, số lần/ngày, có sốt hoặc đau bụng không?")
            lines.append("- Có dấu hiệu mất nước (khát nhiều, tiểu ít, chóng mặt) hoặc đi ngoài ra máu không?")

        return "\n".join(lines)
    
    def get_specialties(self) -> List[Dict]:
        """Get list of available specialties"""
        if self.df_qa is None or 'topic' not in self.df_qa.columns:
            return []
        
        specialty_counts = self.df_qa['topic'].value_counts().to_dict()
        return [
            {"name": name, "count": count}
            for name, count in sorted(specialty_counts.items(), key=lambda x: x[1], reverse=True)
        ]

# Global instance
_rag_engine: Optional[HealthcareRAGEngine] = None

def get_rag_engine() -> HealthcareRAGEngine:
    """Get or create RAG engine singleton"""
    global _rag_engine
    if _rag_engine is None:
        print("[RAG] Step 1: Creating HealthcareRAGEngine instance...")
        _rag_engine = HealthcareRAGEngine()
        print("[RAG] Step 2: Loading models...")
        _rag_engine.load_models()
        print("[RAG] Step 3: Loading data...")
        _rag_engine.load_data()
        print("[RAG] Step 4: Building/loading indices...")
        _rag_engine.build_indices()
        print("[RAG] ✅ All steps completed!")
    return _rag_engine
