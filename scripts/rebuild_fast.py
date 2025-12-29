#!/usr/bin/env python
"""Fast rebuild with GPU and batch processing"""
import sys
sys.path.insert(0, '/root/Healthcare_Recommendation_System')

import os
os.environ["SAMPLE_SIZE"] = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import numpy as np
import pandas as pd
import faiss
from pathlib import Path
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

print("="*70)
print("FAST FAISS INDEX REBUILD WITH GPU")
print("="*70)

# Check GPU
if torch.cuda.is_available():
    print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
    print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    print("⚠️  No GPU found, using CPU")

# Paths
base_dir = Path('/root/Healthcare_Recommendation_System')
data_dir = base_dir / 'data'
cache_dir = base_dir / 'cache'
cache_dir.mkdir(exist_ok=True)

print("\n1. Loading data...")
df_qa = pd.read_csv(data_dir / 'QAs.csv')
df_articles = pd.read_csv(data_dir / 'articles.csv')
print(f"   Q&A: {len(df_qa):,} items")
print(f"   Articles: {len(df_articles):,} items")

print("\n2. Loading PhoBERT model...")
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
dtype = torch.float16 if device == "cuda" else torch.float32
model = AutoModel.from_pretrained("vinai/phobert-base", torch_dtype=dtype).to(device)
model.eval()
print(f"   Model loaded on {device}")

def get_embedding(text, batch_size=32):
    """Get embeddings with batching"""
    embeddings = []
    
    for i in tqdm(range(0, len(text), batch_size), desc="   Encoding"):
        batch = text[i:i+batch_size]
        
        inputs = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors='pt'
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.inference_mode():
            outputs = model(**inputs)
            last_hidden = outputs.last_hidden_state  # [B, T, H]
            attention_mask = inputs.get("attention_mask")
            if attention_mask is not None:
                mask = attention_mask.unsqueeze(-1).to(last_hidden.dtype)
                masked = last_hidden * mask
                # Accumulate in fp32 for stability, then output fp32 for FAISS
                summed = masked.sum(dim=1, dtype=torch.float32)
                counts = mask.sum(dim=1, dtype=torch.float32).clamp(min=1e-6)
                batch_emb = (summed / counts).cpu().numpy().astype('float32')
            else:
                batch_emb = last_hidden.mean(dim=1).cpu().numpy().astype('float32')
            embeddings.append(batch_emb)
    
    return np.vstack(embeddings).astype('float32')

print("\n3. Encoding Q&A pairs...")
qa_texts = (df_qa['question'].fillna('') + ' ' + df_qa['answer'].fillna('')).tolist()
batch_size = int(os.getenv("BATCH_SIZE", "128"))
qa_embeddings = get_embedding(qa_texts, batch_size=batch_size)
print(f"   Shape: {qa_embeddings.shape}")

print("\n4. Building Q&A index...")
faiss.normalize_L2(qa_embeddings)
qa_index = faiss.IndexFlatIP(qa_embeddings.shape[1])
qa_index.add(qa_embeddings)
print(f"   Index size: {qa_index.ntotal:,}")

print("\n5. Encoding articles...")
article_texts = (df_articles['title'].fillna('') + ' ' + df_articles['text'].fillna('')).tolist()
article_embeddings = get_embedding(article_texts, batch_size=batch_size)
print(f"   Shape: {article_embeddings.shape}")

print("\n6. Building article index...")
faiss.normalize_L2(article_embeddings)
article_index = faiss.IndexFlatIP(article_embeddings.shape[1])
article_index.add(article_embeddings)
print(f"   Index size: {article_index.ntotal:,}")

print("\n7. Saving to cache...")
faiss.write_index(qa_index, str(cache_dir / 'qa_index.bin'))
faiss.write_index(article_index, str(cache_dir / 'article_index.bin'))
np.save(cache_dir / 'qa_embeddings.npy', qa_embeddings)
np.save(cache_dir / 'article_embeddings.npy', article_embeddings)

import pickle
metadata = {
    'qa_count': len(df_qa),
    'article_count': len(df_articles),
    'model': 'vinai/phobert-base',
    'embedding_dim': qa_embeddings.shape[1],
    'sample_size': 0,
    'embedding_strategy': 'mean_pool_last_hidden'
}
with open(cache_dir / 'metadata.pkl', 'wb') as f:
    pickle.dump(metadata, f)

print("\n" + "="*70)
print("✅ SUCCESS!")
print("="*70)
print(f"Q&A index: {qa_index.ntotal:,} items")
print(f"Article index: {article_index.ntotal:,} items")
print(f"\nCache saved to: {cache_dir}")
print(f"  - qa_index.bin ({(cache_dir / 'qa_index.bin').stat().st_size / 1e6:.1f} MB)")
print(f"  - article_index.bin ({(cache_dir / 'article_index.bin').stat().st_size / 1e6:.1f} MB)")
print("\nNext: Start API server")
print("  cd /root/Healthcare_Recommendation_System")
print("  SAMPLE_SIZE=0 uvicorn api.main:app --host 0.0.0.0 --port 8000 &")
print("="*70)
