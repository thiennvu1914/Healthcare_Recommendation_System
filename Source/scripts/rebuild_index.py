#!/usr/bin/env python
"""
Script to rebuild FAISS indices with full dataset
This will take time but ensures accurate search results
"""
import sys
sys.path.insert(0, '/root/Healthcare_Recommendation_System')

import os
os.environ["SAMPLE_SIZE"] = "0"   # Use all data (0 = unlimited)

from api.rag_engine import get_rag_engine

print("="*70)
print("REBUILDING FAISS INDICES WITH FULL DATASET")
print("="*70)
print("This will process:")
print("  - 60,234 Q&A pairs")
print("  - 88,590 articles")
print("  - Total: 148,824 medical documents")
print()
print("Expected time: 30-60 minutes")
print("Steps:")
print("  1. Loading PhoBERT model")
print("  2. Encoding all Q&A pairs")
print("  3. Encoding all articles")
print("  4. Building FAISS indices")
print("  5. Saving cache for future use")
print("="*70)
print()

try:
    engine = get_rag_engine()
    print()
    print("="*70)
    print("✅ SUCCESS! Indices rebuilt and cached.")
    print("="*70)
    print(f"  - Q&A index size: {engine.qa_index.ntotal if engine.qa_index else 0:,} items")
    print(f"  - Article index size: {engine.article_index.ntotal if engine.article_index else 0:,} items")
    print()
    print("Next steps:")
    print("  1. Restart API server with new indices:")
    print("     cd /root/Healthcare_Recommendation_System")
    print("     SAMPLE_SIZE=0 uvicorn api.main:app --host 0.0.0.0 --port 8000 &")
    print()
except Exception as e:
    print()
    print("="*70)
    print(f"❌ ERROR: {e}")
    print("="*70)
    import traceback
    traceback.print_exc()
    sys.exit(1)
