"""
GPU Service for FPT AI Factory Container
Chạy file này trên container để serve GPU inference
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
import numpy as np
import uvicorn

app = FastAPI(title="Healthcare GPU Service")

# Global models
tokenizer_phobert = None
model_phobert = None
gen_tokenizer = None
gen_model = None

class EmbeddingRequest(BaseModel):
    texts: List[str]
    max_length: int = 256

class GenerationRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9

@app.on_event("startup")
async def load_models():
    """Load models on startup"""
    global tokenizer_phobert, model_phobert, gen_tokenizer, gen_model
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 Loading models on {device}")
    
    # PhoBERT
    print("📦 Loading PhoBERT...")
    tokenizer_phobert = AutoTokenizer.from_pretrained("vinai/phobert-base", use_fast=False)
    model_phobert = AutoModel.from_pretrained("vinai/phobert-base").to(device)
    model_phobert.eval()
    print(f"✅ PhoBERT ready")
    
    # Vistral
    print("📦 Loading Vistral-7B-Chat...")
    gen_tokenizer = AutoTokenizer.from_pretrained("Viet-Mistral/Vistral-7B-Chat")
    gen_model = AutoModelForCausalLM.from_pretrained(
        "Viet-Mistral/Vistral-7B-Chat",
        torch_dtype=torch.float16,
        device_map="auto"
    )
    gen_model.eval()
    print(f"✅ Vistral ready")
    
    if torch.cuda.is_available():
        vram = torch.cuda.memory_allocated() / 1024**3
        print(f"🎉 Models loaded! VRAM: {vram:.2f} GB")

@app.get("/health")
async def health():
    """Health check"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vram_used = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
    
    return {
        "status": "healthy",
        "device": device,
        "models_loaded": {
            "phobert": model_phobert is not None,
            "vistral": gen_model is not None
        },
        "vram_gb": round(vram_used, 2)
    }

@app.post("/batch_embed")
async def batch_embed(request: EmbeddingRequest):
    """Batch embedding"""
    if model_phobert is None:
        raise HTTPException(503, "Models not loaded")
    
    try:
        device = next(model_phobert.parameters()).device
        
        # Tokenize
        inputs = tokenizer_phobert(
            request.texts,
            return_tensors="pt",
            max_length=request.max_length,
            padding=True,
            truncation=True
        ).to(device)
        
        # Embed
        with torch.no_grad():
            outputs = model_phobert(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1)
        
        return {
            "embeddings": embeddings.cpu().numpy().tolist(),
            "device": str(device),
            "batch_size": len(request.texts)
        }
    except Exception as e:
        raise HTTPException(500, f"Error: {e}")

@app.post("/generate")
async def generate(request: GenerationRequest):
    """Generate text"""
    if gen_model is None:
        raise HTTPException(503, "Generation model not loaded")
    
    try:
        device = next(gen_model.parameters()).device
        
        # Tokenize
        inputs = gen_tokenizer(request.prompt, return_tensors="pt").to(device)
        
        # Generate
        with torch.no_grad():
            outputs = gen_model.generate(
                **inputs,
                max_new_tokens=request.max_new_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                do_sample=True,
                pad_token_id=gen_tokenizer.eos_token_id
            )
        
        # Decode
        generated = gen_tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Remove prompt from output
        if generated.startswith(request.prompt):
            generated = generated[len(request.prompt):].strip()
        
        return {
            "generated_text": generated,
            "device": str(device)
        }
    except Exception as e:
        raise HTTPException(500, f"Error: {e}")

if __name__ == "__main__":
    print("=" * 60)
    print("🚀 Healthcare GPU Service")
    print("=" * 60)
    print("📍 Running on FPT AI Factory container")
    print("🔧 Port 8888 (HTTP exposed)")
    print("=" * 60)
    
    uvicorn.run(app, host="0.0.0.0", port=8888, log_level="info")
