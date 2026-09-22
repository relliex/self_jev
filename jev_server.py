import os
os.environ["HF_ENDPOINT"] = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import time
import json
import torch
import torch.nn as nn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any, Union
from transformers import AutoModel, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType
import numpy as np

# Suppress redundant torchao check
import peft.import_utils
peft.import_utils.is_torchao_available = lambda: False

MODEL_ID = "Qwen/Qwen3.5-0.8B-Base"
CHECKPOINT_PATH = "jev_best_checkpoint.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

app = FastAPI(
    title="Self-Jev: Local System One Decision Gateway",
    description="High-performance, zero-hallucination decision model implementing Noul, Choice, and Score primitives.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QuestionItem(BaseModel):
    type: str = Field(..., description="Decision primitive type: 'noul', 'choice', or 'score'")
    instructions: str = Field(..., description="Decision criteria or question to evaluate")
    criteria: Optional[Union[Dict[str, str], List[str]]] = Field(
        None, description="Optional options list or key-value map for choice primitive"
    )

class SystemOneRequest(BaseModel):
    model: Optional[str] = "jev-latest"
    state: Any = Field(..., description="Program context, agent state, or observation")
    questions: Dict[str, QuestionItem] = Field(..., description="Map of decision questions")

class JevTriPrimitiveModel(nn.Module):
    def __init__(self, model_id):
        super().__init__()
        base_model = AutoModel.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
            trust_remote_code=True
        ).to(DEVICE)
        
        base_model.gradient_checkpointing_enable()
        
        lora_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05
        )
        self.encoder = get_peft_model(base_model, lora_config)
        
        cfg = base_model.config.get_text_config() if hasattr(base_model.config, "get_text_config") else base_model.config
        h = getattr(cfg, "hidden_size", getattr(cfg, "d_model", 1024))
        target_dtype = torch.float16 if DEVICE == "cuda" else torch.float32
        
        self.noul_head = nn.Sequential(nn.Linear(h, h // 2), nn.GELU(), nn.Linear(h // 2, 1)).to(DEVICE, dtype=target_dtype)
        self.choice_head = nn.Sequential(nn.Linear(h, h // 2), nn.GELU(), nn.Linear(h // 2, 4)).to(DEVICE, dtype=target_dtype)
        self.score_head = nn.Sequential(nn.Linear(h, h // 2), nn.GELU(), nn.Linear(h // 2, 1)).to(DEVICE, dtype=target_dtype)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        target_dtype = self.noul_head[0].weight.dtype
        return outputs.last_hidden_state[:, -1, :].to(target_dtype)

print(f"[INFO] Initializing tokenizer and base model ({MODEL_ID})...")
tokenizer = AutoTokenizer.from_pretrained(".", trust_remote_code=True)
tokenizer.padding_side = "left"
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = JevTriPrimitiveModel(MODEL_ID)

if os.path.exists(CHECKPOINT_PATH):
    print(f"[INFO] Loading fine-tuned weights from {CHECKPOINT_PATH}...")
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
else:
    print(f"[WARN] Checkpoint {CHECKPOINT_PATH} not found in current directory. Running in zero-shot initialized mode.")

if DEVICE == "cuda":
    model.half()
model.eval()

# Warmup GPU
dummy = tokenizer("<|im_start|>state\nwarmup<|im_end|>\n<|im_start|>question\nwarmup?<|im_end|>\n<|im_start|>decision\n", return_tensors="pt").to(DEVICE)
with torch.no_grad():
    _ = model(dummy["input_ids"], dummy["attention_mask"])
if DEVICE == "cuda":
    torch.cuda.synchronize()

print("[INFO] Self-Jev System One Gateway operational on http://127.0.0.1:8000/v1/systemone")

@app.post("/v1/systemone")
@app.post("/systemone")
def system_one_endpoint(req: SystemOneRequest):
    state_repr = json.dumps(req.state, ensure_ascii=False) if not isinstance(req.state, str) else req.state
    results = {}
    
    for q_id, q_item in req.questions.items():
        mode = q_item.type.lower()
        instructions = q_item.instructions
        criteria = q_item.criteria
        
        if mode == "choice":
            if isinstance(criteria, dict):
                options = [f"{k}: {v}" for k, v in criteria.items()]
            elif isinstance(criteria, list):
                options = criteria
            else:
                options = ["A: 选项一", "B: 选项二", "C: 选项三", "D: 选项四"]
            prompt = f"<|im_start|>state\n{state_repr}<|im_end|>\n<|im_start|>question\n{instructions}\n候选项: {' | '.join(options)}<|im_end|>\n<|im_start|>decision\n"
        else:
            prompt = f"<|im_start|>state\n{state_repr}<|im_end|>\n<|im_start|>question\n{instructions}<|im_end|>\n<|im_start|>decision\n"
            
        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        
        with torch.no_grad():
            vec = model(inputs["input_ids"], inputs["attention_mask"])
            
            if mode == "noul":
                prob = torch.sigmoid(model.noul_head(vec).squeeze(-1)).item()
                results[q_id] = {
                    "type": "noul",
                    "noul": round(prob, 4),
                    "result": bool(prob > 0.5)
                }
            elif mode == "choice":
                probs = torch.softmax(model.choice_head(vec), dim=-1).squeeze().tolist()
                best_idx = int(np.argmax(probs))
                prob_map = {options[i]: round(probs[i], 4) for i in range(min(len(options), len(probs)))}
                results[q_id] = {
                    "type": "choice",
                    "choice": options[best_idx] if best_idx < len(options) else options[0],
                    "probabilities": prob_map,
                    "confidence": round(probs[best_idx], 4)
                }
            elif mode == "score":
                score_val = torch.sigmoid(model.score_head(vec).squeeze(-1)).item()
                results[q_id] = {
                    "type": "score",
                    "score": round(score_val, 4),
                    "confidence": round(score_val, 4)
                }
            else:
                results[q_id] = {"error": f"Unknown primitive type: {mode}"}
                
    return {
        "model": "jev-latest",
        "results": results
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
