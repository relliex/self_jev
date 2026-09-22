import os
import time
import numpy as np
import torch
import torch.nn as nn

# Disable torchao warning / version checks in peft
import peft.import_utils
peft.import_utils.is_torchao_available = lambda: False
try:
    import peft.tuners.lora.torchao
    peft.tuners.lora.torchao.is_torchao_available = lambda: False
except (ImportError, AttributeError):
    pass

from transformers import AutoModel, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType
from datasets import load_dataset
from torch.utils.data import DataLoader

MODEL_ID = "Qwen/Qwen3.5-0.8B-Base"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_DIR = "jev_tri_primitive_real"
PATIENCE = 3
EVAL_EVERY_STEPS = 50
TOTAL_EPOCHS = 3

# =========================================================
# 1. Dataset Loading & Multi-Task Protocol Alignment
# =========================================================
print("[INFO] Loading datasets from Hugging Face...")

# 1.1 Noul Data Source (BoolQ -> Binary Yes/No)
ds_boolq = load_dataset("google/boolq")
train_boolq = ds_boolq["train"].select(range(600))
val_boolq = ds_boolq["validation"].select(range(100))

# 1.2 Choice Data Source (ARC-Easy -> 4-Way Categorical)
ds_arc = load_dataset("allenai/ai2_arc", "ARC-Easy")
def filter_arc(ds, num_samples):
    data = []
    label_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3, '1': 0, '2': 1, '3': 2, '4': 3}
    for item in ds:
        if len(item['choices']['text']) == 4 and item['answerKey'] in label_map:
            data.append({
                "question": item['question'],
                "choices": item['choices']['text'],
                "label": label_map[item['answerKey']]
            })
        if len(data) >= num_samples:
            break
    return data

train_arc = filter_arc(ds_arc["train"], 600)
val_arc = filter_arc(ds_arc["validation"], 100)

# 1.3 Score Data Source (GLUE STS-B -> Normalized Continuous [0, 1])
ds_stsb = load_dataset("nyu-mll/glue", "stsb")
train_stsb = ds_stsb["train"].select(range(600))
val_stsb = ds_stsb["validation"].select(range(100))

def build_dataset_pool(boolq_data, arc_data, stsb_data):
    pool = []
    # Noul primitive format
    for x in boolq_data:
        p = f"<|im_start|>state\n{x['passage']}<|im_end|>\n<|im_start|>question\n{x['question']}?<|im_end|>\n<|im_start|>decision\n"
        pool.append({"prompt": p, "mode": "noul", "label": 1.0 if x['answer'] else 0.0})
    # Choice primitive format
    for x in arc_data:
        c = x['choices']
        opts = f"A: {c[0]} | B: {c[1]} | C: {c[2]} | D: {c[3]}"
        p = f"<|im_start|>state\n{x['question']}<|im_end|>\n<|im_start|>question\n候选项: {opts}<|im_end|>\n<|im_start|>decision\n"
        pool.append({"prompt": p, "mode": "choice", "label": x['label']})
    # Score primitive format
    for x in stsb_data:
        p = f"<|im_start|>state\n文本A: {x['sentence1']}\n文本B: {x['sentence2']}<|im_end|>\n<|im_start|>question\n请评估两段文本的语义匹配程度百分比。<|im_end|>\n<|im_start|>decision\n"
        norm_score = float(np.clip(x['label'] / 5.0, 0.0, 1.0))
        pool.append({"prompt": p, "mode": "score", "label": norm_score})
    np.random.seed(42)
    np.random.shuffle(pool)
    return pool

train_pool = build_dataset_pool(train_boolq, train_arc, train_stsb)
val_pool = build_dataset_pool(val_boolq, val_arc, val_stsb)
print(f"[INFO] Datasets loaded. Train: {len(train_pool)} | Val: {len(val_pool)}")

# =========================================================
# 2. Model Architecture Assembly (Encoder + 3 Decision Heads)
# =========================================================
print(f"[INFO] Initializing {MODEL_ID}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
tokenizer.padding_side = "left"
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

class JevTriPrimitiveModel(nn.Module):
    """
    Self-Jev Architecture:
    - Backbone: Qwen-0.8B (Feature Extractor) with LoRA adapters
    - Latent Representation: Contextual pooling at the decision boundary token
    - 3 Dedicated Heads:
        * noul_head: Binary classification (BCEWithLogitsLoss)
        * choice_head: Multi-class classification (CrossEntropyLoss)
        * score_head: Continuous regression (MSELoss)
    """
    def __init__(self, model_id):
        super().__init__()
        base_model = AutoModel.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
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
        
        self.noul_head = nn.Sequential(nn.Linear(h, h // 2), nn.GELU(), nn.Linear(h // 2, 1)).to(DEVICE, dtype=torch.float32)
        self.choice_head = nn.Sequential(nn.Linear(h, h // 2), nn.GELU(), nn.Linear(h // 2, 4)).to(DEVICE, dtype=torch.float32)
        self.score_head = nn.Sequential(nn.Linear(h, h // 2), nn.GELU(), nn.Linear(h // 2, 1)).to(DEVICE, dtype=torch.float32)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        target_dtype = self.noul_head[0].weight.dtype
        # Pool the contextual representation at the final token
        return outputs.last_hidden_state[:, -1, :].to(target_dtype)

model = JevTriPrimitiveModel(MODEL_ID)

for p in model.parameters():
    if p.requires_grad:
        p.data = p.data.to(torch.float32)

print("[INFO] Model initialized successfully.")

# =========================================================
# 3. Dynamic Collation & Optimizer Setup
# =========================================================
def collate_fn(batch):
    prompts = [b["prompt"] for b in batch]
    enc = tokenizer(prompts, padding=True, truncation=True, max_length=192, return_tensors="pt")
    enc["mode"] = [b["mode"] for b in batch]
    enc["label"] = [b["label"] for b in batch]
    return enc

BATCH_SIZE = 4
GRAD_ACCUM_STEPS = 4

train_loader = DataLoader(train_pool, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_pool, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

bce_loss = nn.BCEWithLogitsLoss()
ce_loss = nn.CrossEntropyLoss()
mse_loss = nn.MSELoss()

optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1.5e-4)
scaler = torch.amp.GradScaler("cuda" if DEVICE == "cuda" else "cpu")

# =========================================================
# 4. Multi-Task Training with Early Stopping
# =========================================================
print(f"[INFO] Training started. Validation every {EVAL_EVERY_STEPS} steps...")
best_val_loss = float("inf")
patience_counter = 0
early_stopped = False
os.makedirs(SAVE_DIR, exist_ok=True)
checkpoint_path = os.path.join(SAVE_DIR, "jev_best_checkpoint.pt")

model.train()
optimizer.zero_grad()

for epoch in range(TOTAL_EPOCHS):
    if early_stopped:
        break
        
    for step, batch in enumerate(train_loader):
        input_ids = batch["input_ids"].to(DEVICE)
        mask = batch["attention_mask"].to(DEVICE)
        modes = batch["mode"]
        labels = batch["label"]
        
        with torch.amp.autocast(device_type="cuda" if DEVICE == "cuda" else "cpu", dtype=torch.float16 if DEVICE == "cuda" else torch.bfloat16):
            decision_vecs = model(input_ids, mask)
            loss = torch.tensor(0.0, device=DEVICE)
            
            n_idx = [i for i, m in enumerate(modes) if m == "noul"]
            c_idx = [i for i, m in enumerate(modes) if m == "choice"]
            s_idx = [i for i, m in enumerate(modes) if m == "score"]
            
            if n_idx:
                n_preds = model.noul_head(decision_vecs[n_idx]).squeeze(-1)
                n_targets = torch.tensor([labels[i] for i in n_idx], device=DEVICE, dtype=torch.float32)
                loss = loss + bce_loss(n_preds, n_targets)
                
            if c_idx:
                c_preds = model.choice_head(decision_vecs[c_idx])
                c_targets = torch.tensor([labels[i] for i in c_idx], device=DEVICE, dtype=torch.long)
                loss = loss + ce_loss(c_preds, c_targets)
                
            if s_idx:
                s_preds = torch.sigmoid(model.score_head(decision_vecs[s_idx]).squeeze(-1))
                s_targets = torch.tensor([labels[i] for i in s_idx], device=DEVICE, dtype=torch.float32)
                loss = loss + mse_loss(s_preds, s_targets)
                
            loss = loss / GRAD_ACCUM_STEPS
            
        scaler.scale(loss).backward()
        
        if (step + 1) % GRAD_ACCUM_STEPS == 0 or (step + 1) == len(train_loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            
        if (step + 1) % 10 == 0:
            current_loss = loss.item() * GRAD_ACCUM_STEPS
            print(f"[TRAIN] Epoch {epoch+1}/{TOTAL_EPOCHS} | Step [{step+1:03d}/{len(train_loader)}] | Loss: {current_loss:.4f}")
            
        # Periodic validation & early stopping check
        if (step + 1) % EVAL_EVERY_STEPS == 0:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for v_batch in val_loader:
                    v_ids = v_batch["input_ids"].to(DEVICE)
                    v_mask = v_batch["attention_mask"].to(DEVICE)
                    v_modes = v_batch["mode"]
                    v_labels = v_batch["label"]
                    
                    with torch.amp.autocast(device_type="cuda" if DEVICE == "cuda" else "cpu", dtype=torch.float16 if DEVICE == "cuda" else torch.bfloat16):
                        v_vecs = model(v_ids, v_mask)
                        vl = torch.tensor(0.0, device=DEVICE)
                        
                        vn_idx = [i for i, m in enumerate(v_modes) if m == "noul"]
                        vc_idx = [i for i, m in enumerate(v_modes) if m == "choice"]
                        vs_idx = [i for i, m in enumerate(v_modes) if m == "score"]
                        
                        if vn_idx:
                            vl = vl + bce_loss(model.noul_head(v_vecs[vn_idx]).squeeze(-1), torch.tensor([v_labels[i] for i in vn_idx], device=DEVICE, dtype=torch.float32))
                        if vc_idx:
                            vl = vl + ce_loss(model.choice_head(v_vecs[vc_idx]), torch.tensor([v_labels[i] for i in vc_idx], device=DEVICE, dtype=torch.long))
                        if vs_idx:
                            vl = vl + mse_loss(torch.sigmoid(model.score_head(v_vecs[vs_idx]).squeeze(-1)), torch.tensor([v_labels[i] for i in vs_idx], device=DEVICE, dtype=torch.float32))
                            
                        val_loss += vl.item()
                        
            avg_val_loss = val_loss / len(val_loader)
            log_str = f"[EVAL] Step [{step+1:03d}/{len(train_loader)}] | Validation Loss: {avg_val_loss:.4f}"
            
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                torch.save(model.state_dict(), checkpoint_path)
                tokenizer.save_pretrained(SAVE_DIR)
                print(f"{log_str} -> [SAVED] Best checkpoint updated.")
            else:
                patience_counter += 1
                print(f"{log_str} -> [WARNING] No improvement ({patience_counter}/{PATIENCE})")
                if patience_counter >= PATIENCE:
                    print(f"[EARLY_STOP] Validation loss failed to improve for {PATIENCE} checks. Terminating.")
                    early_stopped = True
                    break
                    
            model.train()

print(f"[INFO] Optimal checkpoint saved at: {checkpoint_path}")
