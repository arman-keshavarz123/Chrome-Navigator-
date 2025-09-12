#!/usr/bin/env python3

import json
import re
import traceback
from pathlib import Path

import torch
from flask import Flask, jsonify, request
import json

from models import MiniRPCTransformer

CHECKPOINT = Path("runs/XXX_best_model.pt")
TOKENIZER_PATH = Path("rpc_tokenizer.json")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load checkpoint to get vocabulary info
ckpt = torch.load(CHECKPOINT, map_location=DEVICE)
VOCAB = ckpt.get("vocab", 4127)
PAD_ID = ckpt.get("pad_id", 0)

# Simple tokenizer class
class SimpleTokenizer:
    def __init__(self, vocab_size, pad_id):
        self.vocab_size = vocab_size
        self.pad_id = pad_id
        # Simple word-based tokenization
        self.word_to_id = {}
        self.id_to_word = {}
        # Add basic tokens
        self.word_to_id["<PAD>"] = pad_id
        self.word_to_id["<BOS>"] = 1
        self.word_to_id["<EOS>"] = 2
        self.word_to_id["<UTT>"] = 3
        self.word_to_id["</UTT>"] = 4
        
        # Create reverse mapping
        for word, id in self.word_to_id.items():
            self.id_to_word[id] = word
    
    def encode(self, text, add_special_tokens=False):
        # Simple word-based encoding
        words = text.split()
        ids = []
        for word in words:
            if word in self.word_to_id:
                ids.append(self.word_to_id[word])
            else:
                # Simple hash-based ID for unknown words
                ids.append(hash(word) % (self.vocab_size - 10) + 10)
        return ids
    
    def decode(self, ids, skip_special_tokens=False):
        words = []
        for id in ids:
            if id in self.id_to_word:
                if skip_special_tokens and self.id_to_word[id].startswith("<"):
                    continue
                words.append(self.id_to_word[id])
            else:
                words.append(f"<UNK{id}>")
        return " ".join(words)

tok = SimpleTokenizer(VOCAB, PAD_ID)
BOS_ID = 1
EOS_ID = 2

params = ckpt.get(
    "params",
    {
        "d_model": 256,
        "n_heads": 4,
        "n_enc_layers": 4,
        "n_dec_layers": 4,
        "max_len": 128,
        "dropout": 0.1,
    },
)

model = MiniRPCTransformer(vocab_size=VOCAB, pad_id=PAD_ID, **params).to(DEVICE)
model.load_state_dict(ckpt["model"], strict=True)
model.eval()

print(f"Loaded {CHECKPOINT} • vocab={VOCAB} • device={DEVICE}")

_KV = re.compile(r"(\S+?)=(\S+)")

def dsl_to_rpc(text: str) -> dict:
    body = text.replace("<CMD>", "").replace("</CMD>", "").strip()
    if not body:
        return {}
    method, *rest = body.split()
    params = {k: v.strip('"') for k, v in _KV.findall(" ".join(rest))}
    return {"method": method, "params": params}

@torch.no_grad()
def generate_rpc(utterance: str, max_len: int = 64) -> dict:
    # Tokenize input with T5 tokenizer
    src_encoded = tok.encode(f"<UTT> {utterance} </UTT>", add_special_tokens=False)
    src_ids = torch.tensor([src_encoded], device=DEVICE)
    
    tgt_ids = torch.tensor([[BOS_ID]], device=DEVICE)
    for _ in range(max_len):
        next_id = int(model(src_ids, tgt_ids)[:, -1].argmax(-1))
        tgt_ids = torch.cat(
            [tgt_ids, torch.tensor([[next_id]], device=DEVICE)], dim=1
        )
        if next_id == EOS_ID:
            break
    dsl = tok.decode(tgt_ids.squeeze().tolist(), skip_special_tokens=True)
    return dsl_to_rpc(dsl)

app = Flask(__name__)

@app.route("/infer", methods=["POST"])
def infer():
    utterance = request.json.get("text", "")
    try:
        rpc = generate_rpc(utterance)
        print("INPUT :", utterance)
        print("RPC   :", rpc)
        return jsonify(rpc)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

@app.errorhandler(Exception)
def catch_all(e):
    traceback.print_exc()
    return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    print("Server listening on http://127.0.0.1:6006/infer")
    app.run(host="127.0.0.1", port=6006, threaded=True)
