#!/usr/bin/env python3

import json
import re
import traceback
from pathlib import Path

import torch
from flask import Flask, jsonify, request
from transformers import T5Tokenizer, T5ForConditionalGeneration

# Model paths
MODEL_PATH = Path("./t5-finetuned")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load fine-tuned model and tokenizer
print(f"Loading fine-tuned T5 model from {MODEL_PATH}...")
tokenizer = T5Tokenizer.from_pretrained(MODEL_PATH)
model = T5ForConditionalGeneration.from_pretrained(MODEL_PATH).to(DEVICE)
model.eval()

print(f"Model loaded successfully • device={DEVICE}")

# RPC parsing helper
_KV = re.compile(r"(\S+?)=(\S+)")

def dsl_to_rpc(text: str) -> dict:
    """Convert model output back to RPC format"""
    if not text.strip():
        return {}
    
    parts = text.strip().split()
    if not parts:
        return {}
    
    method = parts[0]
    params = {}
    
    # Parse key=value pairs
    for part in parts[1:]:
        match = _KV.match(part)
        if match:
            key, value = match.groups()
            params[key] = value.strip('"')
    
    return {"method": method, "params": params}

@torch.no_grad()
def generate_rpc(utterance: str) -> dict:
    """Generate RPC from utterance using fine-tuned T5"""
    # Add prefix for T5
    input_text = f"command: {utterance}"
    
    # Tokenize input
    inputs = tokenizer(
        input_text, 
        return_tensors="pt", 
        max_length=128, 
        truncation=True,
        padding=True
    ).to(DEVICE)
    
    # Generate output
    outputs = model.generate(
        inputs.input_ids,
        max_length=64,
        num_beams=4,
        early_stopping=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    
    # Decode output
    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Convert to RPC format
    return dsl_to_rpc(output_text)

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
    app.run(host="127.0.0.1", port=6006, threaded=True) 
