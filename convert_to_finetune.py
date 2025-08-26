import json
import re
from pathlib import Path

def flatten_rpc(rpc):
    """Convert RPC dict to flat string format for fine-tuning"""
    method = rpc.get("method", "")
    params = rpc.get("params", {})
    
    if not params:
        return method
    
    param_str = " ".join([f"{k}={v}" for k, v in params.items()])
    return f"{method} {param_str}".strip()

def convert_dataset(input_file, output_file):
    """Convert dataset from utterance/rpc format to input/output format"""
    with open(input_file, 'r') as f:
        data = [json.loads(line) for line in f]
    
    converted = []
    for item in data:
        utterance = item.get("utterance", "")
        rpc = item.get("rpc", {})
        
        if utterance and rpc:
            converted.append({
                "input": utterance,
                "output": flatten_rpc(rpc)
            })
    
    with open(output_file, 'w') as f:
        for item in converted:
            f.write(json.dumps(item) + '\n')
    
    print(f"Converted {len(converted)} examples to {output_file}")

if __name__ == "__main__":
    # Convert your dataset
    convert_dataset("data/dataset.jsonl", "finetune_data.jsonl")
    
    # Split into train/val (90/10 split)
    with open("finetune_data.jsonl", 'r') as f:
        all_data = [json.loads(line) for line in f]
    
    # Simple split
    split_idx = int(len(all_data) * 0.9)
    train_data = all_data[:split_idx]
    val_data = all_data[split_idx:]
    
    with open("finetune_train.jsonl", 'w') as f:
        for item in train_data:
            f.write(json.dumps(item) + '\n')
    
    with open("finetune_val.jsonl", 'w') as f:
        for item in val_data:
            f.write(json.dumps(item) + '\n')
    
    print(f"Split into {len(train_data)} train and {len(val_data)} validation examples") 