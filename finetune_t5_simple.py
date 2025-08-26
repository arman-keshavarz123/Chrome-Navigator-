import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration
import os

class RPCDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=128):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Load data
        with open(file_path, 'r') as f:
            self.data = [json.loads(line) for line in f]
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        input_text = f"command: {item['input']}"
        target_text = item['output']
        
        # Tokenize input and target
        inputs = self.tokenizer(
            input_text, 
            max_length=self.max_length, 
            truncation=True, 
            padding='max_length',
            return_tensors='pt'
        )
        
        targets = self.tokenizer(
            target_text, 
            max_length=self.max_length, 
            truncation=True, 
            padding='max_length',
            return_tensors='pt'
        )
        
        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'labels': targets['input_ids'].squeeze()
        }

def train_model(model, train_loader, val_loader, num_epochs=3, device='cpu'):
    """Simple training loop without Trainer"""
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        num_batches = 0
        
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        # Validation
        model.eval()
        val_loss = 0
        val_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                val_loss += outputs.loss.item()
                val_batches += 1
        
        avg_train_loss = total_loss / num_batches
        avg_val_loss = val_loss / val_batches
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"  Training Loss: {avg_train_loss:.4f}")
        print(f"  Validation Loss: {avg_val_loss:.4f}")
        print()

def main():
    print("Loading datasets...")
    
    # Load tokenizer and model
    model_name = "t5-small"
    print(f"Loading {model_name}...")
    
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    
    # Add special tokens if needed
    special_tokens = ["<PAD>", "<BOS>", "<EOS>"]
    tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
    model.resize_token_embeddings(len(tokenizer))
    
    # Create datasets
    train_dataset = RPCDataset("finetune_train.jsonl", tokenizer)
    val_dataset = RPCDataset("finetune_val.jsonl", tokenizer)
    
    print(f"Train examples: {len(train_dataset)}")
    print(f"Validation examples: {len(val_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    print(f"Training on device: {device}")
    print("Starting training...")
    
    # Train the model
    train_model(model, train_loader, val_loader, num_epochs=3, device=device)
    
    # Save the model
    print("Saving model...")
    os.makedirs("./t5-finetuned", exist_ok=True)
    model.save_pretrained("./t5-finetuned")
    tokenizer.save_pretrained("./t5-finetuned")
    
    print("Fine-tuning complete! Model saved to ./t5-finetuned/")

if __name__ == "__main__":
    main() 