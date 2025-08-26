import json
import torch
from datasets import Dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split

def load_dataset(file_path):
    """Load JSONL dataset"""
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return Dataset.from_list(data)

def preprocess_function(examples, tokenizer, max_length=128):
    """Tokenize inputs and outputs"""
    # Add prefix for T5
    inputs = [f"command: {x}" for x in examples["input"]]
    targets = examples["output"]
    
    model_inputs = tokenizer(
        inputs, 
        max_length=max_length, 
        truncation=True, 
        padding="max_length"
    )
    labels = tokenizer(
        targets, 
        max_length=max_length, 
        truncation=True, 
        padding="max_length"
    )
    
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def main():
    print("Loading datasets...")
    train_dataset = load_dataset("finetune_train.jsonl")
    val_dataset = load_dataset("finetune_val.jsonl")
    
    print(f"Train examples: {len(train_dataset)}")
    print(f"Validation examples: {len(val_dataset)}")
    
    # Load tokenizer and model
    model_name = "t5-small"  
    print(f"Loading {model_name}...")
    
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    
    # Add special tokens
    special_tokens = ["<PAD>", "<BOS>", "<EOS>"]
    tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
    model.resize_token_embeddings(len(tokenizer))
    
    print("Preprocessing datasets...")
    # Preprocess datasets
    train_dataset = train_dataset.map(
        lambda x: preprocess_function(x, tokenizer), 
        batched=True,
        remove_columns=train_dataset.column_names
    )
    val_dataset = val_dataset.map(
        lambda x: preprocess_function(x, tokenizer), 
        batched=True,
        remove_columns=val_dataset.column_names
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir="./t5-finetuned",
        evaluation_strategy="epoch",
        learning_rate=3e-4,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        save_strategy="epoch",
        logging_dir="./logs",
        logging_steps=100,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
    )
    
    print("Starting training...")
    trainer.train()
    
    print("Saving model...")
    trainer.save_model("./t5-finetuned")
    tokenizer.save_pretrained("./t5-finetuned")
    
    print("Fine-tuning complete! Model saved to ./t5-finetuned/")

if __name__ == "__main__":
    main() 