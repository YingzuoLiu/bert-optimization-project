import os
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from datasets import load_dataset
import matplotlib.pyplot as plt
from pathlib import Path

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Define dataset class
class SST2Dataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        # Encode text
        encoding = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Remove batch dimension
        for key in encoding:
            encoding[key] = encoding[key].squeeze(0)
        
        encoding['labels'] = torch.tensor(label)
        return encoding

def load_and_prepare_data(batch_size=32, max_length=128):
    """Load and prepare the dataset"""
    print("Loading dataset...")
    # Load SST-2 dataset (sentiment analysis task)
    dataset = load_dataset("glue", "sst2")
    train_dataset = dataset["train"]
    validation_dataset = dataset["validation"]
    
    # Load BERT tokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    
    # Create datasets
    print("Preprocessing data...")
    train_texts = train_dataset["sentence"]
    train_labels = train_dataset["label"]
    val_texts = validation_dataset["sentence"]
    val_labels = validation_dataset["label"]
    
    train_dataset = SST2Dataset(train_texts, train_labels, tokenizer, max_length)
    val_dataset = SST2Dataset(val_texts, val_labels, tokenizer, max_length)
    
    # Create DataLoader, optimize data loading
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=0,  # Set to 0 for Windows
        pin_memory=True  # Use pinned memory for faster GPU transfer
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        num_workers=0,
        pin_memory=True
    )
    
    return train_loader, val_loader, tokenizer

def build_model():
    """Build the base BERT model"""
    print("Building BERT classification model...")
    # Use pretrained BERT model
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
    return model

def train_model_mixed_precision(model, train_loader, val_loader, epochs=3, learning_rate=2e-5):
    """Train the model using mixed precision and record the training time"""
    print("Starting mixed precision training...")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model.to(device)
    
    # Create optimizer
    optimizer = AdamW(model.parameters(), lr=learning_rate, no_deprecation_warning=True)
    
    # Create gradient scaler for mixed precision training - note: fixed argument
    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None
    
    # Record start time
    start_time = time.time()
    
    # Training history
    history = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}
    
    # Train model
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, batch in enumerate(train_loader):
            if batch_idx % 100 == 0:
                print(f"  Batch {batch_idx}/{len(train_loader)}")
                
            # Move data to device
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Forward pass with mixed precision
            if device.type == 'cuda':
                with torch.cuda.amp.autocast():
                    outputs = model(**batch)
                    loss = outputs.loss
                
                # Backward pass with gradient scaling
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                # CPU fallback without mixed precision
                outputs = model(**batch)
                loss = outputs.loss
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            predictions = torch.argmax(outputs.logits, dim=-1)
            correct += (predictions == batch['labels']).sum().item()
            total += len(batch['labels'])
        
        # Calculate training metrics
        avg_train_loss = total_loss / len(train_loader)
        train_accuracy = correct / total
        
        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                val_loss += outputs.loss.item()
                predictions = torch.argmax(outputs.logits, dim=-1)
                val_correct += (predictions == batch['labels']).sum().item()
                val_total += len(batch['labels'])
        
        # Calculate validation metrics
        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = val_correct / val_total
        
        # Update history
        history['loss'].append(avg_train_loss)
        history['accuracy'].append(train_accuracy)
        history['val_loss'].append(avg_val_loss)
        history['val_accuracy'].append(val_accuracy)
        
        print(f"  Train loss: {avg_train_loss:.4f}, Train accuracy: {train_accuracy:.4f}")
        print(f"  Validation loss: {avg_val_loss:.4f}, Validation accuracy: {val_accuracy:.4f}")
    
    # Calculate training time
    training_time = time.time() - start_time
    print(f"Training complete! Total training time: {training_time:.2f} seconds")
    
    return model, history, training_time

def evaluate_inference_time(model, val_loader):
    """Evaluate model inference time"""
    print("Evaluating inference performance...")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    # Warm-up
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            if batch_idx >= 2:
                break
            batch = {k: v.to(device) for k, v in batch.items()}
            _ = model(**batch)
    
    # Measure inference time
    start_time = time.time()
    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            _ = model(**batch)
    inference_time = time.time() - start_time
    
    print(f"Inference complete! Total inference time: {inference_time:.2f} seconds")
    return inference_time

def save_model(model, path="./saved_models/bert_mixed_precision"):
    """Save the model"""
    print(f"Saving model to {path}...")
    os.makedirs(path, exist_ok=True)
    model.save_pretrained(path)
    print("Model saved successfully!")
    return path

def plot_training_history(history):
    """Visualize training history"""
    os.makedirs("performance_plots", exist_ok=True)
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    
    plt.subplot(1, 2, 2)
    plt.plot(history['accuracy'])
    plt.plot(history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='lower right')
    
    plt.tight_layout()
    plt.savefig("performance_plots/mixed_precision_training_history.png")
    plt.close()

def main():
    print("Starting mixed precision training optimization test...")
    # Prepare data
    train_loader, val_loader, tokenizer = load_and_prepare_data(batch_size=32)  # Using larger batch size
    
    # Build model
    model = build_model()
    
    # Train with mixed precision
    model, history, training_time = train_model_mixed_precision(model, train_loader, val_loader, epochs=3)
    
    # Evaluate inference time
    inference_time = evaluate_inference_time(model, val_loader)
    
    # Visualize training history
    plot_training_history(history)
    
    # Save model
    model_path = save_model(model)
    
    # Save performance metrics
    performance = {
        "model": "BERT-base-uncased (Mixed Precision)",
        "training_time_seconds": training_time,
        "inference_time_seconds": inference_time,
        "validation_accuracy": history['val_accuracy'][-1],
        "validation_loss": history['val_loss'][-1]
    }
    
    print("\nPerformance Report:")
    for key, value in performance.items():
        print(f"{key}: {value}")
    
    # Ensure directory exists
    os.makedirs("reports", exist_ok=True)
    
    with open("reports/mixed_precision_performance.txt", "w") as f:
        for key, value in performance.items():
            f.write(f"{key}: {value}\n")
    
    print("\nMixed precision training optimization test complete!")
    return performance

if __name__ == "__main__":
    main()