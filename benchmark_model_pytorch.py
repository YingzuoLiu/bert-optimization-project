import os
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from datasets import load_dataset
import matplotlib.pyplot as plt
from pathlib import Path

# 设置随机种子以确保结果可复现
torch.manual_seed(42)
np.random.seed(42)

# 定义数据集类 - 置于全局作用域
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
        
        # 对文本进行编码
        encoding = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # 移除批处理维度
        for key in encoding:
            encoding[key] = encoding[key].squeeze(0)
        
        encoding['labels'] = torch.tensor(label)
        return encoding

def load_and_prepare_data(batch_size=16, max_length=128):
    """加载并准备数据集"""
    print("加载数据集...")
    # 加载SST-2数据集（情感分析任务）
    dataset = load_dataset("glue", "sst2")
    train_dataset = dataset["train"]
    validation_dataset = dataset["validation"]
    
    # 加载BERT tokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    
    # 创建数据集
    print("预处理数据...")
    train_texts = train_dataset["sentence"]
    train_labels = train_dataset["label"]
    val_texts = validation_dataset["sentence"]
    val_labels = validation_dataset["label"]
    
    train_dataset = SST2Dataset(train_texts, train_labels, tokenizer, max_length)
    val_dataset = SST2Dataset(val_texts, val_labels, tokenizer, max_length)
    
    # 创建DataLoader (Windows上设置num_workers=0以避免多进程问题)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size,
        num_workers=0,
        pin_memory=True
    )
    
    return train_loader, val_loader, tokenizer

def build_model():
    """构建基础BERT模型"""
    print("构建BERT分类模型...")
    # 使用预训练的BERT模型
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
    return model

def train_model(model, train_loader, val_loader, epochs=3, learning_rate=2e-5):
    """训练模型并记录训练时间"""
    print("开始训练模型...")
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    model.to(device)
    
    # 创建优化器 
    optimizer = AdamW(model.parameters(), lr=learning_rate, no_deprecation_warning=True)
    
    # 记录训练开始时间
    start_time = time.time()
    
    # 训练历史
    history = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}
    
    # 训练模型
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, batch in enumerate(train_loader):
            if batch_idx % 100 == 0:
                print(f"  Batch {batch_idx}/{len(train_loader)}")
                
            # 将数据移到设备
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # 前向传播
            outputs = model(**batch)
            loss = outputs.loss
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 统计
            total_loss += loss.item()
            predictions = torch.argmax(outputs.logits, dim=-1)
            correct += (predictions == batch['labels']).sum().item()
            total += len(batch['labels'])
        
        # 计算训练指标
        avg_train_loss = total_loss / len(train_loader)
        train_accuracy = correct / total
        
        # 验证
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
        
        # 计算验证指标
        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = val_correct / val_total
        
        # 更新历史
        history['loss'].append(avg_train_loss)
        history['accuracy'].append(train_accuracy)
        history['val_loss'].append(avg_val_loss)
        history['val_accuracy'].append(val_accuracy)
        
        print(f"  训练损失: {avg_train_loss:.4f}, 训练准确率: {train_accuracy:.4f}")
        print(f"  验证损失: {avg_val_loss:.4f}, 验证准确率: {val_accuracy:.4f}")
    
    # 计算训练时间
    training_time = time.time() - start_time
    print(f"训练完成！总训练时间: {training_time:.2f} 秒")
    
    return model, history, training_time

def evaluate_inference_time(model, val_loader):
    """评估模型推理时间"""
    print("评估推理性能...")
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    # 预热
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            if batch_idx >= 2:
                break
            batch = {k: v.to(device) for k, v in batch.items()}
            _ = model(**batch)
    
    # 测量推理时间
    start_time = time.time()
    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            _ = model(**batch)
    inference_time = time.time() - start_time
    
    print(f"推理完成！总推理时间: {inference_time:.2f} 秒")
    return inference_time

def save_model(model, path="./saved_models/bert_baseline"):
    """保存模型"""
    print(f"保存模型到 {path}...")
    os.makedirs(path, exist_ok=True)
    model.save_pretrained(path)
    print("模型保存完成！")
    return path

def plot_training_history(history):
    """可视化训练历史"""
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
    plt.savefig("performance_plots/training_history.png")
    plt.close()

def main():
    print("开始基准测试...")
    # 准备数据
    train_loader, val_loader, tokenizer = load_and_prepare_data(batch_size=16)
    
    # 构建模型
    model = build_model()
    
    # 训练模型
    model, history, training_time = train_model(model, train_loader, val_loader, epochs=3)
    
    # 评估推理时间
    inference_time = evaluate_inference_time(model, val_loader)
    
    # 可视化训练历史
    plot_training_history(history)
    
    # 保存模型
    model_path = save_model(model)
    
    # 保存性能指标
    performance = {
        "model": "BERT-base-uncased (Baseline)",
        "training_time_seconds": training_time,
        "inference_time_seconds": inference_time,
        "validation_accuracy": history['val_accuracy'][-1],
        "validation_loss": history['val_loss'][-1]
    }
    
    print("\n性能报告:")
    for key, value in performance.items():
        print(f"{key}: {value}")
    
    # 确保目录存在
    os.makedirs("reports", exist_ok=True)
    
    with open("reports/baseline_performance.txt", "w") as f:
        for key, value in performance.items():
            f.write(f"{key}: {value}\n")
    
    print("\n基准测试完成!")
    return performance

if __name__ == "__main__":
    main()