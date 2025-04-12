import os
import boto3
import json
import argparse
import torch
from pathlib import Path
import shutil

def check_aws_credentials():
    """Check if AWS credentials are configured"""
    print("Checking AWS credentials...")
    try:
        # Try to use boto3 to check credentials
        identity = boto3.client('sts').get_caller_identity()
        print(f"✅ AWS credentials found. Account ID: {identity['Account']}")
        return True
    except Exception as e:
        print(f"❌ AWS credentials not found or invalid: {str(e)}")
        return False

def prepare_model_for_deployment(model_path, model_type="pytorch"):
    """Prepare model files for deployment"""
    print(f"Preparing {model_type} model for deployment...")
    
    # Create deployment directory
    deployment_dir = Path("deployment_package")
    if deployment_dir.exists():
        shutil.rmtree(deployment_dir)
    deployment_dir.mkdir(exist_ok=True)
    
    # Check if model path exists
    model_path = Path(model_path)
    if not model_path.exists():
        print(f"❌ Model path {model_path} does not exist")
        return None
    
    # Copy model files to deployment directory
    model_files = list(model_path.glob("*.bin")) + list(model_path.glob("*.json"))
    if not model_files:
        print(f"❌ No model files found in {model_path}")
        return None
    
    for file in model_files:
        shutil.copy(file, deployment_dir)
        print(f"Copied {file.name} to deployment package")
    
    # Create app.py
    create_inference_script(deployment_dir, model_type)
    
    # Create requirements.txt
    create_requirements_file(deployment_dir, model_type)
    
    # Create Dockerfile
    create_dockerfile(deployment_dir, model_type)
    
    print(f"✅ Model deployment package prepared at: {deployment_dir}")
    return deployment_dir

def create_inference_script(deployment_dir, model_type):
    """Create inference service script"""
    app_py = """
import os
import json
import torch
import numpy as np
from flask import Flask, request, jsonify
from transformers import BertTokenizer, BertForSequenceClassification
import time

app = Flask(__name__)

# Initialize global variables
model = None
tokenizer = None
device = None

@app.before_first_request
def load_model():
    global model, tokenizer, device
    
    print("Loading model...")
    start_time = time.time()
    
    # Check for GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load tokenizer and model
    model_path = "."
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = BertForSequenceClassification.from_pretrained(model_path)
    model.to(device)
    model.eval()
    
    print(f"Model loading complete, time: {time.time() - start_time:.2f} seconds")

@app.route('/ping', methods=['GET'])
def ping():
    # SageMaker health check endpoint
    return jsonify({"status": "healthy"}), 200

@app.route('/invocations', methods=['POST'])
def invocations():
    # SageMaker inference endpoint
    if model is None:
        load_model()
    
    # Get input text
    if request.content_type == 'application/json':
        data = json.loads(request.data.decode('utf-8'))
        texts = data.get('texts', [])
        if not texts:
            return jsonify({"error": "Please provide text data in format: {'texts': ['text1', 'text2', ...]}"}), 400
    else:
        return jsonify({"error": "Only application/json content type is supported"}), 415
    
    # Run inference
    try:
        # Preprocessing
        start_time = time.time()
        inputs = tokenizer(texts, padding=True, truncation=True, 
                          max_length=128, return_tensors="pt")
        inputs = {key: val.to(device) for key, val in inputs.items()}
        
        # Inference
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Postprocessing
        scores = torch.nn.functional.softmax(outputs.logits, dim=1).cpu().numpy()
        predictions = np.argmax(scores, axis=1)
        
        # Build response
        results = []
        for i, text in enumerate(texts):
            results.append({
                "text": text,
                "sentiment": "positive" if predictions[i] == 1 else "negative",
                "confidence": float(scores[i][predictions[i]]),
                "inference_time_ms": (time.time() - start_time) * 1000
            })
        
        return jsonify(results)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
"""
    
    # Use UTF-8 encoding when writing the file
    with open(deployment_dir / "app.py", "w", encoding="utf-8") as f:
        f.write(app_py)
    print("✅ Created inference service script app.py")

def create_requirements_file(deployment_dir, model_type):
    """Create requirements.txt file"""
    requirements = """
torch>=1.9.0
transformers>=4.15.0
flask>=2.0.0
numpy>=1.19.0
"""
    
    with open(deployment_dir / "requirements.txt", "w", encoding="utf-8") as f:
        f.write(requirements)
    print("✅ Created requirements.txt")

def create_dockerfile(deployment_dir, model_type):
    """Create Dockerfile"""
    dockerfile = """
FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime

WORKDIR /opt/ml/model

COPY . /opt/ml/model/

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Set environment variables
ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE

# Expose port
EXPOSE 8080

# Run application
ENTRYPOINT ["python", "app.py"]
"""
    
    with open(deployment_dir / "Dockerfile", "w", encoding="utf-8") as f:
        f.write(dockerfile)
    print("✅ Created Dockerfile")

def create_ecr_repository(aws_region, repository_name):
    """Create ECR repository if it doesn't exist"""
    print(f"Creating ECR repository: {repository_name}...")
    
    # Create ECR client
    ecr_client = boto3.client('ecr', region_name=aws_region)
    
    # Create ECR repository (if it doesn't exist)
    try:
        ecr_client.create_repository(repositoryName=repository_name)
        print(f"✅ Created ECR repository: {repository_name}")
    except ecr_client.exceptions.RepositoryAlreadyExistsException:
        print(f"ℹ️ ECR repository already exists: {repository_name}")
    
    return True

def create_manual_instructions(deployment_dir, aws_region, aws_account_id, repository_name):
    """Create instructions file for manual Docker steps"""
    instructions = f"""
# Manual Docker Steps for AWS Deployment

## 1. Navigate to deployment package directory
cd {deployment_dir.absolute()}

## 2. Log in to Amazon ECR
aws ecr get-login-password --region {aws_region} | docker login --username AWS --password-stdin {aws_account_id}.dkr.ecr.{aws_region}.amazonaws.com

## 3. Build Docker image
docker build -t {aws_account_id}.dkr.ecr.{aws_region}.amazonaws.com/{repository_name}:latest .

## 4. Push Docker image to ECR
docker push {aws_account_id}.dkr.ecr.{aws_region}.amazonaws.com/{repository_name}:latest

## 5. After completing the above steps, run the following command to complete deployment:
python aws_deployment_finish.py --aws_region {aws_region} --aws_account_id {aws_account_id} --repository_name {repository_name} --model_name bert-sentiment-analysis --role_arn YOUR_ROLE_ARN
"""
    
    with open(deployment_dir / "docker_instructions.md", "w", encoding="utf-8") as f:
        f.write(instructions)
    
    # Also create a convenience batch file
    batch_file = f"""
@echo off
echo AWS ECR Login...
aws ecr get-login-password --region {aws_region} | docker login --username AWS --password-stdin {aws_account_id}.dkr.ecr.{aws_region}.amazonaws.com

echo Building Docker image...
docker build -t {aws_account_id}.dkr.ecr.{aws_region}.amazonaws.com/{repository_name}:latest .

echo Pushing to ECR...
docker push {aws_account_id}.dkr.ecr.{aws_region}.amazonaws.com/{repository_name}:latest

echo Done! Now run the AWS deployment finish script.
pause
"""
    
    with open(deployment_dir / "build_and_push.bat", "w", encoding="utf-8") as f:
        f.write(batch_file)
    
    print("✅ Created Docker instructions and batch file")

def main():
    parser = argparse.ArgumentParser(description='Prepare BERT model for AWS SageMaker deployment')
    parser.add_argument('--model_path', type=str, default='./saved_models/bert_mixed_precision',
                        help='Model path')
    parser.add_argument('--model_type', type=str, default='pytorch',
                        choices=['pytorch'],
                        help='Model type')
    parser.add_argument('--aws_region', type=str, default='ap-southeast-1',
                        help='AWS region')
    parser.add_argument('--aws_account_id', type=str, required=True,
                        help='AWS account ID')
    parser.add_argument('--repository_name', type=str, default='bert-sentiment-analysis',
                        help='ECR repository name')
    
    args = parser.parse_args()
    
    print("\n===== BERT Model AWS Deployment Preparation =====\n")
    
    # Step 1: Check AWS credentials
    if not check_aws_credentials():
        return
    
    # Step 2: Prepare model deployment package
    deployment_dir = prepare_model_for_deployment(args.model_path, args.model_type)
    if not deployment_dir:
        return
    
    # Step 3: Create ECR repository
    if not create_ecr_repository(args.aws_region, args.repository_name):
        return
    
    # Step 4: Create instructions for manual Docker steps
    create_manual_instructions(deployment_dir, args.aws_region, args.aws_account_id, args.repository_name)
    
    print("\n✅ Deployment preparation complete!")
    print("\n🐳 Manual Docker Steps Required:")
    print("-----------------------------")
    print("Please run the following commands in your command prompt to build and push the Docker image:")
    print(f"cd deployment_package")
    print(f"aws ecr get-login-password --region {args.aws_region} | docker login --username AWS --password-stdin {args.aws_account_id}.dkr.ecr.{args.aws_region}.amazonaws.com")
    print(f"docker build -t {args.aws_account_id}.dkr.ecr.{args.aws_region}.amazonaws.com/{args.repository_name}:latest .")
    print(f"docker push {args.aws_account_id}.dkr.ecr.{args.aws_region}.amazonaws.com/{args.repository_name}:latest")
    print("\nAlternatively, you can run the batch file in the deployment_package directory:")
    print("deployment_package\\build_and_push.bat")
    print("\nAfter completing these steps, run the following command to continue with deployment:")
    print(f"python aws_deployment_finish.py --aws_region {args.aws_region} --aws_account_id {args.aws_account_id} --repository_name {args.repository_name} --model_name bert-sentiment-analysis --role_arn YOUR_ROLE_ARN")

if __name__ == "__main__":
    main()