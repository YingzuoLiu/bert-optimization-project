import os
import boto3
import time
import json
import argparse
import torch
from pathlib import Path
from transformers import BertTokenizer
import subprocess
import shutil

def check_aws_cli():
    """检查AWS CLI是否已安装并配置"""
    print("检查AWS CLI配置...")
    try:
        # 尝试调用AWS CLI命令
        result = subprocess.run(["aws", "sts", "get-caller-identity"], 
                               capture_output=True, text=True, check=False)
        
        if result.returncode != 0:
            print("⚠️ AWS CLI未正确配置。请运行 'aws configure' 配置凭证。")
            print(f"错误信息: {result.stderr}")
            return False
            
        identity = json.loads(result.stdout)
        print(f"✅ AWS CLI已配置。账户ID: {identity['Account']}")
        return True
    except FileNotFoundError:
        print("⚠️ 未找到AWS CLI。请安装AWS CLI并运行 'aws configure' 配置凭证。")
        # 尝试通过boto3直接检查凭据
        try:
            boto3.client('sts').get_caller_identity()
            print("但是找到了有效的AWS凭证，可以继续。")
            return True
        except:
            return False

def check_dependencies():
    """检查部署所需的依赖项"""
    print("检查依赖项...")
    dependencies = {
        "torch": "PyTorch",
        "boto3": "Boto3 (AWS SDK)",
        "transformers": "Transformers"
    }
    
    missing = []
    for module, name in dependencies.items():
        try:
            __import__(module)
            print(f"✅ 已安装 {name}")
        except ImportError:
            print(f"❌ 未安装 {name}")
            missing.append(module)
    
    if missing:
        print("\n请安装缺失的依赖项:")
        print(f"pip install {' '.join(missing)}")
        return False
    
    return True

def prepare_model_for_deployment(model_path, model_type="pytorch"):
    """准备模型文件用于部署"""
    print(f"准备{model_type}模型用于部署...")
    
    # 创建部署目录
    deployment_dir = Path("deployment_package")
    if deployment_dir.exists():
        shutil.rmtree(deployment_dir)
    deployment_dir.mkdir(exist_ok=True)
    
    # 确认模型路径存在
    model_path = Path(model_path)
    if not model_path.exists():
        print(f"❌ 模型路径 {model_path} 不存在")
        return None
    
    # 复制模型文件到部署目录
    model_files = list(model_path.glob("*.bin")) + list(model_path.glob("*.json"))
    if not model_files:
        print(f"❌ 在 {model_path} 中未找到模型文件")
        return None
    
    for file in model_files:
        shutil.copy(file, deployment_dir)
        print(f"复制 {file.name} 到部署包")
    
    # 创建app.py
    create_inference_script(deployment_dir, model_type)
    
    # 创建requirements.txt
    create_requirements_file(deployment_dir, model_type)
    
    # 创建Dockerfile
    create_dockerfile(deployment_dir, model_type)
    
    print(f"✅ 模型部署包已准备完毕，位于: {deployment_dir}")
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

def build_and_push_docker_image(deployment_dir, aws_region, aws_account_id, repository_name):
    """构建并推送Docker镜像到ECR"""
    print("构建并推送Docker镜像到ECR...")
    
    # 创建ECR客户端
    ecr_client = boto3.client('ecr', region_name=aws_region)
    
    # 创建ECR仓库（如果不存在）
    try:
        ecr_client.create_repository(repositoryName=repository_name)
        print(f"✅ 已创建ECR仓库: {repository_name}")
    except ecr_client.exceptions.RepositoryAlreadyExistsException:
        print(f"ℹ️ ECR仓库已存在: {repository_name}")
    
    # 获取登录令牌
    login_command = subprocess.run(
        ["aws", "ecr", "get-login-password", "--region", aws_region],
        capture_output=True, text=True, check=True
    ).stdout.strip()
    
    # 登录到ECR
    registry_url = f"{aws_account_id}.dkr.ecr.{aws_region}.amazonaws.com"
    subprocess.run(
        ["docker", "login", "--username", "AWS", "--password", login_command, registry_url],
        capture_output=True, text=True, check=True
    )
    print("✅ 已登录到ECR")
    
    # 构建Docker镜像
    image_tag = f"{registry_url}/{repository_name}:latest"
    print(f"正在构建Docker镜像: {image_tag}")
    subprocess.run(
        ["docker", "build", "-t", image_tag, str(deployment_dir)],
        check=True
    )
    
    # 推送Docker镜像到ECR
    print(f"正在推送镜像到ECR: {image_tag}")
    subprocess.run(
        ["docker", "push", image_tag],
        check=True
    )
    
    print(f"✅ Docker镜像已推送到: {image_tag}")
    return image_tag

def deploy_to_sagemaker(aws_region, model_name, image_uri, role_arn):
    """部署模型到SageMaker"""
    print("部署模型到SageMaker...")
    
    # 创建SageMaker客户端
    sagemaker_client = boto3.client('sagemaker', region_name=aws_region)
    
    # 创建模型
    try:
        response = sagemaker_client.create_model(
            ModelName=model_name,
            PrimaryContainer={
                'Image': image_uri,
                'Mode': 'SingleModel',
            },
            ExecutionRoleArn=role_arn
        )
        print(f"✅ 已创建SageMaker模型: {model_name}")
    except Exception as e:
        print(f"❌ 创建模型时出错: {str(e)}")
        return None
    
    # 创建端点配置
    endpoint_config_name = f"{model_name}-config"
    try:
        response = sagemaker_client.create_endpoint_config(
            EndpointConfigName=endpoint_config_name,
            ProductionVariants=[
                {
                    'VariantName': 'AllTraffic',
                    'ModelName': model_name,
                    'InstanceType': 'ml.c5.large',  # 更经济的实例类型
                    'InitialInstanceCount': 1
                }
            ]
        )
        print(f"✅ 已创建端点配置: {endpoint_config_name}")
    except Exception as e:
        print(f"❌ 创建端点配置时出错: {str(e)}")
        return None
    
    # 创建端点
    endpoint_name = f"{model_name}-endpoint"
    try:
        response = sagemaker_client.create_endpoint(
            EndpointName=endpoint_name,
            EndpointConfigName=endpoint_config_name
        )
        print(f"正在创建端点: {endpoint_name}，这可能需要几分钟时间...")
        
        # 等待端点创建完成
        print("等待端点就绪...")
        waiter = sagemaker_client.get_waiter('endpoint_in_service')
        waiter.wait(EndpointName=endpoint_name)
        print(f"✅ 端点已创建并处于活动状态: {endpoint_name}")
    except Exception as e:
        print(f"❌ 创建端点时出错: {str(e)}")
        return None
    
    return endpoint_name

def test_endpoint(aws_region, endpoint_name):
    """测试SageMaker端点"""
    print(f"测试SageMaker端点: {endpoint_name}...")
    
    # 创建SageMaker运行时客户端
    runtime_client = boto3.client('sagemaker-runtime', region_name=aws_region)
    
    # 准备测试数据
    test_data = {
        "texts": [
            "This movie was fantastic and I really enjoyed it!",
            "I disliked this product, it was poorly made."
        ]
    }
    
    # 调用端点
    try:
        response = runtime_client.invoke_endpoint(
            EndpointName=endpoint_name,
            ContentType='application/json',
            Body=json.dumps(test_data)
        )
        
        # 解析结果
        result = json.loads(response['Body'].read().decode())
        
        print("\n测试结果:")
        for item in result:
            print(f"文本: {item['text']}")
            print(f"情感: {item['sentiment']}")
            print(f"置信度: {item['confidence']:.4f}")
            print(f"推理时间: {item['inference_time_ms']:.2f} ms")
            print()
        
        print("✅ 端点测试成功！")
        print(f"端点URL: https://{aws_region}.console.aws.amazon.com/sagemaker/home?region={aws_region}#/endpoints/{endpoint_name}")
        return True
    except Exception as e:
        print(f"❌ 测试端点时出错: {str(e)}")
        return False

def cleanup_resources(aws_region, model_name, repository_name):
    """清理所有创建的AWS资源"""
    print("\n要清理所有AWS资源，请运行以下命令：")
    print(f"python aws_cleanup.py --aws_region {aws_region} --model_name {model_name} --repository_name {repository_name}")
    print("\n这将删除以下资源以避免产生不必要的费用：")
    print(f" - SageMaker端点: {model_name}-endpoint")
    print(f" - SageMaker端点配置: {model_name}-config")
    print(f" - SageMaker模型: {model_name}")
    print(f" - ECR仓库: {repository_name}")

def main():
    parser = argparse.ArgumentParser(description='将优化的BERT模型部署到AWS SageMaker')
    parser.add_argument('--model_path', type=str, default='./saved_models/bert_mixed_precision',
                        help='模型路径')
    parser.add_argument('--model_type', type=str, default='pytorch',
                        choices=['pytorch'],
                        help='模型类型')
    parser.add_argument('--aws_region', type=str, default='us-east-1',
                        help='AWS区域')
    parser.add_argument('--aws_account_id', type=str, required=True,
                        help='AWS账户ID')
    parser.add_argument('--repository_name', type=str, default='bert-sentiment-analysis',
                        help='ECR仓库名称')
    parser.add_argument('--model_name', type=str, default='bert-sentiment-analysis',
                        help='SageMaker模型名称')
    parser.add_argument('--role_arn', type=str, required=True,
                        help='SageMaker IAM角色ARN')
    
    args = parser.parse_args()
    
    print("\n===== BERT模型AWS部署工具 =====\n")
    
    # 步骤1: 检查AWS CLI和依赖项
    if not check_aws_cli() or not check_dependencies():
        return
    
    # 步骤2: 准备模型部署包
    deployment_dir = prepare_model_for_deployment(args.model_path, args.model_type)
    if not deployment_dir:
        return
    
    # 步骤3: 构建并推送Docker镜像
    try:
        image_uri = build_and_push_docker_image(
            deployment_dir, 
            args.aws_region, 
            args.aws_account_id, 
            args.repository_name
        )
    except Exception as e:
        print(f"❌ 构建或推送Docker镜像时出错: {str(e)}")
        return
    
    # 步骤4: 部署到SageMaker
    endpoint_name = deploy_to_sagemaker(
        args.aws_region,
        args.model_name,
        image_uri,
        args.role_arn
    )
    
    if endpoint_name:
        # 步骤5: 测试端点
        test_endpoint(args.aws_region, endpoint_name)
        
        # 步骤6: 清理资源说明
        cleanup_resources(args.aws_region, args.model_name, args.repository_name)
        
        print("\n✅ 部署流程已完成！")
    else:
        print("\n❌ 部署失败。请检查错误信息并重试。")

if __name__ == "__main__":
    main()