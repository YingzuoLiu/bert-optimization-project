import boto3
import argparse
import time
import sys
from botocore.exceptions import ClientError

def cleanup_sagemaker_resources(aws_region, model_name):
    """清理所有SageMaker资源"""
    print(f"\n正在清理SageMaker资源 (前缀: {model_name})...")
    
    # 创建SageMaker客户端
    sm_client = boto3.client('sagemaker', region_name=aws_region)
    
    # 1. 删除端点
    endpoint_name = f"{model_name}-endpoint"
    print(f"正在删除端点: {endpoint_name}")
    try:
        sm_client.delete_endpoint(EndpointName=endpoint_name)
        print(f"✅ 端点已删除: {endpoint_name}")
    except ClientError as e:
        if "ValidationError" in str(e) and "Could not find endpoint" in str(e):
            print(f"ℹ️ 端点不存在: {endpoint_name}")
        else:
            print(f"⚠️ 删除端点时出错: {str(e)}")
    
    # 等待端点删除完成
    print("等待端点删除操作完成...")
    time.sleep(10)
    
    # 2. 删除端点配置
    endpoint_config_name = f"{model_name}-config"
    print(f"正在删除端点配置: {endpoint_config_name}")
    try:
        sm_client.delete_endpoint_config(EndpointConfigName=endpoint_config_name)
        print(f"✅ 端点配置已删除: {endpoint_config_name}")
    except ClientError as e:
        if "ValidationError" in str(e) and "Could not find endpoint configuration" in str(e):
            print(f"ℹ️ 端点配置不存在: {endpoint_config_name}")
        else:
            print(f"⚠️ 删除端点配置时出错: {str(e)}")
    
    # 3. 删除模型
    print(f"正在删除模型: {model_name}")
    try:
        sm_client.delete_model(ModelName=model_name)
        print(f"✅ 模型已删除: {model_name}")
    except ClientError as e:
        if "ValidationError" in str(e) and "Could not find model" in str(e):
            print(f"ℹ️ 模型不存在: {model_name}")
        else:
            print(f"⚠️ 删除模型时出错: {str(e)}")
    
    print("SageMaker资源清理完成!")

def cleanup_ecr_resources(aws_region, repository_name):
    """清理ECR资源"""
    print(f"\n正在清理ECR资源 (仓库: {repository_name})...")
    
    # 创建ECR客户端
    ecr_client = boto3.client('ecr', region_name=aws_region)
    
    # 检查仓库是否存在
    try:
        response = ecr_client.describe_repositories(repositoryNames=[repository_name])
        repositories = response['repositories']
        
        if repositories:
            # 获取所有镜像
            print(f"正在查找 {repository_name} 仓库中的镜像...")
            try:
                images = ecr_client.list_images(repositoryName=repository_name)['imageIds']
                
                if images:
                    # 删除所有镜像
                    print(f"发现 {len(images)} 个镜像，正在删除...")
                    ecr_client.batch_delete_image(
                        repositoryName=repository_name,
                        imageIds=images
                    )
                    print("✅ 所有镜像已删除")
                else:
                    print("ℹ️ 仓库中没有镜像")
                
                # 删除仓库
                print(f"正在删除仓库: {repository_name}")
                ecr_client.delete_repository(repositoryName=repository_name)
                print(f"✅ 仓库已删除: {repository_name}")
            
            except ClientError as e:
                print(f"⚠️ 删除镜像或仓库时出错: {str(e)}")
        else:
            print(f"ℹ️ 仓库不存在: {repository_name}")
    
    except ClientError as e:
        if "RepositoryNotFoundException" in str(e):
            print(f"ℹ️ 仓库不存在: {repository_name}")
        else:
            print(f"⚠️ 检查仓库时出错: {str(e)}")
    
    print("ECR资源清理完成!")

def cleanup_cloudwatch_logs(aws_region, model_name):
    """清理CloudWatch日志"""
    print(f"\n正在清理CloudWatch日志 (前缀: {model_name})...")
    
    # 创建CloudWatch Logs客户端
    logs_client = boto3.client('logs', region_name=aws_region)
    
    try:
        # 获取所有日志组
        log_groups = logs_client.describe_log_groups()['logGroups']
        
        # 搜索相关的日志组
        for log_group in log_groups:
            log_group_name = log_group['logGroupName']
            
            # 检查是否是SageMaker日志组并且与模型名称相关
            if '/aws/sagemaker/' in log_group_name and model_name in log_group_name:
                print(f"正在删除日志组: {log_group_name}")
                logs_client.delete_log_group(logGroupName=log_group_name)
                print(f"✅ 日志组已删除: {log_group_name}")
    
    except ClientError as e:
        print(f"⚠️ 删除日志组时出错: {str(e)}")
    
    print("CloudWatch日志清理完成!")

def main():
    parser = argparse.ArgumentParser(description='清理AWS中部署的BERT资源')
    parser.add_argument('--aws_region', type=str, default='us-east-1',
                        help='AWS区域')
    parser.add_argument('--model_name', type=str, default='bert-sentiment-analysis',
                        help='模型名称/前缀')
    parser.add_argument('--repository_name', type=str, default='bert-sentiment-analysis',
                        help='ECR仓库名称')
    parser.add_argument('--skip_confirm', action='store_true',
                        help='跳过确认提示')
    
    args = parser.parse_args()
    
    if not args.skip_confirm:
        print(f"\n⚠️  警告: 这将删除以下AWS资源:")
        print(f"   - SageMaker端点: {args.model_name}-endpoint")
        print(f"   - SageMaker端点配置: {args.model_name}-config")
        print(f"   - SageMaker模型: {args.model_name}")
        print(f"   - ECR仓库及其所有镜像: {args.repository_name}")
        print(f"   - 相关的CloudWatch日志组")
        print("\n这些资源将被永久删除且无法恢复!")
        
        confirm = input("\n请输入 'yes' 确认删除操作: ")
        if confirm.lower() != 'yes':
            print("操作已取消")
            sys.exit(0)
    
    # 执行清理
    cleanup_sagemaker_resources(args.aws_region, args.model_name)
    cleanup_ecr_resources(args.aws_region, args.repository_name)
    cleanup_cloudwatch_logs(args.aws_region, args.model_name)
    
    print("\n✅ AWS资源清理完成！所有部署的资源已被删除。")

if __name__ == "__main__":
    main()