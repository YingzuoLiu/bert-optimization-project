import boto3
import time
import json
import argparse

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

def check_ecr_image(aws_region, aws_account_id, repository_name):
    """Check if the Docker image exists in ECR"""
    print(f"Checking for Docker image in ECR repository: {repository_name}...")
    
    # Create ECR client
    ecr_client = boto3.client('ecr', region_name=aws_region)
    
    try:
        # Check if repository exists
        repositories = ecr_client.describe_repositories(repositoryNames=[repository_name])['repositories']
        if not repositories:
            print(f"❌ ECR repository {repository_name} not found")
            return False
        
        # Check for images in repository
        images = ecr_client.list_images(repositoryName=repository_name)['imageIds']
        if not images:
            print(f"❌ No images found in repository {repository_name}")
            return False
        
        print(f"✅ Found {len(images)} images in ECR repository")
        return True
    except ecr_client.exceptions.RepositoryNotFoundException:
        print(f"❌ ECR repository {repository_name} not found")
        return False
    except Exception as e:
        print(f"❌ Error checking ECR image: {str(e)}")
        return False

def deploy_to_sagemaker(aws_region, model_name, aws_account_id, repository_name, role_arn):
    """Deploy model to SageMaker"""
    print("Deploying model to SageMaker...")
    
    # Create SageMaker client
    sagemaker_client = boto3.client('sagemaker', region_name=aws_region)
    
    # Set image URI
    image_uri = f"{aws_account_id}.dkr.ecr.{aws_region}.amazonaws.com/{repository_name}:latest"
    
    # Create model
    try:
        response = sagemaker_client.create_model(
            ModelName=model_name,
            PrimaryContainer={
                'Image': image_uri,
                'Mode': 'SingleModel',
            },
            ExecutionRoleArn=role_arn
        )
        print(f"✅ Created SageMaker model: {model_name}")
    except Exception as e:
        print(f"❌ Error creating model: {str(e)}")
        return None
    
    # Create endpoint config
    endpoint_config_name = f"{model_name}-config"
    try:
        response = sagemaker_client.create_endpoint_config(
            EndpointConfigName=endpoint_config_name,
            ProductionVariants=[
                {
                    'VariantName': 'AllTraffic',
                    'ModelName': model_name,
                    'InstanceType': 'ml.c5.large',  # More economical instance type
                    'InitialInstanceCount': 1
                }
            ]
        )
        print(f"✅ Created endpoint configuration: {endpoint_config_name}")
    except Exception as e:
        print(f"❌ Error creating endpoint configuration: {str(e)}")
        return None
    
    # Create endpoint
    endpoint_name = f"{model_name}-endpoint"
    try:
        response = sagemaker_client.create_endpoint(
            EndpointName=endpoint_name,
            EndpointConfigName=endpoint_config_name
        )
        print(f"Creating endpoint: {endpoint_name}, this may take several minutes...")
        
        # Wait for endpoint to be ready
        print("Waiting for endpoint to be ready...")
        waiter = sagemaker_client.get_waiter('endpoint_in_service')
        waiter.wait(EndpointName=endpoint_name)
        print(f"✅ Endpoint created and active: {endpoint_name}")
    except Exception as e:
        print(f"❌ Error creating endpoint: {str(e)}")
        return None
    
    return endpoint_name

def test_endpoint(aws_region, endpoint_name):
    """Test SageMaker endpoint"""
    print(f"Testing SageMaker endpoint: {endpoint_name}...")
    
    # Create SageMaker runtime client
    runtime_client = boto3.client('sagemaker-runtime', region_name=aws_region)
    
    # Prepare test data
    test_data = {
        "texts": [
            "This movie was fantastic and I really enjoyed it!",
            "I disliked this product, it was poorly made."
        ]
    }
    
    # Call endpoint
    try:
        response = runtime_client.invoke_endpoint(
            EndpointName=endpoint_name,
            ContentType='application/json',
            Body=json.dumps(test_data)
        )
        
        # Parse result
        result = json.loads(response['Body'].read().decode())
        
        print("\nTest Results:")
        for item in result:
            print(f"Text: {item['text']}")
            print(f"Sentiment: {item['sentiment']}")
            print(f"Confidence: {item['confidence']:.4f}")
            print(f"Inference time: {item['inference_time_ms']:.2f} ms")
            print()
        
        print("✅ Endpoint test successful!")
        print(f"Endpoint URL: https://{aws_region}.console.aws.amazon.com/sagemaker/home?region={aws_region}#/endpoints/{endpoint_name}")
        return True
    except Exception as e:
        print(f"❌ Error testing endpoint: {str(e)}")
        return False

def provide_cleanup_instructions(aws_region, model_name, repository_name):
    """Provide instructions for cleaning up AWS resources"""
    print("\nTo clean up all AWS resources, run the following command:")
    print(f"python aws_cleanup.py --aws_region {aws_region} --model_name {model_name} --repository_name {repository_name}")
    print("\nThis will delete the following resources to avoid unnecessary charges:")
    print(f" - SageMaker endpoint: {model_name}-endpoint")
    print(f" - SageMaker endpoint configuration: {model_name}-config")
    print(f" - SageMaker model: {model_name}")
    print(f" - ECR repository: {repository_name}")

def main():
    parser = argparse.ArgumentParser(description='Complete AWS SageMaker deployment after Docker image push')
    parser.add_argument('--aws_region', type=str, default='ap-southeast-1',
                        help='AWS region')
    parser.add_argument('--aws_account_id', type=str, required=True,
                        help='AWS account ID')
    parser.add_argument('--repository_name', type=str, default='bert-sentiment-analysis',
                        help='ECR repository name')
    parser.add_argument('--model_name', type=str, default='bert-sentiment-analysis',
                        help='SageMaker model name')
    parser.add_argument('--role_arn', type=str, required=True,
                        help='SageMaker IAM role ARN')
    
    args = parser.parse_args()
    
    print("\n===== BERT Model AWS Deployment Completion =====\n")
    
    # Step 1: Check AWS credentials
    if not check_aws_credentials():
        return
    
    # Step 2: Check if Docker image exists in ECR
    if not check_ecr_image(args.aws_region, args.aws_account_id, args.repository_name):
        print("Please make sure you have built and pushed the Docker image to ECR.")
        print(f"Run the instructions in deployment_package/docker_instructions.md")
        return
    
    # Step 3: Deploy to SageMaker
    endpoint_name = deploy_to_sagemaker(
        args.aws_region,
        args.model_name,
        args.aws_account_id,
        args.repository_name,
        args.role_arn
    )
    
    if endpoint_name:
        # Step 4: Test endpoint
        test_endpoint(args.aws_region, endpoint_name)
        
        # Step 5: Provide cleanup instructions
        provide_cleanup_instructions(args.aws_region, args.model_name, args.repository_name)
        
        print("\n✅ Deployment process completed!")
    else:
        print("\n❌ Deployment failed. Please check error messages and try again.")
        print("You may need to clean up any partially created resources:")
        print(f"python aws_cleanup.py --aws_region {args.aws_region} --model_name {args.model_name} --repository_name {args.repository_name}")

if __name__ == "__main__":
    main()