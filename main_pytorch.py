import os
import argparse
import time
import subprocess
import sys
from pathlib import Path

def run_command(command, description):
    """Run command and display progress"""
    print(f"\n{'='*80}")
    print(f"Starting: {description}")
    print(f"{'='*80}")
    
    try:
        process = subprocess.run(command, shell=True, check=True)
        print(f"\n✅ Completed: {description}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Failed: {description}")
        print(f"Error: {e}")
        return False

def create_directory_structure():
    """Create project directory structure"""
    print("\nCreating project directory structure...")
    
    directories = [
        "data",
        "saved_models",
        "performance_plots",
        "reports",
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"Created directory: {directory}")
    
    print("✅ Directory structure created")

def main():
    parser = argparse.ArgumentParser(description='BERT Model Optimization and Deployment Pipeline')
    parser.add_argument('--skip_baseline', action='store_true',
                        help='Skip baseline test step')
    parser.add_argument('--skip_mixed_precision', action='store_true',
                        help='Skip mixed precision training step')
    parser.add_argument('--skip_tensorrt', action='store_true',
                        help='Skip TensorRT optimization step')
    parser.add_argument('--skip_aws_deployment', action='store_true',
                        help='Skip AWS deployment step')
    parser.add_argument('--manual_docker', action='store_true',
                        help='Prepare Docker files but run Docker commands manually')
    parser.add_argument('--aws_account_id', type=str,
                        help='AWS account ID (for deployment)')
    parser.add_argument('--aws_role_arn', type=str,
                        help='AWS IAM role ARN (for deployment)')
    parser.add_argument('--aws_region', type=str, default='ap-southeast-1',
                        help='AWS region (for deployment)')
    parser.add_argument('--cleanup_aws', action='store_true',
                        help='Clean up AWS resources')
    parser.add_argument('--model_name', type=str, default='bert-sentiment-analysis',
                        help='Model name for deployment')
    parser.add_argument('--repository_name', type=str, default='bert-sentiment-analysis',
                        help='ECR repository name for deployment')
    
    args = parser.parse_args()
    
    # If only cleaning up AWS resources, do that and exit
    if args.cleanup_aws:
        cleanup_cmd = (f"python aws_cleanup.py "
                     f"--aws_region {args.aws_region} "
                     f"--model_name {args.model_name} "
                     f"--repository_name {args.repository_name}")
        
        success = run_command(cleanup_cmd, "Cleaning up AWS resources")
        if not success:
            print("AWS resource cleanup failed.")
        return
    
    # Record start time
    start_time = time.time()
    
    # Step 0: Create directory structure
    create_directory_structure()
    
    # Step 1: Run baseline test
    if not args.skip_baseline:
        success = run_command("python benchmark_model_pytorch.py", "Baseline BERT model training and testing")
        if not success:
            print("Baseline test failed, but continuing with other steps.")
    
    # Step 2: Run mixed precision optimization
    if not args.skip_mixed_precision:
        success = run_command("python optimization_mixed_precision_pytorch.py", "Mixed precision training optimization")
        if not success:
            print("Mixed precision optimization failed, but continuing with other steps.")
    
    # Step 3: Run ONNX and TensorRT optimization
    if not args.skip_tensorrt:
        success = run_command("python onnx_tensorrt_pytorch.py", "ONNX and TensorRT optimization")
        if not success:
            print("ONNX and TensorRT optimization failed, but continuing with other steps.")
    
    # Step 4: Run performance comparison
    success = run_command("python performance_comparison_pytorch.py", "Performance comparison analysis")
    if not success:
        print("Performance comparison analysis failed, but continuing with other steps.")
    
    # Step 5: AWS deployment (if specified)
    if not args.skip_aws_deployment:
        if args.aws_account_id and args.aws_role_arn:
            # If manual Docker handling is requested
            if args.manual_docker:
                # First, prepare deployment package
                deploy_prepare_cmd = (f"python aws_deployment_prepare.py "
                                   f"--model_path ./saved_models/bert_mixed_precision "
                                   f"--aws_region {args.aws_region} "
                                   f"--aws_account_id {args.aws_account_id} "
                                   f"--repository_name {args.repository_name}")
                
                success = run_command(deploy_prepare_cmd, "Preparing model for AWS deployment")
                if not success:
                    print("AWS deployment preparation failed.")
                    
                # Print instructions for manual Docker commands
                print("\n🐳 Manual Docker Steps Required:")
                print("-----------------------------")
                print("Please run the following commands in your command prompt to build and push the Docker image:")
                print(f"cd deployment_package")
                print(f"aws ecr get-login-password --region {args.aws_region} | docker login --username AWS --password-stdin {args.aws_account_id}.dkr.ecr.{args.aws_region}.amazonaws.com")
                print(f"docker build -t {args.aws_account_id}.dkr.ecr.{args.aws_region}.amazonaws.com/{args.repository_name}:latest .")
                print(f"docker push {args.aws_account_id}.dkr.ecr.{args.aws_region}.amazonaws.com/{args.repository_name}:latest")
                print("\nAfter completing these steps, run the following command to continue with deployment:")
                print(f"python aws_deployment_finish.py --aws_region {args.aws_region} --aws_account_id {args.aws_account_id} --repository_name {args.repository_name} --model_name {args.model_name} --role_arn {args.aws_role_arn}")
            else:
                # Run full deployment
                deploy_cmd = (f"python aws_deployment_pytorch.py "
                            f"--model_path ./saved_models/bert_mixed_precision "
                            f"--model_type pytorch "
                            f"--aws_region {args.aws_region} "
                            f"--aws_account_id {args.aws_account_id} "
                            f"--repository_name {args.repository_name} "
                            f"--model_name {args.model_name} "
                            f"--role_arn {args.aws_role_arn}")
                
                success = run_command(deploy_cmd, "Deploying model to AWS")
                if not success:
                    print("AWS deployment failed.")
        else:
            print("Skipping AWS deployment: missing required AWS parameters (aws_account_id and aws_role_arn).")
    
    # Calculate total runtime
    total_time = time.time() - start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    print(f"\n{'='*80}")
    print(f"✅ Project execution completed!")
    print(f"Total runtime: {int(hours):02d}:{int(minutes):02d}:{seconds:.2f}")
    print(f"{'='*80}")
    
    # Prompt to view results
    print("\n📊 Please check the following folders and files for results:")
    print("- reports/performance_report.html: Detailed performance analysis report")
    print("- performance_plots/: Contains performance comparison charts")
    print("- saved_models/: Contains optimized models")
    
    # Deployment instructions
    if not args.skip_aws_deployment and args.aws_account_id and args.aws_role_arn:
        print("\n🌐 Model has been deployed (or prepared for deployment) to AWS SageMaker:")
        print("- You can view and manage the deployed endpoint through the SageMaker console")
        print("- Or access the model programmatically using the AWS SDK")
        print(f"\n⚠️  Remember to clean up AWS resources when done to avoid charges:")
        print(f"python main_pytorch.py --cleanup_aws --aws_region {args.aws_region} --model_name {args.model_name} --repository_name {args.repository_name}")

if __name__ == "__main__":
    main()