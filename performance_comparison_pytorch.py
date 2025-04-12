import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import re

def load_performance_data():
    """Load performance data from report files"""
    print("Loading performance metrics from optimization methods...")
    
    performance_data = {}
    reports_dir = Path("reports")
    
    # Check if directory exists
    if not reports_dir.exists():
        print(f"Directory {reports_dir} does not exist, please run optimization methods first")
        return {}
    
    performance_files = list(reports_dir.glob("*performance*.txt"))
    if not performance_files:
        print(f"No performance report files found in {reports_dir}")
        return {}
    
    print(f"Found {len(performance_files)} performance report files")
    
    # Extract data from each file
    for file_path in performance_files:
        method_name = get_method_name(file_path.name)
        print(f"Processing: {method_name} ({file_path.name})")
        
        try:
            performance_data[method_name] = {}
            with open(file_path, "r") as f:
                for line in f:
                    if ":" not in line:
                        continue
                    
                    parts = line.strip().split(":", 1)
                    if len(parts) != 2:
                        continue
                    
                    key, value = parts[0].strip(), parts[1].strip()
                    
                    # Try to convert to numeric
                    if value.lower() == "none":
                        performance_data[method_name][key] = None
                    else:
                        try:
                            # Try to convert to float
                            performance_data[method_name][key] = float(value)
                        except ValueError:
                            # If not a number, keep as string
                            performance_data[method_name][key] = value
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
    
    return performance_data

def get_method_name(filename):
    """Get optimization method name from filename"""
    if "baseline" in filename.lower():
        return "Baseline"
    elif "mixed_precision" in filename.lower():
        return "Mixed Precision"
    elif "pruning" in filename.lower() or "quantization" in filename.lower():
        return "Pruning & Quantization"
    elif "onnx" in filename.lower() or "tensorrt" in filename.lower():
        return "ONNX/TensorRT"
    else:
        return filename.split("_")[0]

def normalize_data(performance_data):
    """Normalize keys from different reports to standard names"""
    normalized_data = {}
    
    for method, metrics in performance_data.items():
        normalized_data[method] = {}
        
        # Define key mappings of possible names to standard names
        key_mappings = {
            "training_time": ["training_time_seconds", "training_time", "pruning_time_seconds"],
            "inference_time": ["inference_time_seconds", "onnx_inference_time_seconds", "tensorrt_inference_time_seconds"],
            "accuracy": ["validation_accuracy", "accuracy", "onnx_accuracy", "tensorrt_accuracy"],
            "model_size_reduction": ["size_reduction_percent", "size_reduction_percent_total"],
            "model": ["model"]
        }
        
        # Map each metric to standard name
        for standard_key, possible_keys in key_mappings.items():
            for possible_key in possible_keys:
                if possible_key in metrics:
                    normalized_data[method][standard_key] = metrics[possible_key]
                    break
            
            # If no matching key found, set to None
            if standard_key not in normalized_data[method]:
                normalized_data[method][standard_key] = None
    
    return normalized_data

def create_comparison_dataframe(normalized_data):
    """Create a DataFrame for comparison"""
    print("Creating performance comparison table...")
    
    # Define columns
    columns = {
        "Optimization Method": [],
        "Training Time (sec)": [],
        "Inference Time (sec)": [],
        "Accuracy": [],
        "Model Size Reduction (%)": []
    }
    
    # Get baseline inference time (if exists)
    baseline_inference_time = None
    if "Baseline" in normalized_data:
        baseline_inference_time = normalized_data["Baseline"].get("inference_time")
    
    # Fill data
    for method, metrics in normalized_data.items():
        columns["Optimization Method"].append(method)
        columns["Training Time (sec)"].append(metrics.get("training_time"))
        columns["Inference Time (sec)"].append(metrics.get("inference_time"))
        columns["Accuracy"].append(metrics.get("accuracy"))
        columns["Model Size Reduction (%)"].append(metrics.get("model_size_reduction"))
        
    # Create DataFrame
    df = pd.DataFrame(columns)
    
    # Calculate inference speedup (if baseline exists)
    if baseline_inference_time is not None and baseline_inference_time > 0:
        df["Inference Speedup"] = df["Inference Time (sec)"].apply(
            lambda x: baseline_inference_time / x if x and x > 0 else None
        )
    
    return df

def plot_performance_metrics(df):
    """Plot performance metrics comparison charts"""
    print("Generating performance comparison charts...")
    
    # Create output directory
    output_dir = Path("performance_plots")
    output_dir.mkdir(exist_ok=True)
    
    # Set chart style
    plt.style.use('ggplot')
    
    # 1. Training time comparison (if data exists)
    if not df["Training Time (sec)"].isna().all():
        plot_bar_chart(
            df, 
            "Training Time (sec)", 
            "Training Time Comparison Across Optimization Methods",
            "Training Time (seconds)",
            output_dir / "training_time_comparison.png",
            value_format="{:.2f}s"
        )
    
    # 2. Inference time comparison
    if not df["Inference Time (sec)"].isna().all():
        plot_bar_chart(
            df, 
            "Inference Time (sec)", 
            "Inference Time Comparison Across Optimization Methods",
            "Inference Time (seconds)",
            output_dir / "inference_time_comparison.png",
            value_format="{:.2f}s"
        )
    
    # 3. Accuracy comparison
    if not df["Accuracy"].isna().all():
        plot_bar_chart(
            df, 
            "Accuracy", 
            "Accuracy Comparison Across Optimization Methods",
            "Accuracy",
            output_dir / "accuracy_comparison.png",
            value_format="{:.4f}",
            y_min=max(0.8, df["Accuracy"].min() * 0.99) if not df["Accuracy"].isna().all() else 0
        )
    
    # 4. Inference speedup comparison (if exists)
    if "Inference Speedup" in df.columns and not df["Inference Speedup"].isna().all():
        # Filter out baseline (usually speedup = 1.0)
        speedup_df = df[df["Optimization Method"] != "Baseline"].copy()
        if not speedup_df.empty:
            plot_bar_chart(
                speedup_df, 
                "Inference Speedup", 
                "Inference Speedup Comparison",
                "Speedup (relative to baseline)",
                output_dir / "inference_speedup_comparison.png",
                value_format="{:.2f}x"
            )
    
    # 5. Model size reduction comparison (if data exists)
    non_null_size = df["Model Size Reduction (%)"].dropna()
    if not non_null_size.empty and (non_null_size > 0).any():
        size_df = df[df["Model Size Reduction (%)"].notna() & (df["Model Size Reduction (%)"] > 0)].copy()
        if not size_df.empty:
            plot_bar_chart(
                size_df, 
                "Model Size Reduction (%)", 
                "Model Size Reduction Comparison",
                "Size Reduction (%)",
                output_dir / "model_size_reduction.png",
                value_format="{:.1f}%"
            )
    
    print(f"Charts saved to {output_dir} directory")

def plot_bar_chart(df, metric, title, ylabel, output_path, value_format="{:.2f}", y_min=None):
    """Plot bar chart for specific metric"""
    # Filter out NaN values
    plot_df = df[df[metric].notna()].copy()
    if plot_df.empty:
        print(f"No valid data for {metric}, skipping chart")
        return
    
    plt.figure(figsize=(10, 6))
    
    # Sort data for better comparison (for most metrics, smaller is better, for accuracy and speedup, larger is better)
    if metric in ["Accuracy", "Inference Speedup", "Model Size Reduction (%)"]:
        plot_df = plot_df.sort_values(by=metric)
    else:
        plot_df = plot_df.sort_values(by=metric, ascending=False)
    
    # Create bar chart
    bars = plt.bar(plot_df["Optimization Method"], plot_df[metric], color='skyblue')
    
    # Set title and labels
    plt.title(title, fontsize=14)
    plt.xlabel('Optimization Method', fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.xticks(rotation=45, ha='right')
    
    # Set y-axis minimum value (if provided)
    if y_min is not None:
        plt.ylim(bottom=y_min)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.annotate(
            value_format.format(height),
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha='center', va='bottom'
        )
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def generate_performance_report(df):
    """Generate performance comparison report"""
    print("Generating performance report...")
    
    # Create report directory
    report_dir = Path("reports")
    report_dir.mkdir(exist_ok=True)
    
    # Save CSV format data
    df.to_csv(report_dir / "performance_comparison.csv", index=False)
    
    # Prepare HTML report
    html_report = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>BERT Model Optimization Performance Report</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 20px;
                line-height: 1.6;
            }}
            h1, h2, h3 {{
                color: #333;
            }}
            table {{
                border-collapse: collapse;
                width: 100%;
                margin-bottom: 20px;
            }}
            th, td {{
                border: 1px solid #ddd;
                padding: 8px;
                text-align: left;
            }}
            th {{
                background-color: #f2f2f2;
            }}
            tr:nth-child(even) {{
                background-color: #f9f9f9;
            }}
            .image-container {{
                display: flex;
                flex-wrap: wrap;
                justify-content: space-between;
                margin-top: 20px;
            }}
            .image-item {{
                width: 48%;
                margin-bottom: 20px;
            }}
            img {{
                max-width: 100%;
                height: auto;
            }}
            .highlight {{
                font-weight: bold;
                color: #2c6fad;
            }}
            .warning {{
                color: #e74c3c;
            }}
        </style>
    </head>
    <body>
        <h1>BERT Model Optimization Performance Report</h1>
        
        <h2>Performance Comparison Table</h2>
        {df.to_html(index=False)}
        
        <h2>Performance Analysis</h2>
    """
    
    # Analyze training time
    if not df["Training Time (sec)"].isna().all():
        try:
            fastest_training = df.loc[df["Training Time (sec)"].idxmin()]
            baseline_training = df[df["Optimization Method"] == "Baseline"]["Training Time (sec)"].values[0] if "Baseline" in df["Optimization Method"].values else None
            
            html_report += "<h3>Training Performance Analysis</h3><ul>"
            
            html_report += f"<li>Fastest training method is <span class='highlight'>{fastest_training['Optimization Method']}</span>, with training time of {fastest_training['Training Time (sec)']:.2f} seconds.</li>"
            
            if baseline_training and baseline_training > 0:
                speedup = baseline_training / fastest_training["Training Time (sec)"]
                html_report += f"<li>Compared to baseline, the fastest method provides <span class='highlight'>{speedup:.2f}x</span> training speedup.</li>"
            
            html_report += "</ul>"
        except Exception as e:
            print(f"Error analyzing training time: {e}")
    
    # Analyze inference time
    if not df["Inference Time (sec)"].isna().all():
        try:
            fastest_inference = df.loc[df["Inference Time (sec)"].idxmin()]
            html_report += "<h3>Inference Performance Analysis</h3><ul>"
            
            html_report += f"<li>Fastest inference method is <span class='highlight'>{fastest_inference['Optimization Method']}</span>, with inference time of {fastest_inference['Inference Time (sec)']:.2f} seconds.</li>"
            
            if "Inference Speedup" in df.columns and not fastest_inference["Inference Speedup"] is None:
                html_report += f"<li>Compared to baseline, it provides <span class='highlight'>{fastest_inference['Inference Speedup']:.2f}x</span> inference speedup.</li>"
            
            html_report += "</ul>"
        except Exception as e:
            print(f"Error analyzing inference time: {e}")
    
    # Analyze accuracy
    if not df["Accuracy"].isna().all():
        try:
            most_accurate = df.loc[df["Accuracy"].idxmax()]
            baseline_accuracy = df[df["Optimization Method"] == "Baseline"]["Accuracy"].values[0] if "Baseline" in df["Optimization Method"].values else None
            
            html_report += "<h3>Model Accuracy Analysis</h3><ul>"
            
            html_report += f"<li>Most accurate method is <span class='highlight'>{most_accurate['Optimization Method']}</span>, with accuracy of {most_accurate['Accuracy']:.4f}.</li>"
            
            if baseline_accuracy:
                for _, row in df.iterrows():
                    if row["Optimization Method"] != "Baseline" and not pd.isna(row["Accuracy"]):
                        acc_diff = row["Accuracy"] - baseline_accuracy
                        if abs(acc_diff) > 0.01:  # Changes over 1% are worth noting
                            direction = "increased" if acc_diff > 0 else "decreased"
                            html_report += f"<li><span class='highlight'>{row['Optimization Method']}</span> {direction} accuracy by {abs(acc_diff):.4f} ({abs(acc_diff) * 100:.2f}%) compared to baseline.</li>"
            
            html_report += "</ul>"
        except Exception as e:
            print(f"Error analyzing accuracy: {e}")
    
    # Analyze model size
    if not df["Model Size Reduction (%)"].isna().all() and (df["Model Size Reduction (%)"] > 0).any():
        try:
            best_reduction = df.loc[df["Model Size Reduction (%)"].idxmax()]
            
            html_report += "<h3>Model Size Analysis</h3><ul>"
            
            html_report += f"<li>Largest model size reduction comes from <span class='highlight'>{best_reduction['Optimization Method']}</span>, reducing size by {best_reduction['Model Size Reduction (%)']:.2f}%.</li>"
            
            html_report += "</ul>"
        except Exception as e:
            print(f"Error analyzing model size: {e}")
    
    # Comprehensive analysis and recommendations
    html_report += """
        <h2>Comprehensive Analysis and Recommendations</h2>
        <p>
            Based on the analysis above, we can see that different optimization methods have their own strengths and weaknesses:
        </p>
        <ul>
    """
    
    # Recommendations for different scenarios
    html_report += """
            <li><strong>For training speed priority</strong>: Mixed precision training typically provides the best training acceleration with minimal impact on model accuracy.</li>
            <li><strong>For deployment on resource-constrained devices</strong>: Model pruning and quantization can significantly reduce model size, though may slightly affect accuracy.</li>
            <li><strong>For low-latency inference scenarios</strong>: TensorRT optimization typically provides the best inference performance boost, especially on CUDA-supported devices.</li>
            <li><strong>For cross-platform deployment</strong>: ONNX format provides good cross-platform compatibility while maintaining decent inference performance.</li>
        </ul>
        
        <p>
            The ideal optimization strategy is often a combination of these methods. For example, you might first apply mixed precision training to speed up the training process, then prune and quantize the trained model to reduce its size,
            and finally use TensorRT for inference optimization to get the best end-to-end performance.
        </p>
        
        <h3>Important Considerations</h3>
        <p class="warning">
            Optimization involves trade-offs. The fastest method may not be the most accurate, and the smallest model may not provide the best performance. Choose the optimization method combination that best fits your specific application needs.
        </p>
    """
    
    # Add charts
    html_report += """
        <h2>Performance Visualization</h2>
        <div class="image-container">
    """
    
    # Add all possible charts (if they exist)
    charts = [
        ("training_time_comparison.png", "Training Time Comparison"),
        ("inference_time_comparison.png", "Inference Time Comparison"),
        ("accuracy_comparison.png", "Accuracy Comparison"),
        ("inference_speedup_comparison.png", "Inference Speedup Comparison"),
        ("model_size_reduction.png", "Model Size Reduction Comparison")
    ]
    
    for chart_file, chart_desc in charts:
        if (Path("performance_plots") / chart_file).exists():
            html_report += f"""
            <div class="image-item">
                <img src="../performance_plots/{chart_file}" alt="{chart_desc}">
                <p>{chart_desc}</p>
            </div>
            """
    
    html_report += """
        </div>
        
        <h2>Conclusion</h2>
        <p>
            Deep learning model optimization is a crucial step for practical applications. This project has demonstrated that by combining multiple optimization techniques,
            we can significantly improve BERT model training efficiency and inference performance without substantially sacrificing accuracy.
        </p>
        <p>
            Each optimization technique has its applicable scenarios and potential limitations. Choosing the right optimization strategy requires consideration of specific application requirements and hardware environment.
            Continuous monitoring and analysis of optimization effects are key steps to ensure optimal model performance.
        </p>
    </body>
    </html>
    """
    
    # Save HTML report (with UTF-8 encoding)
    with open(report_dir / "performance_report.html", "w", encoding="utf-8") as f:
        f.write(html_report)
    
    print(f"Performance report generated and saved to {report_dir / 'performance_report.html'}")

def main():
    """Main function"""
    print("Starting performance comparison analysis...")
    
    # Load performance data
    performance_data = load_performance_data()
    
    if not performance_data:
        print("No performance data found. Please run optimization models and generate performance reports first.")
        return
    
    # Normalize data
    normalized_data = normalize_data(performance_data)
    
    # Create comparison table
    df = create_comparison_dataframe(normalized_data)
    
    # Print comparison table
    print("\nPerformance comparison table:")
    print(df)
    
    # Plot performance metrics
    plot_performance_metrics(df)
    
    # Generate performance report
    generate_performance_report(df)
    
    print("Performance comparison analysis complete!")

if __name__ == "__main__":
    main()