#!/usr/bin/env python3
"""
Master Script for Running All MPS-RetNet Ablation Studies
Runs all ablation experiments and generates a summary report
"""

import os
import sys
import subprocess
import time
from datetime import datetime
import pandas as pd

# Configuration
RESULTS_DIR = "ablation_results"
SCRIPTS = [
    ("ablation_components.py", "Component Ablation Study"),
    ("ablation_hyperparameters.py", "Hyperparameter Sensitivity Analysis"),
    ("ablation_backbones.py", "Backbone Architecture Comparison"),
    ("ablation_label_efficiency.py", "Label Efficiency Analysis")
]

def print_header(text):
    """Print formatted header"""
    print("\n" + "="*70)
    print(text.center(70))
    print("="*70 + "\n")

def run_script(script_name, description):
    """Run a single ablation script"""
    print_header(f"Starting: {description}")
    print(f"Script: {script_name}")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    start_time = time.time()
    
    try:
        # Run the script
        result = subprocess.run(
            [sys.executable, script_name],
            capture_output=True,
            text=True
        )
        
        # Print output
        if result.stdout:
            print("\n--- Script Output ---")
            print(result.stdout)
        
        # Check for errors
        if result.returncode != 0:
            print("\n--- Error Output ---")
            print(result.stderr)
            return False, time.time() - start_time
        
        elapsed_time = time.time() - start_time
        print(f"\nCompleted in {elapsed_time/60:.1f} minutes")
        return True, elapsed_time
        
    except Exception as e:
        print(f"\n*** ERROR: {e}")
        return False, time.time() - start_time

def generate_summary_report():
    """Generate a comprehensive summary report from all results"""
    print_header("Generating Summary Report")
    
    summary = {
        'Study': [],
        'Results File': [],
        'Status': []
    }
    
    result_files = [
        ('Component Ablation', 'table_ablation_components.csv'),
        ('Hyperparameter Sensitivity', 'table_hyperparameter_sensitivity.csv'),
        ('Backbone Comparison', 'table_backbone_comparison.csv'),
        ('Label Efficiency', 'table_label_efficiency.csv')
    ]
    
    print("\nResults Summary:")
    print("-" * 70)
    
    for study_name, filename in result_files:
        filepath = os.path.join(RESULTS_DIR, filename)
        
        if os.path.exists(filepath):
            try:
                df = pd.read_csv(filepath)
                print(f"\n{study_name}:")
                print(f"  File: {filename}")
                print(f"  Rows: {len(df)}")
                print(f"  Preview:")
                print(df.head(3).to_string(index=False))
                
                summary['Study'].append(study_name)
                summary['Results File'].append(filename)
                summary['Status'].append('✓ Complete')
            except Exception as e:
                print(f"\n{study_name}:")
                print(f"  File: {filename}")
                print(f"  ✗ Error reading file: {e}")
                
                summary['Study'].append(study_name)
                summary['Results File'].append(filename)
                summary['Status'].append('✗ Error')
        else:
            print(f"\n{study_name}:")
            print(f"  File: {filename}")
            print(f"  ✗ File not found")
            
            summary['Study'].append(study_name)
            summary['Results File'].append(filename)
            summary['Status'].append('✗ Not found')
    
    # Save summary
    summary_df = pd.DataFrame(summary)
    summary_path = os.path.join(RESULTS_DIR, 'SUMMARY_REPORT.csv')
    summary_df.to_csv(summary_path, index=False)
    
    print(f"\n\nSummary saved to: {summary_path}")
    print("\n" + summary_df.to_string(index=False))

def create_latex_tables():
    """Generate LaTeX table code for easy paper integration"""
    print_header("Generating LaTeX Tables")
    
    latex_output = []
    latex_output.append("% LaTeX Tables for MPS-RetNet Ablation Studies")
    latex_output.append("% Generated: " + datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    latex_output.append("")
    
    result_files = {
        'Component Ablation': 'table_ablation_components.csv',
        'Hyperparameter Sensitivity': 'table_hyperparameter_sensitivity.csv',
        'Backbone Comparison': 'table_backbone_comparison.csv',
        'Label Efficiency': 'table_label_efficiency.csv'
    }
    
    for study_name, filename in result_files.items():
        filepath = os.path.join(RESULTS_DIR, filename)
        
        if os.path.exists(filepath):
            try:
                df = pd.read_csv(filepath)
                
                latex_output.append(f"\n% {study_name}")
                latex_output.append("\\begin{table}[htbp]")
                latex_output.append("\\centering")
                latex_output.append(f"\\caption{{{study_name} Results}}")
                latex_output.append(f"\\label{{tab:{filename.replace('.csv', '').replace('_', '')}}}")
                
                # Generate column specification
                num_cols = len(df.columns)
                col_spec = 'l' + 'c' * (num_cols - 1)
                latex_output.append(f"\\begin{{tabular}}{{{col_spec}}}")
                latex_output.append("\\toprule")
                
                # Header
                header = " & ".join(df.columns) + " \\\\"
                latex_output.append(header)
                latex_output.append("\\midrule")
                
                # Data rows
                for _, row in df.iterrows():
                    row_str = " & ".join(str(val) for val in row.values) + " \\\\"
                    latex_output.append(row_str)
                
                latex_output.append("\\bottomrule")
                latex_output.append("\\end{tabular}")
                latex_output.append("\\end{table}")
                latex_output.append("")
                
            except Exception as e:
                latex_output.append(f"% Error processing {filename}: {e}")
    
    # Save LaTeX tables
    latex_path = os.path.join(RESULTS_DIR, 'LATEX_TABLES.tex')
    with open(latex_path, 'w') as f:
        f.write('\n'.join(latex_output))
    
    print(f"LaTeX tables saved to: {latex_path}")

def main():
    """Main execution function"""
    print_header("MPS-RetNet Ablation Studies - Master Script")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Create results directory
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print(f"Results directory: {RESULTS_DIR}")
    
    # Check if scripts exist
    missing_scripts = []
    for script_name, _ in SCRIPTS:
        if not os.path.exists(script_name):
            missing_scripts.append(script_name)
    
    if missing_scripts:
        print("\n*** WARNING: The following scripts are missing:")
        for script in missing_scripts:
            print(f"  - {script}")
        print("\nPlease ensure all scripts are in the current directory.")
        response = input("\nContinue anyway? (y/n): ")
        if response.lower() != 'y':
            return
    
    # Run each script
    results = []
    total_start_time = time.time()
    
    for script_name, description in SCRIPTS:
        if script_name in missing_scripts:
            print(f"\nSkipping {script_name} (file not found)")
            results.append((description, False, 0))
            continue
        
        success, elapsed_time = run_script(script_name, description)
        results.append((description, success, elapsed_time))
        
        if not success:
            print(f"\n*** {description} FAILED ***")
            response = input("\nContinue with remaining studies? (y/n): ")
            if response.lower() != 'y':
                break
    
    total_elapsed_time = time.time() - total_start_time
    
    # Print summary
    print_header("Execution Summary")
    print(f"Total time: {total_elapsed_time/3600:.2f} hours ({total_elapsed_time/60:.1f} minutes)")
    print("\nIndividual Study Results:")
    print("-" * 70)
    
    for description, success, elapsed_time in results:
        status = "✓ SUCCESS" if success else "✗ FAILED"
        time_str = f"{elapsed_time/60:.1f} min" if elapsed_time > 0 else "N/A"
        print(f"{description:.<45} {status:.<10} {time_str:.>10}")
    
    # Generate reports
    if any(success for _, success, _ in results):
        generate_summary_report()
        create_latex_tables()
    
    print_header("ALL STUDIES COMPLETE")
    print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nResults are available in: {RESULTS_DIR}/")
    print("\nGenerated files:")
    print("  - table_ablation_components.csv")
    print("  - table_hyperparameter_sensitivity.csv")
    print("  - table_backbone_comparison.csv")
    print("  - table_label_efficiency.csv")
    print("  - SUMMARY_REPORT.csv")
    print("  - LATEX_TABLES.tex")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n*** Execution interrupted by user ***")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n*** Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)