import json
import pandas as pd
from pathlib import Path

def load_evaluation_data(file_path):
    """Load evaluation data from JSON file"""
    with open(file_path, 'r') as f:
        return json.load(f)

def create_results_table(data):
    """Create a simple table of all evaluation results"""
    
    presence_data = data['presence_per_label']
    stance_data = data['stance_per_label']
    macro_avg = data['macro_averages']
    
    # Create table for presence detection
    presence_rows = []
    for cat in sorted(presence_data.keys()):
        p_data = presence_data[cat]
        presence_rows.append({
            'Category': cat,
            'Precision': f"{p_data['precision']:.3f}",
            'Recall': f"{p_data['recall']:.3f}",
            'F1-Score': f"{p_data['f1']:.3f}",
            'True Positives': p_data['tp'],
            'False Positives': p_data['fp'],
            'False Negatives': p_data['fn']
        })
    
    # Add macro average row
    presence_rows.append({
        'Category': 'MACRO AVG',
        'Precision': f"{macro_avg['presence']['precision']:.3f}",
        'Recall': f"{macro_avg['presence']['recall']:.3f}",
        'F1-Score': f"{macro_avg['presence']['f1']:.3f}",
        'True Positives': '-',
        'False Positives': '-',
        'False Negatives': '-'
    })
    
    presence_df = pd.DataFrame(presence_rows)
    
    # Create table for stance classification
    stance_rows = []
    for cat in sorted(stance_data.keys()):
        s_data = stance_data[cat]
        stance_rows.append({
            'Category': cat,
            'Class 1 Precision': f"{s_data['class_1']['precision']:.3f}",
            'Class 1 Recall': f"{s_data['class_1']['recall']:.3f}",
            'Class 1 F1': f"{s_data['class_1']['f1']:.3f}",
            'Class -1 Precision': f"{s_data['class_-1']['precision']:.3f}",
            'Class -1 Recall': f"{s_data['class_-1']['recall']:.3f}",
            'Class -1 F1': f"{s_data['class_-1']['f1']:.3f}",
            'Macro Precision': f"{s_data['macro']['precision']:.3f}",
            'Macro Recall': f"{s_data['macro']['recall']:.3f}",
            'Macro F1': f"{s_data['macro']['f1']:.3f}"
        })
    
    # Add macro average row
    stance_rows.append({
        'Category': 'MACRO AVG',
        'Class 1 Precision': '-',
        'Class 1 Recall': '-',
        'Class 1 F1': '-',
        'Class -1 Precision': '-',
        'Class -1 Recall': '-',
        'Class -1 F1': '-',
        'Macro Precision': f"{macro_avg['stance']['precision']:.3f}",
        'Macro Recall': f"{macro_avg['stance']['recall']:.3f}",
        'Macro F1': f"{macro_avg['stance']['f1']:.3f}"
    })
    
    stance_df = pd.DataFrame(stance_rows)
    
    return presence_df, stance_df

def find_all_json_files(results_dir="results"):
    """Recursively find all JSON files in the results directory"""
    results_path = Path(results_dir)
    json_files = list(results_path.rglob("*.json"))
    return json_files

def get_file_info(file_path):
    """Extract model name and configuration from file path"""
    # Remove .json extension and split path
    parts = file_path.stem.split('_')
    
    # Extract model name from path
    model_name = file_path.parent.name
    
    # Extract configuration info from filename
    if len(parts) >= 3:
        task_type = parts[0]  # broad or granular
        shot_type = parts[1]  # 0_shot or 1_shot
        run_number = parts[2]  # run number
        config = f"{task_type}_{shot_type}_run{run_number}"
    else:
        config = file_path.stem
    
    return model_name, config

def main():
    """Main function to generate and display tables for all JSON files"""
    
    # Find all JSON files recursively
    json_files = find_all_json_files()
    
    if not json_files:
        print("No JSON files found in the results directory.")
        return
    
    print(f"Found {len(json_files)} JSON files to process:")
    for file in json_files:
        print(f"  - {file}")
    
    # Create output directory
    output_dir = Path("results_tables")
    output_dir.mkdir(exist_ok=True)
    
    # Process each JSON file
    for json_file in json_files:
        print(f"\n{'='*80}")
        print(f"Processing: {json_file}")
        print(f"{'='*80}")
        
        try:
            # Load the data
            data = load_evaluation_data(json_file)
            print(f"Successfully loaded data from {json_file}")
            
            # Get model and configuration info
            model_name, config = get_file_info(json_file)
            print(f"Model: {model_name}, Config: {config}")
            
            # Generate tables
            presence_df, stance_df = create_results_table(data)
            
            # Display tables
            print("\n" + "="*60)
            print("PRESENCE DETECTION RESULTS")
            print("="*60)
            print(presence_df.to_string(index=False))
            
            print("\n" + "="*80)
            print("STANCE CLASSIFICATION RESULTS")
            print("="*80)
            print(stance_df.to_string(index=False))
            
            # Save tables to CSV files with descriptive names
            safe_model_name = model_name.replace('/', '_').replace('-', '_')
            safe_config = config.replace('/', '_').replace('-', '_')
            
            presence_csv = output_dir / f"{safe_model_name}_{safe_config}_presence.csv"
            stance_csv = output_dir / f"{safe_model_name}_{safe_config}_stance.csv"
            
            presence_df.to_csv(presence_csv, index=False)
            stance_df.to_csv(stance_csv, index=False)
            
            print(f"\nTables saved:")
            print(f"  - {presence_csv}")
            print(f"  - {stance_csv}")
            
        except FileNotFoundError:
            print(f"Error: Could not find {json_file}")
            continue
        except json.JSONDecodeError:
            print(f"Error: Invalid JSON format in {json_file}")
            continue
        except Exception as e:
            print(f"Error processing {json_file}: {e}")
            continue
    
    print(f"\n{'='*80}")
    print("PROCESSING COMPLETE")
    print(f"{'='*80}")
    print(f"All tables saved to '{output_dir}' directory:")
    for csv_file in output_dir.glob("*.csv"):
        print(f"  - {csv_file.name}")

if __name__ == "__main__":
    main() 