import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from config import CAT_TO_LABELS 
import os

# Set up plotting style
plt.style.use('default')
sns.set_palette("husl")

def get_category_labels():
    """Get human-readable category labels"""
    # Create mapping from index to category name
    index_to_category = {i: cat for i, cat in enumerate(CAT_TO_LABELS.keys())}
    
    # Create nice display names
    display_names = {
        "PatientCaregiver_Employment": "Employment",
        "HousingInstability": "Housing Instability", 
        "FoodInsecurity": "Food Insecurity",
        "FinancialStrain": "Financial Strain",
        "Transportation": "Transportation",
        "Childcare": "Childcare",
        "Permanency": "Permanency",
        "SubstanceAbuse": "Substance Abuse",
        "Safety": "Safety"
    }
    
    return index_to_category, display_names

def create_individual_label_tables(data):
    """Create individual tables for each label"""
    presence_data = data['presence_per_label']
    stance_data = data['stance_per_label']
    
    index_to_category, display_names = get_category_labels()
    
    label_tables = {}
    for cat_idx in sorted(presence_data.keys()):
        cat_idx = int(cat_idx)
        category_name = index_to_category[cat_idx]
        print(category_name)

        display_name = display_names[category_name]
        
        p_data = presence_data[str(cat_idx)]
        s_data = stance_data[str(cat_idx)]
        
        # Create table for this label
        label_table = pd.DataFrame([
            {
                'Metric': 'Presence Detection',
                'Precision': f"{p_data['precision']:.3f}",
                'Recall': f"{p_data['recall']:.3f}",
                'F1-Score': f"{p_data['f1']:.3f}",
                'Accuracy': f"{p_data['accuracy']:.3f}"
            },
            {
                'Metric': 'Stance Classification',
                'Precision': f"{s_data['precision']:.3f}",
                'Recall': f"{s_data['recall']:.3f}",
                'F1-Score': f"{s_data['f1']:.3f}",
                'Accuracy': f"{s_data['accuracy']:.3f}"
            }
        ])
        
        label_tables[display_name] = label_table
    
    return label_tables

def create_macro_presence_table(data):
    """Create macro presence detection results table"""
    macro_avg = data['macro_averages']['presence']
    
    macro_presence_df = pd.DataFrame([
        {
            'Precision': f"{macro_avg['precision']:.3f}",
            'Recall': f"{macro_avg['recall']:.3f}",
            'F1-Score': f"{macro_avg['f1']:.3f}",
            'Accuracy': f"{macro_avg['accuracy']:.3f}"
        }
    ])
    
    return macro_presence_df

def create_macro_stance_table(data):
    """Create macro stance classification results table"""
    macro_avg = data['macro_averages']['stance']
    
    macro_stance_df = pd.DataFrame([
        {
            'Precision': f"{macro_avg['precision']:.3f}",
            'Recall': f"{macro_avg['recall']:.3f}",
            'F1-Score': f"{macro_avg['f1']:.3f}",
            'Accuracy': f"{macro_avg['accuracy']:.3f}"
        }
    ])
    
    return macro_stance_df

def get_tables(json_file):
    """Visualize results from a single JSON file
    
    Args:
        json_file_path (str): Path to the JSON file containing evaluation results
        output_dir (str, optional): Directory to save output files. Defaults to "results_tables"
    
    Returns:
        dict: Dictionary containing all generated tables and metadata
        
    Example:
        results = visualize_single_json("results/Llama-3.1-8B-Instruct/broad_0_shot.txt_0.json")
        print(f"Model: {results['model_name']}")
        print(f"Config: {results['task_type']}, {results['shots']} shot")
        for label_name, table in results['label_tables'].items():
            print(f"{label_name}: {table}")
    """
    output_dir = Path("tables")
    output_dir.mkdir(exist_ok=True)
    
    try:
        with open(json_file, "r") as f:
            data = json.load(f)
        print(f"Successfully loaded data from {json_file}")
        
        # Get model and configuration info
        model_name, task_type, shots = get_file_info(json_file)
        print(f"Model: {model_name}, {task_type}, {shots} shot")
        
        # Generate individual label tables
        label_tables = create_individual_label_tables(data)
        
        # Generate macro tables
        macro_presence_df = create_macro_presence_table(data)
        macro_stance_df = create_macro_stance_table(data)
        
        print(f"\n{'='*100}")
        print(f"RESULTS FOR {model_name.upper()} - {task_type}, {shots} shot")
        print(f"{'='*100}")

        os.makedirs(output_dir / model_name, exist_ok=True)
        os.makedirs(output_dir / model_name / f"{task_type}_{shots}_shot", exist_ok=True)
        
        for label_name, label_table in label_tables.items():
            print(f"\n" + "="*80)
            print(f"{label_name.upper()}")
            print("="*80)
            print(label_table.to_string(index=False))
            
            # Save individual label table
            label_csv = output_dir / model_name / f"{task_type}_{shots}_shot" / f"{label_name}.csv"
            label_table.to_csv(label_csv, index=False)
            print(f"Saved: {label_csv}")
        
        # Display macro tables
        print(f"\n{'='*100}")
        print("MACRO PRESENCE DETECTION RESULTS")
        print("="*100)
        print(macro_presence_df.to_string(index=False))
        
        print(f"\n{'='*100}")
        print("MACRO STANCE CLASSIFICATION RESULTS")
        print("="*100)
        print(macro_stance_df.to_string(index=False))
        
        # Save macro tables
        macro_presence_csv = output_dir / model_name / f"{task_type}_{shots}_shot" / "macro_presence.csv"
        macro_stance_csv = output_dir / model_name / f"{task_type}_{shots}_shot" / "macro_stance.csv"
        
        macro_presence_df.to_csv(macro_presence_csv, index=False)
        macro_stance_df.to_csv(macro_stance_csv, index=False)
        
        print(f"\nMacro tables saved:")
        print(f"  - {macro_presence_csv}")
        print(f"  - {macro_stance_csv}")
        
        return {
            'label_tables': label_tables,
            'macro_presence': macro_presence_df,
            'macro_stance': macro_stance_df,
            'model_name': model_name
        }
        
    except FileNotFoundError:
        print(f"Error: Could not find {json_file}")
        raise
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in {json_file}")
        raise
    except Exception as e:
        print(f"Error processing {json_file}: {e}")
        raise

def get_file_info(file_path):

    """Extract model name and configuration from file path"""
    # Remove .json extension and split path
    parts = file_path.split('/')
    model_name = parts[-2]
    
    # Extract model name from path
    json_part = parts[-1].split('_')
    task_type = json_part[0]  # broad or granular
    shots = json_part[1]
    
    return model_name, task_type, shots

def main():
    """Main function to demonstrate usage of visualize_single_json"""
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python visualize_results.py <json_file_path>")
        print("Example: python visualize_results.py results/Llama-3.1-8B-Instruct/broad_0_shot.txt_0.json")
        return
    
    json_file_path = sys.argv[1]
    
    get_tables(json_file_path)

if __name__ == "__main__":
    main() 