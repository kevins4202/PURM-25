import json
import pandas as pd
from pathlib import Path
from config import CAT_TO_LABELS

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

def create_presence_table(data, model_prompt):
    """Create presence detection table with one row per model/prompt"""
    presence_data = data['presence_per_label']
    macro_avg = data['macro_averages']['presence']
    
    # Get category labels
    index_to_category, display_names = get_category_labels()
    
    # Initialize row data
    row_data = {'model_prompt': model_prompt}
    
    # Add individual label metrics (27 columns: 9 labels * 3 metrics)
    for cat_idx in sorted(presence_data.keys()):
        cat_idx = int(cat_idx)
        category_name = index_to_category[cat_idx]
        display_name = display_names[category_name]
        
        p_data = presence_data[str(cat_idx)]
        
        # Add precision, recall, f1 for this label
        row_data[f'{display_name}_precision'] = p_data['precision']
        row_data[f'{display_name}_recall'] = p_data['recall']
        row_data[f'{display_name}_f1'] = p_data['f1']
    
    # Add macro metrics (3 columns)
    row_data['macro_precision'] = macro_avg['precision']
    row_data['macro_recall'] = macro_avg['recall']
    row_data['macro_f1'] = macro_avg['f1']
    
    return pd.DataFrame([row_data])

def create_stance_table(data, model_prompt):
    """Create stance classification table with one row per model/prompt"""
    stance_data = data['stance_per_label']
    macro_avg = data['macro_averages']['stance']
    
    # Get category labels
    index_to_category, display_names = get_category_labels()
    
    # Initialize row data
    row_data = {'model_prompt': model_prompt}
    
    # Add individual label metrics (27 columns: 9 labels * 3 metrics)
    for cat_idx in sorted(stance_data.keys()):
        cat_idx = int(cat_idx)
        category_name = index_to_category[cat_idx]
        display_name = display_names[category_name]
        
        s_data = stance_data[str(cat_idx)]
        
        # Add precision, recall, f1 for this label
        row_data[f'{display_name}_precision'] = s_data['precision']
        row_data[f'{display_name}_recall'] = s_data['recall']
        row_data[f'{display_name}_f1'] = s_data['f1']
    
    # Add macro metrics (3 columns)
    row_data['macro_precision'] = macro_avg['precision']
    row_data['macro_recall'] = macro_avg['recall']
    row_data['macro_f1'] = macro_avg['f1']
    
    return pd.DataFrame([row_data])

def create_social_needs_table(data, model_prompt):
    """Create social needs table with one row per model/prompt"""
    social_needs_data = data['social_needs']
    
    # Initialize row data
    row_data = {'model_prompt': model_prompt}
    
    # Add social needs metrics (3 columns)
    row_data['precision'] = social_needs_data['precision']
    row_data['recall'] = social_needs_data['recall']
    row_data['f1'] = social_needs_data['f1']
    
    return pd.DataFrame([row_data])

def get_tables(json_file):
    """Create three CSV tables from a single JSON file
    
    Args:
        json_file (str): Path to the JSON file containing evaluation results
    
    Returns:
        dict: Dictionary containing the three DataFrames
    """
    output_dir = Path("tables")
    output_dir.mkdir(exist_ok=True)
    
    try:
        with open(json_file, "r") as f:
            data = json.load(f)
        print(f"Successfully loaded data from {json_file}")
        
        # Get model and configuration info
        model_name, task_type, shots = get_file_info(json_file)
        model_prompt = f"{model_name}_{task_type}_{shots}_shot"
        print(f"Model: {model_name}, {task_type}, {shots} shot")
        
        # Create the three tables
        presence_df = create_presence_table(data, model_prompt)
        stance_df = create_stance_table(data, model_prompt)
        social_needs_df = create_social_needs_table(data, model_prompt)
        
        # Save CSV files
        presence_csv = output_dir / "presence.csv"
        stance_csv = output_dir / "stance.csv"
        social_needs_csv = output_dir / "social_needs.csv"
        
        # Append to existing files or create new ones
        presence_df.to_csv(presence_csv, mode='a', header=not presence_csv.exists(), index=False)
        stance_df.to_csv(stance_csv, mode='a', header=not stance_csv.exists(), index=False)
        social_needs_df.to_csv(social_needs_csv, mode='a', header=not social_needs_csv.exists(), index=False)
        
        print(f"Tables saved:")
        print(f"  - {presence_csv}")
        print(f"  - {stance_csv}")
        print(f"  - {social_needs_csv}")
        
        return {
            'presence': presence_df,
            'stance': stance_df,
            'social_needs': social_needs_df
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

def main():
    """Main function to demonstrate usage of get_tables"""
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python get_tables.py <json_file_path>")
        print("Example: python get_tables.py results/Llama-3.1-8B-Instruct/broad_0_shot.json")
        return
    
    json_file_path = sys.argv[1]
    
    presence_df, stance_df, social_needs_df = get_tables(json_file_path)
    print(presence_df)
    print(stance_df)
    print(social_needs_df)

if __name__ == "__main__":
    main() 