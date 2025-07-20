import os
import pandas as pd

def parse_generated_notes(path):
    notes = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            cat_str, note = line.split('\t', 1)
            categories = [int(idx) for idx in cat_str.split(',') if idx]
            notes.append({'labels': categories, 'note': note})
    return notes

def add_text_column(data, notes_dir="combined/notes"):
    """
    Add a 'text' column to the DataFrame containing the contents of each file.
    
    Args:
        data: pandas DataFrame with 'file' column containing filenames
        notes_dir: directory containing the text files
    
    Returns:
        DataFrame with new 'text' column
    """
    def read_file_content(filename):
        """Read the content of a file from the notes directory"""
        file_path = os.path.join(notes_dir, filename)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read().strip()
        except FileNotFoundError:
            print(f"Warning: File {filename} not found in {notes_dir}")
            return ""
        except Exception as e:
            print(f"Error reading {filename}: {e}")
            return ""
    
    # Add the text column by reading each file
    data['text'] = data['file'].apply(read_file_content)
    
    return data

if __name__ == "__main__":
    # Example usage
    DATA_DIR = os.path.join(os.path.dirname(__file__), 'data', 'chop')
    
    # Read the labels CSV
    data = pd.read_csv(os.path.join(DATA_DIR, 'labels_cleaned.csv')).fillna('')
    
    # Add text column
    data_with_text = add_text_column(data)
    
    print(f"DataFrame shape: {data_with_text.shape}")
    print(f"Columns: {data_with_text.columns.tolist()}")
    print("\nFirst few rows:")
    print(data_with_text[['file', 'cats', 'text']].head())
    
    # Check for empty text files
    empty_files = data_with_text[data_with_text['text'] == '']
    print(f"\nFiles with empty text: {len(empty_files)}")
    if len(empty_files) > 0:
        print("Empty files:", empty_files['file'].tolist()[:5])
