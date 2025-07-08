import os

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

if __name__ == "__main__":
    DATA_DIR = os.path.join(os.path.dirname(__file__), 'data', 'generated')
    notes = parse_generated_notes(os.path.join(DATA_DIR, 'generated_notes.txt'))
    print(f"Parsed {len(notes)} notes. Example:")
    print(notes[0] if notes else 'No notes found.')
