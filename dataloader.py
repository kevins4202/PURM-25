import os
import random
from torch.utils.data import Dataset, DataLoader

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

class SDOHDataset(Dataset):
    def __init__(self, notes_and_labels):
        self.notes = []
        self.labels = []
        for note_and_labels in notes_and_labels:
            self.notes.append(note_and_labels['note'])
            self.labels.append(note_and_labels['labels'])

    def __len__(self):
        return len(self.notes)
    
    def __getitem__(self, idx):
        return self.notes[idx], self.labels[idx]

DATA_DIR = os.path.join(os.getcwd(), 'data', 'generated')
examples = parse_generated_notes(os.path.join(DATA_DIR, 'generated_notes.txt'))

random.seed(42)

def get_dataloaders(batch_size=16, split=False):
    if split:
        n = len(examples)
        train_end = int(n * 0.8)
        val_end = int(n * 0.9)

        train_examples = examples[:train_end]
        val_examples = examples[train_end:val_end]
        test_examples = examples[val_end:]

        train_dataset = SDOHDataset(train_examples)
        val_dataset = SDOHDataset(val_examples)
        test_dataset = SDOHDataset(test_examples)

        print(f"Train dataset size: {len(train_dataset)}")
        print(f"Validation dataset size: {len(val_dataset)}")
        print(f"Test dataset size: {len(test_dataset)}")

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        return train_dataloader, val_dataloader, test_dataloader
    else:
        dataloader = DataLoader(examples, batch_size=batch_size, shuffle=False)
        
        return dataloader
    
if __name__ == "__main__":
    dataloader = get_dataloaders(batch_size=1, split=False)
    for batch in dataloader:
        print(batch)
        break