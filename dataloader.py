import os
import random
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch

class SDOHDataset(Dataset):
    def __init__(self, notes_and_labels):
        self.notes = []
        self.labels = []
        for note_and_labels in notes_and_labels:
            self.notes.append(note_and_labels[0])
            self.labels.append(note_and_labels[1])

    def __len__(self):
        return len(self.notes)
    
    def __getitem__(self, idx):
        return self.notes[idx], self.labels[idx]

def custom_collate_fn(batch):
    """Custom collate function to prevent automatic padding"""
    notes = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    
    # Convert labels to tensors without padding
    label_tensors = [torch.tensor(label, dtype=torch.long) for label in labels]
    
    return {"note": notes, "labels": label_tensors}

def get_dataloaders(batch_size=16, split=False):
    print("Loading data")
    random.seed(42)

    DATA_DIR = os.path.join(os.getcwd(), 'data', 'chop')

    labels = pd.read_csv(os.path.join(DATA_DIR, 'labels_cleaned_with_notes.csv')).fillna('')
    
    
    labels = labels.set_index('file')
    examples = [[v['text'], v['cats']] for _, v in labels.to_dict(orient='index').items()]
    for i, [_, v] in enumerate(examples):
        labels_tmp = [0 for _ in range(9)]

        if v:
            for label in v.split(';'):
                labels_tmp[int(label[0])] = 1 if label[1] == '+' else -1
        examples[i][1] = labels_tmp

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

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn)

        return train_dataloader, val_dataloader, test_dataloader
    else:
        dataset = SDOHDataset(examples)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn)
        
        return dataloader
    
