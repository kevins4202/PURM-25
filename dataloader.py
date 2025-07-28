import os
import random
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch
from config import CAT_TO_LABELS

DUPLICATE_FILES = [
    "798507904.txt",
    "804045612.txt",
    "792948743.txt",
    "793084875.txt",
    "759511715.txt",
    "776431140.txt",
    "779183024.txt",
    "785000449.txt",
    "742651922.txt",
    "746965405.txt",
    "798713223.txt",
    "771650368.txt",
    "803876336.txt",
    "768815667.txt",
    "734332849.txt",
    "801051895.txt",
    "781773363.txt",
    "758037306.txt",
    "748444760.txt",
    "764662087.txt",
    "795228404.txt",
    "804257333.txt",
    "744173368.txt",
    "764679905.txt",
    "777218429.txt",
    "750399483.txt",
    "765925964.txt",
    "797805510.txt",
    "810089169.txt",
    "799345370.txt",
    "767847492.txt",
    "751097166.txt",
    "744375909.txt",
    "796718222.txt",
    "735442263.txt",
    "734766290.txt",
    "759861099.txt",
    "806902810.txt",
    "802376323.txt",
    "806340991.txt",
    "791441082.txt",
    "754207788.txt",
    "771649119.txt",
    "803613093.txt",
    "802346171.txt",
    "810011531.txt",
    "764551203.txt",
    "776150262.txt",
    "731767882.txt",
    "757699951.txt",
    "765754673.txt",
    "755951737.txt",
    "736280032.txt",
    "740381832.txt",
    "759064018.txt",
    "784302284.txt",
    "799456608.txt",
    "740299111.txt",
    "776453489.txt",
    "809754881.txt",
    "808807726.txt",
    "768215003.txt",
    "765872700.txt",
    "745793220.txt",
    "766106630.txt",
    "796974687.txt",
    "755307792.txt",
    "796549830.txt",
    "738793331.txt",
    "757989498.txt",
    "779497591.txt",
    "782298339.txt",
    "787368367.txt",
    "795900425.txt",
    "768708833.txt",
    "788560718.txt",
    "765431613.txt",
    "788583045.txt",
    "798526278.txt",
    "780098488.txt",
    "770313451.txt",
    "786497530.txt",
    "767815498.txt",
    "768868815.txt",
    "781911433.txt",
    "752706429.txt",
    "749442330.txt",
    "735867514.txt",
    "804043381.txt",
    "753611477.txt",
    "797621560.txt",
    "810379272.txt",
    "804543869.txt",
    "777108514.txt",
    "778881257.txt",
    "738955133.txt",
    "787757589.txt",
    "778414792.txt",
    "768893621.txt",
    "782515112.txt",
    "790879095.txt",
    "805302551.txt",
    "793785981.txt",
    "797019912.txt",
    "785453340.txt",
    "797105119.txt",
    "749043767.txt",
    "803840123.txt",
    "736891241.txt",
    "808752059.txt",
    "764575520.txt",
    "760372592.txt",
    "739680775.txt",
    "799862984.txt",
    "792265992.txt",
    "805427836.txt",
    "774849055.txt",
    "774321063.txt",
    "809181242.txt",
    "801412799.txt",
    "802577642.txt",
    "770627301.txt",
    "774903198.txt",
    "785974267.txt",
    "791668082.txt",
    "799056259.txt",
    "773076726.txt",
    "799524607.txt",
    "767493261.txt",
    "734432298.txt",
    "800335751.txt",
    "800106647.txt",
    "805799866.txt",
    "782974265.txt",
    "784246638.txt",
    "790246163.txt",
    "803186934.txt",
    "748893550.txt",
    "804639848.txt",
    "774407879.txt",
    "806415776.txt",
    "748125722.txt",
    "790423553.txt",
    "739682316.txt",
    "759203392.txt",
    "795129491.txt",
    "779854647.txt",
    "800373442.txt",
    "764479802.txt",
    "747970234.txt",
    "787106177.txt",
    "779730381.txt",
    "752106546.txt",
    "761088227.txt",
]


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


def get_dataloaders(batch_size=16, split=False, zero_shot=True):
    print("Loading data")
    random.seed(42)

    DATA_DIR = os.path.join(os.getcwd(), "data", "chop")

    labels = pd.read_csv(
        os.path.join(DATA_DIR, "labels_cleaned_with_notes.csv")
    ).fillna("")

    if not zero_shot:
        print("getting rid of duplicates")
        labels = labels[~labels["file"].isin(DUPLICATE_FILES)]

    labels = labels.set_index("file")
    examples = [
        [v["text"], v["cats"]] for k, v in labels.to_dict(orient="index").items()
    ]
    for i, [_, v] in enumerate(examples):
        labels_tmp = [0 for _ in range(len(CAT_TO_LABELS))]

        if v:
            for label in v.split(";"):
                labels_tmp[int(label[0])] = 1 if label[1] == "+" else -1
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

        train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=custom_collate_fn,
        )
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=custom_collate_fn,
        )
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=custom_collate_fn,
        )

        return train_dataloader, val_dataloader, test_dataloader
    else:
        dataset = SDOHDataset(examples)
        dataloader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn
        )

        return dataloader
