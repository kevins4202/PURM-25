import os
import random

# Categories and mapping
cats = ["employment", "childcare", "transportation", "housing", "food_insecurity", "financial_strain", "permanency", "substance_abuse", "home", "community", "noneeds"]
catstoi = {cat: i for i, cat in enumerate(cats)}

# Directory containing the category txt files
DATA_DIR = os.path.join(os.getcwd(), 'data', 'generated')

# Read all sentences for each category
cat_sentences = {}
for cat in cats:
    path = os.path.join(DATA_DIR, f"{cat}.txt")
    with open(path, 'r') as f:
        # Remove empty lines and strip whitespace
        cat_sentences[cat] = [line.strip() for line in f if line.strip()]

# Number of notes to generate
NUM_NOTES = 480

with open(os.path.join(DATA_DIR, 'generated_notes.txt'), 'w') as out_f:
    for _ in range(NUM_NOTES):
        n_cats = random.randint(1, len(cats) - 1)
        chosen_cats = random.sample(cats[:-1], n_cats)
        indices = [catstoi[cat] for cat in chosen_cats]
        sentences = [random.choice(cat_sentences[cat]) for cat in chosen_cats]
        note = ' '.join(sentences)
        indices_str = ','.join(str(idx) for idx in sorted(indices))
        out_f.write(f"{indices_str}\t{note}\n")
    for noneed in cat_sentences["noneeds"]:
        out_f.write(f"{catstoi["noneeds"]}\t{noneed}\n")
