import pandas as pd
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import balanced_accuracy_score

# CUDA_VISIBLE_DEVICES="" python test.py

df = pd.read_csv("./data/train_stage2.csv")
df = df[(df.source=="ubco")]

class_map = {'CC': 0, 'EC': 1, 'HGSC': 2, 'LGSC': 3, 'MC': 4}
labels = F.one_hot(
    torch.from_numpy(df.label.map(class_map).values), 
    num_classes=5,
    ).type(torch.DoubleTensor)


all_preds = []
all_labels = []
for img_id, label in zip(df.image_id.values, labels):
    pred = torch.load("./data/preds/{}.pt".format(img_id), map_location=torch.device('cpu'))
    pred = pred.type(torch.DoubleTensor)
    pred = F.softmax(pred, dim=0)

    all_preds.append(pred)
    all_labels.append(label)
all_preds = torch.stack(all_preds, dim=0).float()
all_labels = torch.stack(all_labels, dim=0).float()


weights = torch.tensor([0.2]*5)
best_weights = torch.tensor([0.2]*5)
best_score = 0.0

# {'CC': 0, 'EC': 1, 'HGSC': 2, 'LGSC': 3, 'MC': 4}
# [0.1112, 0.3052, 0.2084, 0.1897, 0.1231]

# Evolutionary algorithm (Add random noise until no more improvenments)
for _ in range(250):
    for i in range(100):
        if i == 0:
            noise = weights
            tmp_preds = all_preds * noise
        else:
            noise = weights + (torch.rand(5) - 0.5) * 0.10
            tmp_preds = all_preds * noise
        
        score = balanced_accuracy_score(np.argmax(all_labels, axis=1), np.argmax(tmp_preds, axis=1))
        if score > best_score:
            best_score = score
            best_weights = noise
    weights = best_weights

    print(best_score, best_weights)
print("-"*10, " Best ", "-"*10)
print(best_score, best_weights)
