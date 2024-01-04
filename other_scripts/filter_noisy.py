import pandas as pd
import torch
import torch.nn.functional as F

"""
Script to filter noisy training images using
Euclidian Norm and threshold.
"""

threshold = 1.2
df = pd.read_csv("./data/train_stage2.csv")
df = df[(df.source=="ubco")]

class_map = {'CC': 0, 'EC': 1, 'HGSC': 2, 'LGSC': 3, 'MC': 4}
labels = F.one_hot(
    torch.from_numpy(df.label.map(class_map).values), 
    num_classes=5,
    ).type(torch.DoubleTensor)


all_dists = {}
for img_id, label in zip(df.image_id.values, labels):
    print(img_id, label)
    pred = torch.load("./data/preds/{}.pt".format(img_id), map_location=torch.device('cpu'))
    pred = pred.type(torch.DoubleTensor)
    pred = F.softmax(pred)

    # L2 norm / Euclidian Norm
    dist = torch.norm(pred - label, p=2).item()

    all_dists[img_id] = {
        "img_id": img_id,
        "pred": [round(float(item), 5) for item in pred],
        "label": [int(item) for item in label],
        "dist": round(dist, 5),
    }

c = 0
all_dists = dict(sorted(all_dists.items(), key=lambda x: -x[1]["dist"]))
denoise_arr = []
for k in list(all_dists.keys()):
    if all_dists[k]["dist"] < threshold:
        break
    c+=1
    denoise_arr.append(all_dists[k]["img_id"])
    print(all_dists[k])

print("Size: ", len(all_dists))
print("Missed: ", c)
print("Pct: {:.3f} %".format(100*c / len(all_dists)))
print(denoise_arr)
