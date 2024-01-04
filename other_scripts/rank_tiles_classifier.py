import torch
import timm
import os
import cv2
import skimage.io as io
import numpy as np
from tqdm import tqdm

"""
Script to select tiles based on lightweight
"Tumor Tissue" classifier.
"""

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, img_id: str, img_dir: str, fnames: list):
        self.img_id = img_id
        self.img_dir = img_dir
        self.fnames = fnames

    def __len__(self):
        return len(self.fnames)
    
    def __getitem__(self, idx):
        fname = self.fnames[idx]
        full_fname = os.path.join(self.img_dir, fname)

        img = cv2.imread(full_fname)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img).permute(2,0,1) / 255.0

        return img, fname

def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load classifier
    model = timm.create_model(
            "efficientvit_b3.r288_in1k", 
            pretrained=True,
            num_classes=1,
        )
    model.load_state_dict(torch.load("./efficientvit_b3.r288_in1k_all.pt"))
    model = model.to(device)
    model.eval()

    # Load img tiles
    main_dir = "./temp_sfu"
    for img_id in os.listdir(main_dir):
        img_dir = os.path.join(main_dir, img_id)
        all_tiles = os.listdir(img_dir)

        # Create outdir
        outdir = "./temp_sfu_v2/{}/".format(img_id)
        if not os.path.exists(outdir):
            os.mkdir(outdir)
        else:
            print("ALREADY EXISTS: {}".format(outdir))
            continue

        # Dataset + Dataloader
        dataset = CustomDataset(
            img_id = img_id,
            img_dir = img_dir,
            fnames = all_tiles,
        )

        dataloader = torch.utils.data.DataLoader(
            dataset = dataset,
            batch_size = 16,
            num_workers = 4,
            pin_memory = True,
        )
        
        # Score all_tiles
        all_fnames = []
        all_scores = []
        for batch in tqdm(dataloader):
            imgs, fnames = batch
            imgs = imgs.to(device)
            with torch.no_grad():
                out = model(imgs)
                out = torch.sigmoid(out)

            all_scores.append(out.cpu().numpy())
            all_fnames.extend(fnames)
        all_scores = np.concatenate(all_scores).ravel()

        # Save top tiles
        top_score_idxs = np.argsort(all_scores)[::-1]
        for i in range(min(len(top_score_idxs), 64)):
            idx = top_score_idxs[i]
            score = np.round(all_scores[idx], 5)
            if i >= 16 and score < 0.001:
                break

            fname = all_fnames[idx]
            img = cv2.imread(os.path.join(img_dir, fname))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            io.imsave("./temp_sfu_v2/{}/{}_{}.png".format(img_id, fname.split(".")[0], score), img)
    return


if __name__ == "__main__":
    main()