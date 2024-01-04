import os
import cv2
import skimage.io as io
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from ubco_stage2.mil_model.model import StainNet

"""
Script to test the stainnet model.

Model source: https://github.com/khtao/StainNet
"""

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, img_folder, out_dir):
        self.data_dir = img_folder
        self.out_dir = out_dir
        self.img_ids, self.fpaths = self.load_records()

    def load_records(self):
        df = pd.read_csv(os.path.join(self.data_dir, "train_stage2.csv"))

        all_fpaths = []
        all_img_ids = []
        for img_id in df.image_id.unique():
            cur_path = "imgs/tiles1280_v3/{}".format(img_id)
            fpaths = os.listdir(os.path.join(self.data_dir, cur_path))
            fpaths = sorted(fpaths, key=lambda x: -float(x.split("_")[-1].split(".png")[0]))[:32]

            for fpath in fpaths:
                all_fpaths.append(os.path.join(self.data_dir, cur_path, fpath))
                all_img_ids.append(img_id)
        return all_img_ids, all_fpaths


    def __len__(self):
        return len(self.img_ids)
    
    def __getitem__(self, idx):
        fpath = self.fpaths[idx]
        img_id = self.img_ids[idx]
        
        out_dir = os.path.join(self.out_dir, img_id)
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)

        img = cv2.imread(fpath)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = np.transpose(img.astype(np.float32), (2, 0, 1))
        img = ((img / 255) - 0.5) / 0.5
        return img, fpath.replace("tiles1280_v3", "tiles1280_v4")
    
def norm(image):
    # image = np.array(image).astype(np.float32)
    image = image.transpose((2, 0, 1))
    image = ((image / 255) - 0.5) / 0.5
    image=image[np.newaxis, ...]
    image=torch.from_numpy(image)
    return image

def un_norm(image):
    image = image.cpu().detach().numpy()[0]
    image = ((image * 0.5 + 0.5) * 255).astype(np.uint8).transpose((1,2,0))
    return image
        

if __name__ == "__main__":
    model = StainNet().cuda()
    model.load_state_dict(torch.load("../data/models/StainNet-Public_layer3_ch32.pt"))

    dataset = CustomDataset(
        img_folder = "../data",
        out_dir = "../data/imgs/tiles1280_v4",
    )
    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=1,
        shuffle=False
    )

    with torch.no_grad():
        for batch in tqdm(dataloader):
            x, out_path = batch
            x = x.cuda()
            res = model(x)
            res = un_norm(res)
            io.imsave(out_path[0], res)
            
