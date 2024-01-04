import lightning.pytorch as pl
import torch
import torch.nn.functional as F
import os
import cv2
import albumentations as A
from typing import List

import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold
from ubco_stage2.augmentations import tma_augmentation

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, tile_size, num_tiles, top_n_imgs, min_n_imgs, val_fold, val_tta, train_all_data, is_train, transform=None):
        self.data_dir = data_dir
        self.tile_size = tile_size
        self.num_tiles = num_tiles
        self.top_n_imgs = top_n_imgs
        self.min_n_imgs = min_n_imgs
        self.val_fold = val_fold
        self.val_tta = val_tta
        self.train_all_data = train_all_data

        self.is_train = is_train
        self.transform = transform
        self.stratified_kfold = StratifiedKFold(n_splits=4, shuffle=True, random_state=0)
        self.imgs, self.labels, self.is_tmas = self.load_records()
        self.labelled_dict = self.load_labelled()
        
        # Heavy Augmentations
        P = 1.0
        self.train_tile_transform = A.Compose([
            A.RandomResizedCrop(self.tile_size, self.tile_size, scale=(0.5, 1.0)),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Transpose(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.3,
                scale_limit=0.3,
                rotate_limit=20,
                border_mode=4,
                p=P,
            ),
            A.OneOf(
                [
                    A.MotionBlur(blur_limit=3),
                    A.MedianBlur(blur_limit=3),
                    A.GaussNoise(var_limit=(10.0, 50.0)),
                ],
                p=P),
            A.OneOf(
                [
                    A.OpticalDistortion(distort_limit=0.5),
                    A.GridDistortion(num_steps=5, distort_limit=0.5),
                ],
                p=0.1,
            ),
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=25, val_shift_limit=20, p=P),
            A.RandomBrightnessContrast(brightness_limit=0.05, contrast_limit=0.05, p=P),
            A.OneOf([
                A.CoarseDropout(
                    max_height=int(1280 * 0.20),
                    min_height=int(1280 * 0.05),
                    max_width=int(1280 * 0.20),
                    min_width=int(1280 * 0.05),
                    min_holes=3,
                    max_holes=15,
                    fill_value=255,
                    p=1.0,
                    ),
                A.CoarseDropout(
                    max_height=int(1280 * 0.20),
                    min_height=int(1280 * 0.05),
                    max_width=int(1280 * 0.20),
                    min_width=int(1280 * 0.05),
                    min_holes=3,
                    max_holes=15,
                    fill_value=0,
                    p=1.0,
                    ),
            ], p=P)
        ])

        self.val_tile_transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Transpose(p=0.5),
        ])

    def load_records(self):
        """
        Loads metadata.
        """
        df = pd.read_csv(os.path.join(self.data_dir, "train_stage2.csv"))

        # Mislabelled Imgs
        mislabelled = {
            '15583': 'MC',
            '51215': 'LGSC', 
            '21432': 'CC',
            '50878': 'LGSC',
            '19569': 'MC',
            '38097': 'EC',
            '29084': 'CC',
            '63836': 'LGSC',
        }
        df['label'] = df['image_id'].map(mislabelled).fillna(df['label'])

        # Poorly classified images based on OOF
        drop_idxs = [
            # Based on OOF CV w/ 
            '41361', '29200', '18896', '36678', '39425', '49281', '64771', '28066', 
            '54506', '22654', '6363', '11559', '45185', '32636', '20312', '53402', 
            '37190', '46435', '29888', '50246', '22924', '42125', '1943', '17365', 
            '17738', '26124',
            # Based on Visual
            '1252', '1289',
            # Noli Alonso's list
            # Source: https://www.kaggle.com/competitions/UBC-OCEAN/discussion/445804#2559062
            '32035', '281', '3222', '5264', '9154', '12244', '31793', '32192', '33839', '41099', '52308', '63836' # Distorted
            ]
        df = df[~df.image_id.isin(drop_idxs)]


        # Creating mulitple DFs based on source + img_type
        ubco_tma_df = df[(df["is_tma"] == True) & (df["source"] == "ubco")].copy()
        ubco_wsi_df = df[(df["is_tma"] == False) & (df["source"] == "ubco")].copy()
        ext_df = df[df["source"] != "ubco"].copy()

        # Train all | 4Fold CV
        if self.train_all_data:
            if self.is_train == False:
                return [], [], []
            else:
                if self.tile_size > 1280:
                    df = df[df.is_tma == False].copy()
                df = pd.concat([df, df[df["label"].isin(["MC", "LGSC"])]], ignore_index=True)
        else:
            for i, (train_idx, valid_idx) in enumerate(self.stratified_kfold.split(ubco_wsi_df, ubco_wsi_df["label"].values)):
                # Manually selecting 0th fold here
                if i == self.val_fold:
                    if self.is_train:
                        # External data always in train
                        df = pd.concat([
                            ubco_wsi_df[ubco_wsi_df.fold != self.val_fold],
                            ext_df,
                            ],
                        ignore_index=True)

                        # Oversample small classes by 2x
                        df = pd.concat([df, df[df["label"].isin(["MC", "LGSC"])]], ignore_index=True).copy()
                    else:
                        # UBCO tma and val_fold in valid
                        df = pd.concat([
                            ubco_wsi_df[ubco_wsi_df.fold == self.val_fold],
                            ubco_tma_df, 
                            ], 
                            ignore_index=True).copy()
                        
        # df = df[(df.image_id.str.contains("36_")) | (df.image_id.str.contains("8_"))]

        # Label Mapping
        class_map = {'CC': 0, 'EC': 1, 'HGSC': 2, 'LGSC': 3, 'MC': 4}
        labels = F.one_hot(
            torch.from_numpy(df.label.map(class_map).values), 
            num_classes=5,
            ).type(torch.DoubleTensor)
 
        return df["image_id"].values, labels, df["is_tma"].values
    
    def load_labelled(self):
        """
        Load the file names for all tiles of known tissue. Based on
        labelled annotations from organizers.

        Source: https://www.kaggle.com/datasets/sohier/ubc-ovarian-cancer-competition-supplemental-masks
        """
        result_dict = {}
        directory = os.path.join(self.data_dir, "tiles_{}_known_pngs".format(self.tile_size))
        for root, dirs, files in os.walk(directory):
            subfolder = os.path.relpath(root, directory)
            file_names = [file for file in files if file.endswith(".png")]
            result_dict[subfolder] = file_names
        return result_dict
    
    def get_ink_color(self):
        """
        Random generation of ink color
        for staining.
        """
        ink_colors = [
            [(0,0,0), (15,15,50)], # black
            [(0,0,0), (15,15,130)], # blue
            [(10,150,0), (40,180,50)], # green
        ]
        ink_idx = np.random.randint(0, len(ink_colors))
        min_values, max_values = ink_colors[ink_idx]
        return tuple(np.random.randint(min_val, max_val + 1) for min_val, max_val in zip(min_values, max_values))
    
    def load_fnames(self, img_id: str) -> List[str]:
        """
        Load all images for a given img_id, sorted
        by score.

        Args:
            img_id (str): Image id
        
        Returns:
            fname (List[str]): List of image file paths
        """
        main_dir = os.path.join(self.data_dir, "imgs/tiles{}_v5_pngs/{}".format(self.tile_size, img_id))

        # Load all img_names
        fnames = [x for x in os.listdir(main_dir) if x.endswith(".png")]
        all_scores = [int(x.split("_")[0]) for x in fnames]

        # Sort by rank
        sorted_idxs = np.argsort(all_scores)
        fnames = [os.path.join(main_dir, fnames[i]) for i in sorted_idxs][:self.top_n_imgs]
        return fnames

    
    def __getitem__(self, index):

        img_id = self.imgs[index]
        label = self.labels[index]
        is_tma = self.is_tmas[index]

        # Select imgs from within masks if available
        if self.tile_size in [1280, 2048] and self.is_train and img_id in self.labelled_dict and np.random.random() < 1.0:
            main_dir = os.path.join(self.data_dir, "tiles_{}_known_pngs/{}".format(self.tile_size, img_id))
            img_fnames = [os.path.join(main_dir, x) for x in os.listdir(main_dir) if x.endswith(".png")]
            np.random.shuffle(img_fnames)

            # Fill with more imgs
            if len(img_fnames) < self.num_tiles:
                img_fnames = img_fnames + self.load_fnames(img_id)[:self.num_tiles - len(img_fnames)]

        # Select Imgs w/ out masks
        else:
            img_fnames = self.load_fnames(img_id)

        # Select tiles
        if len(img_fnames) < self.num_tiles:
            img_fnames = img_fnames + np.random.choice(img_fnames, size=self.num_tiles-len(img_fnames), replace=True).tolist()

        # Shuffle image order if training
        if self.is_train:
            np.random.shuffle(img_fnames)
            img_fnames = img_fnames[:self.num_tiles]
        else:
            img_fnames = img_fnames[:12]
            np.random.shuffle(img_fnames)
            img_fnames = img_fnames[:self.num_tiles]

        # Random Mixup
        if self.is_train and np.random.random() < 0.4:

            # Get img_id with same label
            equality_tensor = torch.eq(self.labels, label)
            row_indices = torch.all(equality_tensor, dim=1)
            selected_rows = torch.nonzero(row_indices).squeeze()
            random_img_id = self.imgs[selected_rows[np.random.randint(0, len(selected_rows))].item()]

            mixup_fnames = self.load_fnames(random_img_id)[:self.top_n_imgs]
            np.random.shuffle(mixup_fnames)

            for i in range(min(2, len(mixup_fnames))):
                img_fnames[i] = mixup_fnames[i]

        # Occasional Ink staining
        if self.is_train and np.random.random() < 0.05:
            ink_color = self.get_ink_color()
            ink_transform = A.RandomSunFlare(flare_roi=(0, 0, 1, 1), 
                            angle_lower=0, angle_upper=1, 
                            num_flare_circles_lower=0, 
                            num_flare_circles_upper=1, 
                            src_radius=self.tile_size, src_color=ink_color,
                            p=0.4,
                            )
        else:
            ink_transform = None

        # # TMA Augmentation: (Not use in the end)
        # if np.random.random() < 0.10:
        #     tma_transform = True
        # else:
        #     tma_transform = False
        tma_transform = False

        # Apply TTA during validation
        tiles = []
        if self.is_train: val_tta = 1
        else: val_tta = self.val_tta

        # Load tiles 
        all_tiles = []
        for i in range(val_tta): 
            tiles = []
            for j, fname in enumerate(img_fnames[:self.num_tiles]):
                img_tile = cv2.imread(fname)
                img_tile = cv2.cvtColor(img_tile, cv2.COLOR_BGR2RGB)

                # Tile Transformation
                if self.is_train:
                    img_tile = self.train_tile_transform(image=img_tile)['image']
                    if tma_transform:
                        img_tile = tma_augmentation(img_tile, crop_type=j)
                    if ink_transform:
                        img_tile = ink_transform(image=img_tile)["image"]
                else:
                    img_tile = self.val_tile_transform(image=img_tile)['image']

                # Insert tile into stack
                tiles.append(img_tile)
            tiles = np.stack(tiles, axis=0)
            all_tiles.append(tiles)
        all_tiles = np.stack(all_tiles, axis=0)

        # # Sanity Check: Check images are generated properly
        # import skimage.io as io
        # io.imsave("{}_{}.png".format(img_id, int(self.is_train)), tiles[0, ...])
        # # assert 1 > 2

        # Permute to (batch_size, val_tta, channels, H, W)
        tiles = torch.from_numpy(all_tiles).permute(0, 1, 4, 2, 3).float() / 255.0
        return tiles, label, is_tma, img_id
    
    def __len__(self):
        return len(self.imgs)


class CustomDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        tile_size: int,
        num_tiles: int,
        top_n_imgs: int,
        min_n_imgs: int,
        val_fold: int,
        val_tta: int,
        train_all_data: bool,
        batch_size: int,
        num_workers: int,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.train_transform, self.val_transform = self._init_transforms()
    
    def _init_transforms(self):
        # Transforms in CustomDataset
        train = A.Compose([
        ])
        valid = A.Compose([
        ])
        return train, valid

    def setup(self, stage):        
        if stage == "fit" or stage is None:
            self.train_dataset = self._dataset(transform=self.train_transform, is_train=True)
            self.val_dataset = self._dataset(transform=self.val_transform, is_train=False)
            print("Dataset sizes. T: {:_}, V: {:_}".format(len(self.train_dataset), len(self.val_dataset)))

        elif stage == "validate":
            self.val_dataset = self._dataset(transform=self.val_transform, is_train=False)
            
    def _dataset(self, transform, is_train):
        return CustomDataset(
            data_dir=self.hparams.data_dir,
            tile_size=self.hparams.tile_size,
            num_tiles=self.hparams.num_tiles,
            top_n_imgs=self.hparams.top_n_imgs,
            min_n_imgs=self.hparams.min_n_imgs,
            val_fold=self.hparams.val_fold,
            val_tta=self.hparams.val_tta,
            train_all_data=self.hparams.train_all_data,
            is_train=is_train,
            transform=transform,
            )
    
    def train_dataloader(self):
        return self._dataloader(self.train_dataset, is_train=True)
    
    def val_dataloader(self):
        return self._dataloader(self.val_dataset, is_train=False)

    def _dataloader(self, dataset, is_train=False):
        return torch.utils.data.DataLoader(
            dataset,
            batch_size = self.hparams.batch_size if is_train else 1,
            num_workers = self.hparams.num_workers,
            pin_memory = True,
            drop_last = is_train,
            shuffle = is_train,
        )