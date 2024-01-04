import os
os.environ['VIPS_DISC_THRESHOLD'] = '50gb'
os.environ['VIPS_CONCURRENCY'] = '4'

import numpy as np
import pandas as pd
import skimage.io as io
from scipy.ndimage import zoom
import albumentations as A
import cv2

import pyvips
# import heapq
import pickle
import gc
import os
import time
from datetime import timedelta
import multiprocessing as mp
from types import SimpleNamespace

"""
Script to select create a set of known 
tumor tissue tiles.
"""

config = SimpleNamespace(
    # num_tiles = 10,
    tile_size = 1280, # Fails on >= 742 (Need to change TMA crop function if we want larger)
    test_image_n = 200,
    image_path = "./tmp_imgs/",
    thumbnail_path = "./tmp_thumbs/",
    mask_image_path = "./masks/",
    # image_path = "./test_imgs/"
    outdir_format = "./tiles_{}_known/{}",
)

def load_idxs():
    idxs = os.listdir(config.image_path)
    tmas = [False]*len(idxs)
    return idxs, tmas

def split_df(df, n_splits=3, split=0):
    """
    Returns (start, end) idxs for splitting a df
    into chunks. -1 returns all rows.
    """
    
    # Get size of each split
    split_sizes = [len(df)//n_splits]*n_splits
    for i in range(len(df)%n_splits):
        split_sizes[i]+=1
        
    # Get start:end idxs
    split_idxs = [0]*n_splits
    start = 0
    for i in range(n_splits):
        split_idxs[i] = (start, start + split_sizes[i])
        start = split_idxs[i][1]
        
    return split_idxs[split]

def largest_nonzero_seq(arr):
    """
    Find largest nonzero sequence in 1D numpy array.
    """
    # Find non-zero sequences
    non_zero_mask = (arr != 0).astype(int)
    seq_arr = np.diff(np.concatenate(([0], non_zero_mask, [0])))
    
    # Find start + end idxs
    seq_start = np.where(seq_arr == 1)[0]
    seq_end = np.where(seq_arr == -1)[0]
    
    # Find longest seq
    non_zero_lengths = seq_end - seq_start
    seq_idx = np.argmax(non_zero_lengths)
    
    return seq_start[seq_idx], seq_end[seq_idx]

def crop_image(thumb_image, threshold=0.15, viz=False):
    """
    Greedy method to crop largest object from WSI thumbnail image.
    """
    # Load thumbnail
    height, width = thumb_image.shape[0], thumb_image.shape[1]
    
    # Calculating X boundaries
    vs = np.sum(np.sum(thumb_image, axis=0), axis=-1)
    left, right = largest_nonzero_seq(vs)
    
    # Calculating Y boundaries
    bottom, top = 0, 0
    vs = np.sum(np.sum(thumb_image[:, left:right, :], axis=1), axis=-1)
    top, bottom = largest_nonzero_seq(vs)
    
    # Calculate crop ratio (for loading larger image)
    crop_ratios = (top/height, bottom/height, left/width, right/width)
    return crop_ratios

def tune_crop_ratio(image, crop_ratios):
    """
    Updates crop ratios to be divisible by tile size.
    """
    # Range after Crop 1
    top = int(image.height*crop_ratios[0])
    bottom = int(image.height*crop_ratios[1])
    left = int(image.width*crop_ratios[2])
    right = int(image.width*crop_ratios[3])

    # Crop 2: Crop to multiple of tile size
    h_diff = (bottom - top) % config.tile_size
    w_diff = (right - left) % config.tile_size

    # Trim from each side
    left += w_diff//2
    right -= w_diff//2 + (w_diff%2)
    top += h_diff//2
    bottom -= h_diff//2 + (h_diff%2)
    
#     # DEBUG
#     print("diffs", h_diff, w_diff)
#     print(top, bottom, left, right)
#     print("should be zero: ", (bottom - top)%config.tile_size, (right - left)%config.tile_size)
#     print("processed size: {:_} x {:_}".format(right - left, bottom - top))
    
    # Updated ratios
    crop_ratios = (top/image.height, bottom/image.height, left/image.width, right/image.width)
    width_tiles_n = (right - left) / config.tile_size
    height_tiles_n = (bottom - top) / config.tile_size
    
    return crop_ratios, width_tiles_n, height_tiles_n, top, bottom, left, right

def next_largest_divisible(num, divisor):
    """
    Returns next largest integer divisible by divisor.
    """
    if num % divisor == 0:
        return num
    else:
        return num + (divisor - num % divisor)
    
def expand_tma(img, ts):
    """
    Expand TMA image using mirroring.
    """
    # Center crop
    transform = A.CenterCrop(height=ts//2, width=ts//2)
    img = transform(image=img)['image']
    
    # Double image size w/ mirroring
    final_img = np.zeros((ts, ts, 3), dtype=np.uint8)    
    final_img[:ts//2, :ts//2, :] = img[:, :, :]
    final_img[ts//2:, :ts//2, :] = img[::-1, :, :]
    final_img[:ts//2, ts//2:, :] = img[:, ::-1, :]
    final_img[ts//2:, ts//2:, :] = img[::-1, ::-1, :] 
    return final_img

def get_xy_pairs(img_id, thumb_image, crop_ratios, width_tiles_n, height_tiles_n, left, top, debug=False):
    
    # Crop to updated ratio
    height, width = thumb_image.shape[0], thumb_image.shape[1]
    crop = thumb_image[
        int(crop_ratios[0]*height):int(crop_ratios[1]*height), 
        int(crop_ratios[2]*width):int(crop_ratios[3]*width), 
        :,
    ]
    
    # Find black pixels and replace them with white
    black_pixels = np.all(crop == [0, 0, 0], axis=-1) # Replace mask pixels w/ white
    crop[black_pixels] = np.array([255, 255, 255])
    
    # Resize to a multiple of tiles in full image
    new_height = next_largest_divisible(crop.shape[0], height_tiles_n)
    new_width = next_largest_divisible(crop.shape[1], width_tiles_n)
    
    # Resize image
    y_scale = new_height / crop.shape[0]
    x_scale = new_width / crop.shape[1]
    
    crop = zoom(crop, (y_scale, x_scale, 1), order=0, mode="nearest")
    if debug: print(crop.shape, width_tiles_n, height_tiles_n, new_width, new_height, x_scale, y_scale)
    
    h_step = int(crop.shape[0] / height_tiles_n)
    h_start = int(h_step // 2)
    h_end = int(crop.shape[0] - h_start)
    
    w_step = int(crop.shape[1] / width_tiles_n) 
    w_start = int(w_step // 2)
    w_end = int(crop.shape[1] - w_start)
    
    if debug: print(h_step, h_start, h_end, w_step, w_start, w_end)
    
    xy_pairs = []
    c = 0
    
    # 1. Iteration w/ strict conditions
    h_iters = range(h_start, h_end, h_step)
    w_iters = range(w_start, w_end, w_step)
    for i, h in enumerate(h_iters):
        for j, w in enumerate(w_iters):
            c+=1
            # Skip white tile areas
            if np.all(crop[h, w, :] == np.array([255, 255, 255])):
                continue
            
            if debug: crop[h:h+20, w:w+20, :] = np.full((20, 20, 3), [220, 20, 50], dtype=np.uint8)
            xy_pairs.append((j, i)) # (width, height)  
                
    # Add offsets and convert to full image dimensions
    xy_pairs = [(left+(x*config.tile_size), top+(y*config.tile_size)) for x,y in xy_pairs]
    return xy_pairs

def process_image(img_id, is_tma):
    # Load full img w/ pyvips
    start = time.time()
    image = pyvips.Image.new_from_file(os.path.join(config.image_path, img_id))    
    mask_image = pyvips.Image.new_from_file(os.path.join(config.mask_image_path, img_id))
    print("original size: {:_} x {:_}".format(image.width, image.height))

    # Convert black pixels to white
    mask = (image == 0).bandand()
    image = mask.ifthenelse([255, 255, 255], image)
    del mask

    # WSI 
    if is_tma == False:
        thumb_path = img_id.split(".")[0] + "_thumbnail.png"
        thumb_image = io.imread(os.path.join(config.thumbnail_path, thumb_path))
        crop_ratios = crop_image(thumb_image)

        # Update crop_ratios
        crop_ratios, width_tiles_n, height_tiles_n, top, bottom, left, right = tune_crop_ratio(image, crop_ratios)

        # Get X,Y pairs from the thumbnail image
        xy_pairs = get_xy_pairs(img_id, thumb_image, crop_ratios, width_tiles_n, height_tiles_n, left, top, debug=False)


        # Rank tiles
        all_scores = []
        has_red_total = 0
        for x,y in xy_pairs[:config.test_image_n]:
            tile_raw = image.crop(x, y, config.tile_size, config.tile_size)
            tile = tile_raw.numpy()
            tile_mask = mask_image.crop(x, y, config.tile_size, config.tile_size).numpy()
            
            # PCT of tile that is red..
            tile_fname = "{}_{}.png".format(x, y)
            pct_tumor = np.count_nonzero(tile_mask[:, :, 0]) / tile_mask[:, :, 0].size
            if pct_tumor >= 0.3:
                # Create outdir
                outdir = config.outdir_format.format(config.tile_size, img_id.split(".")[0])
                if not os.path.exists(outdir):
                    os.mkdir(outdir)
                tile_raw.write_to_file("{}/{}".format(outdir, tile_fname))
    return

def main():
    start = time.time()    
    idxs, img_tmas = load_idxs()

    # Process imgs
    print("-"*10 + " Count: {} ".format(len(idxs)) + "-"*10)
    all_pairs = [(img_id, is_tma) for img_id, is_tma in zip(idxs, img_tmas)]
    
    # 4 processors = OOM
    with mp.Pool(processes=15) as pool:
        results = pool.starmap(process_image, all_pairs)
        
    # Summary
    elapsed = time.time() - start
    print("Elapsed: {:0>8}".format(str(timedelta(seconds=int(elapsed)))))

if __name__ == "__main__":
    main()