from PIL import Image
import os
import multiprocessing as mp

"""
Script to convert all images in all subdirectories
from JPGs to PNGs.
"""

def convert_png_to_jpg(input_folder, output_folder):
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Loop through all files in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith(".png"):
            # Open the PNG image
            png_path = os.path.join(input_folder, filename)
            img = Image.open(png_path)

            # Create the corresponding JPG filename in the output folder
            jpg_filename = os.path.splitext(filename)[0] + ".jpg"
            jpg_path = os.path.join(output_folder, jpg_filename)

            # Convert and save as JPG
            img.convert("RGB").save(jpg_path)
    print(output_folder)

if __name__ == "__main__":
    # Specify your input and output folders
    input_folder = "./data/tiles_1280_known"
    output_folder = "./data/tiles_1280_known_jpgs"

    all_dirs = []
    for main_dir in os.listdir(input_folder):
        if "." in main_dir:
            continue

        cur_dir = os.path.join(input_folder, main_dir)
        out_dir = os.path.join(output_folder, main_dir)

        all_dirs.append((cur_dir, out_dir))

    with mp.Pool(processes=25) as pool:
        pool.starmap(convert_png_to_jpg, all_dirs)
