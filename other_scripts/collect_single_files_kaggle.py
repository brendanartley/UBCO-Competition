import kaggle
import os

"""
Script to download individual large files from Kaggle.
"""

mask_ids = [
             39146, 19030, 49281, 45185, 10896, 63941, 18138, 37655, 61823, 24617, 1952, 33976, 18568, 24759, 11557, 
             45630, 36063, 35754, 39425, 52420, 14424, 61424, 8279, 16325, 48502, 51346, 38901, 36678, 1925, 31033, 
             8531, 21910, 10143, 56947, 38019, 34277, 17291, 21432, 28562, 56221, 16986, 17174, 50304, 59760, 6281, 
             15671, 15486, 4797, 38048, 6898, 39466, 37190, 14542, 26219, 30738, 11263, 46815, 54949, 16876, 56861, 
             50048, 18607, 14401, 26533, 17854, 66, 26950, 15209, 19255, 4211, 39172, 29200, 6558, 9183, 36499, 22740, 
             37307, 21445, 1020, 42549, 17738, 10800, 2666, 39728, 47960, 59515, 38849, 27245, 11431, 63165, 33708, 
             15470, 5851, 20316, 5251, 5992, 22155, 12442, 61493, 61320, 15188, 16064, 63121, 10252, 53688, 34247, 
             32432, 50878, 13987, 10246, 12522, 56875, 28393, 48506, 7329, 35239, 57162, 19569, 39258, 65533, 65022, 
             21929, 31383, 22489, 43390, 40888, 5852, 15139, 58895, 1101, 46688, 47105, 27315, 28821, 22221, 9658, 
             6951, 49872, 22425, 9697, 65094, 27950, 51893, 43815, 1252, 58947, 2227, 44232, 46139, 4963, 39255, 55287
            ]

for i, val in enumerate(mask_ids):
    try:
        os.system("kaggle competitions download -c UBC-OCEAN -f train_thumbnails/{}_thumbnail.png -p ./tmp_thumbs".format(val))
        os.system("unzip ./tmp_thumbs/{}_thumbnail.png.zip -d tmp_thumbs/ && rm -f ./tmp_thumbs/{}_thumbnail.png.zip".format(val, val))

        os.system("kaggle competitions download -c UBC-OCEAN -f train_images/{}.png -p ./tmp_imgs".format(val))
        os.system("unzip ./tmp_imgs/{}.png.zip -d tmp_imgs/ && rm -f ./tmp_imgs/{}.png.zip".format(val, val))
    except:
        print("FAILED FAILED FAILY FAIL", val)