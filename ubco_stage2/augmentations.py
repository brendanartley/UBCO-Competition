import PIL
from PIL import Image, ImageDraw, ImageFilter
import numpy as np

from typing import Tuple

"""
Script that takes random crops out of an image 
to look like TMAs tiles.
"""

def get_random_greyish_rgb() -> Tuple[int, int, int]:
    """
    Generates random greyish RGB.
    
    Args:
        None
        
    Returns:
        rgb (Tuple): Random RGB tuple
    """
    g = np.random.randint(245,255)
    rgb = (g,g,g)
    return rgb

def get_random_purplish_rgb() -> Tuple[int, int, int]:
    """
    Generates random purplish RGB.
    
    Args:
        None
        
    Returns:
        rgb (Tuple): Random RGB tuple
    """
    red = np.random.randint(150, 175)
    green = np.random.randint(120, 150)
    blue = np.random.randint(190, 200)

    rgb = (red, green, blue)
    return rgb

def create_random_background(in_img: PIL.Image):
    """
    Create a random background for TMA
    augmentation.
    
    Args:
        in_img (PIL.Image): Image to be augmented
        
    Returns:
        back_img (PIL.Image): Random background image
    """
    # Params
    width = in_img.width
    height = in_img.height
    
    # Background color
    if np.random.random() > 0.1:
        color = get_random_purplish_rgb()
    else:
        color = get_random_greyish_rgb()
    
    # Create background image
    img = Image.new('RGB', (width, height), color)
    draw = ImageDraw.Draw(img)

    # Add noise
    noise_intensity = 5
    for _ in range(width * height // np.random.randint(7, 50)):
        x = np.random.randint(0, width - 1)
        y = np.random.randint(0, height - 1)
        color = np.random.randint(0, noise_intensity)
        draw.point((x, y), fill=(color, color, color))

    # Smooth noise
    back_img = img.filter(ImageFilter.GaussianBlur(radius=np.random.randint(4,13)))
    return back_img

def tma_augmentation(img: PIL.Image, crop_type: int = 0) -> PIL.Image:
    """
    Augments a tile from WSI image
    to look like TMA image.
    
    Args:
        img (PIL.Image): Image to augment
        i (int): Crop type. Value 0-7 inclusive.
        
    Returns:
        img_final (PIL.Image): Augmented image
        
    """

    # Margin parameters
    img = Image.fromarray(np.uint8(img)).convert('RGB')
    w, h = img.width, img.height

    # Create a white background
    img_final = create_random_background(img)

    # Create a mask with a white circle
    mask = Image.new('L', (w,h), 0)
    draw = ImageDraw.Draw(mask)
    
    # Crop around circle
    w = np.random.uniform(0.5*w, 1.1*w)
    h = np.random.uniform(0.5*h, 1.1*h)
    all_transforms = [
        [(0, 0),(w*2, h*2)], # top left
        [(-w//2, 0), (w+w//2, h*2)], # top
        [(-w, 0), (w, h*2)], # top right
        [(0, -h//2),(w*2, h*2-h//2)], # mid left
        # [(-w//2, -h//2), (w+w//2, h+h//2)], # mid
        [(-w, -h//2), (w, h*2-h//2)], # mid right
        [(0, -h),(w*2, h)], # bottom left
        [(-w//2, -h), (w+w//2, h)], # bottom
        [(-w, -h), (w, h)], # bottom right
    ]
    
    # Add variability in crop
    t = all_transforms[crop_type]
    w_offset = w // 2
    h_offset = h // 2
    w_offset = np.random.randint(-w_offset, w_offset)
    h_offset = np.random.randint(-h_offset, h_offset)
    t[0] = (t[0][0]+w_offset, t[0][1]+h_offset)
    t[1] = (t[1][0]+w_offset, t[1][1]+h_offset)
    draw.pieslice(t, 0, 360, fill=255)

    # Combine mask + img
    img_final.paste(img, (0, 0), mask)
    return np.asarray(img_final)
