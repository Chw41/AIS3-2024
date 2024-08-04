from PIL import Image, ImageChops

def is_same_image(img1: Image.Image, img2: Image.Image) -> bool:
    return ImageChops.difference(img1, img2).getbbox() == None
