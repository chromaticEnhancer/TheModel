from PIL import Image, ImageStat
import numpy as np
import os

# if not extracted:
# import preprocess
# preprocess.main()
# preprocess.remove_coloured()
# preprocess.remove_coloured_impostor()

def BW_img_mean_std(img_path):
    """
    For BW image:
    returns mean and stddev of 1 channel as m,s
    """
    try:
        img = Image.open(img_path)
        img_data= np.asarray(img)
        return img_data.mean(), img_data.std()
    except:
        print(f"Error Opening file: {img_path}")
        return 


def C_image_mean_std(img_path):
    """
    For Coloured image:
    returns mean and stddev of 3 channels as m1,m2,m3,s1,s2,s3
    """
    try:
        img=Image.open(img_path)
        stat= ImageStat.Stat(img)
        return tuple(stat.mean[:3]+stat.stddev[:3])
    except:
        print(f"Error Opening file: {img_path}")
        return 

def main():
    bw_img_path = './goodeg.png'
    c_img_path = './testpng.png'

    print(BW_img_mean_std(bw_img_path))
    print(C_image_mean_std(c_img_path))

if __name__ == '__main__':
    main()


import torch
from torchvision.io import read_image

def get_mean_std(img_path):
    """
    Calculate the mean and standard deviation of an image tensor

    Args:
        img_path (str): The path to the image file.

    Returns:
        tuple: A tuple containing two lists - the means and standard deviations of the image channels.

    """
    means = []
    stds = []
    try:
        img = read_image(img_path)
        for i in range(img.shape[0]):
            means.append(img[i, :, :].float().mean().item())
            stds.append(img[i, :, :].float().std().item())
        return means, stds
    except Exception as e:
        print(f"Error Opening file: {img_path}. Error: {e}")
        return
    
print(get_mean_std('./goodeg.png'))
print(get_mean_std('./c_goodeg.jpg'))