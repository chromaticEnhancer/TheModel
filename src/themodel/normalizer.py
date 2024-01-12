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


# TODO: decide per image, batch or entire
 
#### FOR WHOLE DATASET, WORKAROUND STD

# def BW_list_mean_std(img_dir):
#     dataset_means=[]
#     dataset_stds=[]
#     for filename in os.listdir(img_dir):
#         filepath= os.path.join(img_dir, filename)
#         img_mean, img_std= BW_img_mean_std(filepath)
        
#         dataset_means.append(img_mean)
#         dataset_stds.append(img_std)
    
#     return dataset_means, dataset_stds

#combined_dataset_mean = np.mean(means)
#combined_dataset_sd = np.mean(stds)
#print(combined_dataset_mean, combined_dataset_sd)

