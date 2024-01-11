import os
import zipfile as zf
from glob import glob
from PIL import Image

import os
from glob import glob

def remove_coloured(main_output_dir='./output/'):
    for extracted_file in glob(os.path.join(main_output_dir, "*.jpg")):
        print('rm coloured:', extracted_file)
        os.remove(extracted_file)


def remove_coloured_impostor(main_output_dir ='./output/'):
    for png_name in os.listdir(main_output_dir):
        if not (png_name.endswith('.png') and (Image.open(os.path.join(main_output_dir, png_name)).mode== 'L')):

            print('rm incompatible:',png_name)
            os.remove(os.path.join(main_output_dir, png_name))

def cbz_to_png(cbz_file_path, main_output_dir):
    output_subfolder = main_output_dir
    #title = os.path.splitext(os.path.basename(cbz_file_path))[0]
    #output_subfolder=os.path.join(main_output_dir, title)
    os.makedirs(output_subfolder, exist_ok=True)
    with zf.ZipFile(cbz_file_path, 'r') as cbz_object:
        cbz_object.extractall(output_subfolder)

def main(input_dir= './inputs',  main_output_dir = './output/'):
   
    for filename in os.listdir(input_dir):
        try:
            print('\nextracting: ', filename)
            if filename.endswith('.cbz'):
                cbz_file_path = os.path.join(input_dir, filename)
                cbz_to_png(cbz_file_path, main_output_dir)
            print('\ndone!')
        except:
            print('skipped: ', filename)
            continue
    # remove_coloured(main_output_dir)
    # remove_coloured_impostor(main_output_dir)

if __name__ == "__main__":
    main()
