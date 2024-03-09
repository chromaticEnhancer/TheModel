import os
from themodel.options.test_options import TestOptions
from themodel.data import CreateDataLoader
from themodel.models import create_model
from themodel.util.util import tensor2im, save_image
from skimage.transform import resize
import ntpath

def colorize(imgPath):
    opt = TestOptions().parse()
    try:
        opt.dataroot = imgPath
    except:
        print("given path of image incorrect")
    # hard-code some parameters for test
    opt.num_threads = 1   # test code only supports num_threads = 1
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # no shuffle
    opt.no_flip = True    # no flip
    opt.display_id = -1   # no visdom display
    opt.dataset = 'single'
    opt.name = 'demo'
    opt.resize_or_crop = 'scale_width'
    opt.loadSize = 512
    opt.fineSize = 512
    aspect_ratio = opt.aspect_ratio
    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    model = create_model(opt)
    model.setup(opt)
    # create a website
    # test with eval mode. This only affects layers like batchnorm and dropout.
    # pix2pix: we use batchnorm and dropout in the original pix2pix. You can experiment it with and without eval() mode.
    # CycleGAN: It should not affect CycleGAN as CycleGAN uses instancenorm without dropout.
    for data in dataset:
        model.set_input(data)
        model.test()
        visuals = model.get_current_visuals()
        img_path = model.get_image_paths()
    
        print(f'processing {img_path}')
        
        for _,im_data in visuals.items():
            im = tensor2im(im_data)
            filename = ntpath.basename(img_path[0]).split('.')[0]

            image_name = f"colored_{filename}.png" 
            # create a subdirectory named colored inside opt.dataroot
            if not os.path.exists(opt.dataroot + '/colored/'):
                os.makedirs(opt.dataroot + '/colored/')
            
            save_path = os.path.join(opt.dataroot + '/colored/', image_name)
            h, w, _ = im.shape
            if aspect_ratio > 1.0:
                im = resize(im, (h, int(w * aspect_ratio)), mode='reflect', anti_aliasing=True)
            if aspect_ratio < 1.0:
                im = resize(im, (int(h / aspect_ratio), w), mode='reflect', anti_aliasing=True)
            save_image(im, save_path)
        
            
