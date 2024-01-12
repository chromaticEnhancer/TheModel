import torch
import torchvision
from themodel.generator import UNet
from themodel.utils import CheckpointTypes, load_generator
from themodel.utils import normalize_image, denormalize_image

def colorize(input_path: str, output_path: str):
    model = UNet(in_channels=1, out_channels=3)
    load_generator(model=model, checkpoint_type=CheckpointTypes.COLOR_GENERATOR)

    image = torchvision.io.read_image(path=input_path, mode=torchvision.io.ImageReadMode.RGB)
    normalize = normalize_image(is_color=False)
    image = normalize(image/1.0)
    out = model(image.unsqueeze(0))
    out = out.squeeze(0).byte()
    
    torchvision.io.write_png(out, filename=output_path)



if __name__ == "__main__":
    colorize(input_path='./data/train/bw/goku.png', output_path='./output/generated_goku.png')
