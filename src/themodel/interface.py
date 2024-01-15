import torchvision
from themodel import settings
from themodel.generator import UNet
from themodel.utils import CheckpointTypes, load_generator
from themodel.utils import normalize_image, denormalize_image

settings.DEVICE = 'cuda'
settings.IMAGE_HEIGHT = 184
settings.IMAGE_WIDTH = 128

def colorize(input_path: str, output_path: str):
    model = UNet(in_channels=3, out_channels=3)
    load_generator(model=model, checkpoint_type=CheckpointTypes.COLOR_GENERATOR)

    image = torchvision.io.read_image(path=input_path, mode=torchvision.io.ImageReadMode.GRAY)
    normalize = normalize_image(is_color=False)
    
    image = normalize(image / 255.0)
    image = image.repeat(3, 1, 1)
    out = model(image.unsqueeze(0))
    out = out * 255
    out = out.squeeze(0).byte()

    torchvision.io.write_png(out, filename=output_path)


if __name__ == "__main__":
    colorize(input_path='./data/train/bw/zang.png', output_path='.color.png')