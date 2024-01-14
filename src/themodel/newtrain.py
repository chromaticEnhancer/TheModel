import torch
import torchvision
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader

from themodel import settings
from themodel.generator import UNet
from themodel.discriminator import PatchGAN
from themodel.dataset import BWColorMangaDataset
from themodel.utils import save_model, CheckpointTypes, save_plots

def get_models() -> tuple[nn.Module, nn.Module, nn.Module]:
    generator = UNet(in_channels=1, out_channels=3)
    discriminator = PatchGAN(in_channels=3)
    vgg16 = torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.DEFAULT).eval().to(settings.DEVICE)

    for param in vgg16.parameters():
        param.requires_grad = False

    return generator, discriminator, vgg16

def get_optimizers(generator: nn.Module, discriminator: nn.Module) -> tuple[torch.optim.Optimizer, torch.optim.Optimizer]:
    optiGen = torch.optim.Adam(params=generator.parameters(), lr=settings.GENERATOR_LR)
    optiDisc = torch.optim.Adam(params=discriminator.parameters(), lr=settings.DISCRIMINATOR_LR)

    return optiGen, optiDisc

def kaiming_initialization(generator: nn.Module) -> None:
    for m in generator.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')

            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

def xaivier_initialization(discriminator: nn.Module) -> None:
    for m in discriminator.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight)

def decrease_lr(optimizer: torch.optim.Optimizer) -> None:
    for group in optimizer.param_groups:
        group['lr'] /= 10

def discriminator_step(bw: torch.Tensor, color: torch.Tensor, generator: nn.Module, discriminator: nn.Module, optimizer: torch.optim.Optimizer) -> float:
    # we will only train the discriminator
    adverserialLoss = nn.BCEWithLogitsLoss()

    for p in discriminator.parameters():
        p.requires_grad = True

    for p in generator.parameters():
        p.requires_grad = False

    discriminator.zero_grad()

    bw = bw.to(settings.DEVICE)
    color = color.to(settings.DEVICE)

    with torch.no_grad():
        generatedColor = generator(bw)
        generatedColor.detach()

    discOutputForGeneratedColor = discriminator(generatedColor)
    discOutputForRealColor = discriminator(color)

    adverserialLossForGeneratedColor = adverserialLoss(discOutputForGeneratedColor, torch.zeros_like(discOutputForGeneratedColor))
    adverserialLossForRealColor = adverserialLoss(discOutputForRealColor, torch.ones_like(discOutputForRealColor))

    discriminatorLoss = adverserialLossForRealColor + adverserialLossForGeneratedColor
    discriminatorLoss.backward()
    optimizer.step()

    return discriminatorLoss.item()
    
def generator_step(bw: torch.Tensor, color: torch.Tensor, generator: nn.Module, discriminator: nn.Module, vgg16: nn.Module, optimizer: torch.optim.Optimizer) -> float:
    # we will only train the generator
    for p in discriminator.parameters():
        p.requires_grad = False
    
    for p in generator.parameters():
        p.requires_grad = True

    generator.zero_grad()

    bw = bw.to(settings.DEVICE)
    color = color.to(settings.DEVICE)

    generatedColor = generator(bw)
    discOutputForGeneratedColor = discriminator(generatedColor)

    percepOutputForGeneratedColor = vgg16(generatedColor)
    with torch.no_grad():
        percepOutputForRealColor = vgg16(color)


    l1Loss = nn.L1Loss()
    mseLoss = nn.MSELoss()
    adverserialLosss = nn.BCEWithLogitsLoss()

    l1LossBetweenGeneratedColorAndRealColor = l1Loss(generatedColor, color)
    percepLossBetweenGeneratedColorAndRealColor = mseLoss(percepOutputForGeneratedColor, percepOutputForRealColor)
    advLossForDiscOutForGeneratedColor = adverserialLosss(discOutputForGeneratedColor, torch.ones_like(discOutputForGeneratedColor)) #fooling disc

    generatorLoss = 10 * l1LossBetweenGeneratedColorAndRealColor + percepLossBetweenGeneratedColorAndRealColor + advLossForDiscOutForGeneratedColor
    
    generatorLoss.backward()
    optimizer.step()

    return generatorLoss.item()

def train(generator: nn.Module, discriminator: nn.Module, vgg16: nn.Module, loader: DataLoader, optimizerGen: torch.optim.Optimizer, optimizerDisc: torch.optim.Optimizer) -> tuple[list[float], list[float]]:

    # specify this is training
    generator.train()
    discriminator.train()

    #Initially train the discriminator
    isDiscTurn = True

    generatorLoss = []
    discriminatorLoss = []

    for epoch in range(settings.NUM_EPOCHS):

        if(epoch == settings.DECAY_EPOCH):
            #now decrease the learning rate
            decrease_lr(optimizer=optimizerGen)
            decrease_lr(optimizer=optimizerDisc)

        
        totalDiscLoss = 0.0
        totalGenLoss = 0.0

        n = 0
        # loop over all the dataset
        for i, (bw, color) in enumerate(tqdm(loader, leave=True, desc=f"Epoch_no: {epoch}")):
            n = i

            if isDiscTurn:
                stepLoss = discriminator_step(bw=bw, color=color, generator=generator, discriminator=discriminator, optimizer=optimizerDisc)
                totalDiscLoss += stepLoss
            else:
                stepLoss = generator_step(bw=bw, color=color, generator=generator, discriminator=discriminator, vgg16=vgg16, optimizer=optimizerGen)
                totalGenLoss += stepLoss

            isDiscTurn = not isDiscTurn
        
        epochGenLoss = totalGenLoss / ( n // 2 + 1)
        epochDiscLoss = totalDiscLoss / ( n // 2 + 1)

        generatorLoss.append(epochGenLoss)
        discriminatorLoss.append(epochDiscLoss)

    return generatorLoss, discriminatorLoss

def main() -> None:
    erros = None
    gen, disc, vgg16 = get_models()

    #send the models to device
    gen = gen.to(settings.DEVICE)
    disc = disc.to(settings.DEVICE)
    vgg16 = vgg16.to(settings.DEVICE)

    #dataset preparation
    dataset = BWColorMangaDataset(bw_manga_path=settings.TRAIN_BW_MANGA_PATH, color_manga_path=settings.TRAIN_COLOR_MANGA_PATH)

    loader = DataLoader(dataset=dataset, batch_size=settings.BATCH_SIZE, shuffle=True, num_workers=2)

    # Weights Initialization
    # kaiming_initialization(generator=gen)
    # xaivier_initialization(discriminator=disc)

    # Optimizers
    OptimGen, OptimDisc = get_optimizers(generator=gen, discriminator=disc)

    # Train
    try:
        genLoss, discLoss = train(generator=gen, discriminator=disc, vgg16=vgg16, loader=loader, optimizerDisc=OptimDisc, optimizerGen=OptimGen)

        save_plots(genLoss, "Generator", discLoss, "Discriminator", "Loss During Training")

    except Exception as e:
        erros = e

    # Save the Generator and Discriminator
    if settings.SAVE_CHECKPOINTS:
        save_model(model=gen, optimizer=OptimGen, checkpoint_type=CheckpointTypes.COLOR_GENERATOR)
        save_model(model=disc, optimizer=OptimDisc, checkpoint_type=CheckpointTypes.COLOR_DISC)

    if erros is not None:
        raise erros

if __name__ == "__main__":
    main()



