import torchvision

from themodel.generator import UNet
from themodel.ogenerator import Generator
from themodel.discriminator import PatchGAN

from themodel.config import settings
from themodel.dataset import BWColorMangaDataset
from themodel.losses import white_color_penalty
from themodel.utils import (
    save_model,
    CheckpointTypes,
    save_plots,
)

import torch
import torch.nn as nn

from tqdm import tqdm
from torch.utils.data import DataLoader

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

def get_models() -> tuple[nn.Module, nn.Module, nn.Module, nn.Module, nn.Module]:

    # generatorBW = UNet(in_channels=3, out_channels=3).to(settings.DEVICE)
    # generatorColor = UNet(in_channels=3, out_channels=3).to(settings.DEVICE)
    generatorBW = Generator(img_channels=3).to(settings.DEVICE)
    generatorColor = Generator(img_channels=3).to(settings.DEVICE)

    discriminatorBW = PatchGAN(in_channels=3).to(settings.DEVICE)
    discriminatorColor = PatchGAN(in_channels=3).to(settings.DEVICE)

    vgg16 = torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.DEFAULT).eval().to(settings.DEVICE)
    for param in vgg16.parameters():
        param.requires_grad = False

    return generatorBW, generatorColor, discriminatorBW, discriminatorColor, vgg16

def get_optimizers(generatorBW: nn.Module, generatorColor: nn.Module, discriminatorBW: nn.Module, discriminatorColor: nn.Module) -> tuple[torch.optim.Optimizer, torch.optim.Optimizer]:
    optimizerGen = torch.optim.Adam(params=list(generatorBW.parameters()) + list(generatorColor.parameters()), lr=settings.GENERATOR_LR, betas=(0.5, 0.999))
    optimizerDisc = torch.optim.Adam(params=list(discriminatorBW.parameters()) + list(discriminatorColor.parameters()), lr=settings.DISCRIMINATOR_LR, betas=(0.5, 0.999))

    return optimizerGen, optimizerDisc

def decrease_lr(optimizer: torch.optim.Optimizer) -> None:
    for group in optimizer.param_groups:
        group['lr'] /= 10

def discriminator_step(bw: torch.Tensor, color: torch.Tensor, generatorBW: nn.Module, generatorColor: nn.Module, discriminatorBW: nn.Module, discriminatorColor: nn.Module, optimizer: torch.optim.Optimizer) -> tuple[float, float]:
    # we will only train the discriminator
    adverserialLoss = nn.BCEWithLogitsLoss()

    for p in discriminatorBW.parameters():
        p.requires_grad = True

    for p in discriminatorColor.parameters():
        p.requires_grad = True
    
    for p in generatorBW.parameters():
        p.requires_grad = False

    for p in generatorColor.parameters():
        p.requires_grad = False
    
    discriminatorBW.zero_grad()
    discriminatorColor.zero_grad()

    bw = bw.to(settings.DEVICE)
    color = color.to(settings.DEVICE)

    # Train on real data
    with torch.no_grad():
        generatedColor = generatorColor(bw)
        generatedBW = generatorBW(color)

        generatedColor = generatedColor.detach()
        generatedBW = generatedBW.detach()

    
    discOutputForGeneratedBW = discriminatorBW(generatedBW)
    discOutputForRealBW = discriminatorBW(bw)

    discOutputForGeneratedColor = discriminatorColor(generatedColor)
    discOutputForRealColor = discriminatorColor(color)

    adverserialLossForGeneratedBW = adverserialLoss(discOutputForGeneratedBW, torch.zeros_like(discOutputForGeneratedBW))
    adverserialLossForRealBW = adverserialLoss(discOutputForRealBW, torch.ones_like(discOutputForRealBW))

    adverserialLossForGeneratedColor = adverserialLoss(discOutputForGeneratedColor, torch.zeros_like(discOutputForGeneratedColor))
    adverserialLossForRealColor = adverserialLoss(discOutputForRealColor, torch.ones_like(discOutputForRealColor))

    discriminatorLossBW = adverserialLossForRealBW + adverserialLossForGeneratedBW
    discriminatorLossColor = adverserialLossForRealColor + adverserialLossForGeneratedColor

    discriminatorLoss = (discriminatorLossBW + discriminatorLossColor)
    discriminatorLoss.backward()
    optimizer.step()

    return discriminatorLossBW.item(), discriminatorLossColor.item()

def generator_step(bw: torch.Tensor, color: torch.Tensor, generatorBW: nn.Module, generatorColor: nn.Module, discriminatorBW: nn.Module, discriminatorColor: nn.Module, vgg16: nn.Module, optimizer: torch.optim.Optimizer) -> tuple[float, float]:
    # we will only train the generator

    for p in discriminatorBW.parameters():
        p.requires_grad = False
    
    for p in discriminatorColor.parameters():
        p.requires_grad = False

    for p in generatorBW.parameters():
        p.requires_grad = True
    
    for p in generatorColor.parameters():
        p.requires_grad = True

    generatorBW.zero_grad()
    generatorColor.zero_grad()

    bw = bw.to(settings.DEVICE)
    color = color.to(settings.DEVICE)

    generatedColor = generatorColor(bw)
    generatedBW = generatorBW(color)

    discOutputForGeneratedBW = discriminatorBW(generatedBW)
    discOutputForGeneratedColor = discriminatorColor(generatedColor)

    percepOutputForGeneratedBW = vgg16(generatedBW)
    percepOutputForGeneratedColor = vgg16(generatedColor)

    with torch.no_grad():
        percepOutputForRealBW = vgg16(bw)
        percepOutputForRealColor = vgg16(color)
    
    l1Loss = nn.L1Loss()
    mseLoss = nn.MSELoss()
    adverserialLoss = nn.BCEWithLogitsLoss()

    l1LossBetweenGeneratedBWAndRealBW = l1Loss(generatedBW, bw)
    l1LossBetweenGeneratedColorAndRealColor = l1Loss(generatedColor, color)

    percepLossBetweenGeneratedBWAndRealBW = mseLoss(percepOutputForGeneratedBW, percepOutputForRealBW)
    percepLossBetweenGeneratedColorAndRealColor = mseLoss(percepOutputForGeneratedColor, percepOutputForRealColor)

    adverserialLossForGeneratedBW = adverserialLoss(discOutputForGeneratedBW, torch.ones_like(discOutputForGeneratedBW))
    adverserialLossForGeneratedColor = adverserialLoss(discOutputForGeneratedColor, torch.ones_like(discOutputForGeneratedColor))

    cycleBW = generatorBW(generatedColor)
    cycleColor = generatorColor(generatedBW)
    cycleLossBW = l1Loss(cycleBW, bw)
    cycleLossColor = l1Loss(cycleColor, color)

    whiteColorPenalty = white_color_penalty(color, generatedColor) if settings.USE_WHITE_COLOR_LOSS else 0

    generatorLossBW = adverserialLossForGeneratedBW + l1LossBetweenGeneratedBWAndRealBW + percepLossBetweenGeneratedBWAndRealBW + cycleLossBW * settings.LAMBDA_CYCLE

    generatorLossColor = adverserialLossForGeneratedColor + l1LossBetweenGeneratedColorAndRealColor + whiteColorPenalty + percepLossBetweenGeneratedColorAndRealColor + cycleLossColor * settings.LAMBDA_CYCLE

    generatorLoss = (generatorLossBW + generatorLossColor)
    generatorLoss.backward()
    optimizer.step()

    return generatorLossBW.item(), generatorLossColor.item()

def train(generatorBW: nn.Module, generatorColor: nn.Module, discriminatorBW: nn.Module, discriminatorColor: nn.Module, vgg16: nn.Module, optimizerGen: torch.optim.Optimizer, optimizerDisc: torch.optim.Optimizer, loader: DataLoader) -> tuple[list[float], list[float], list[float], list[float]]:

    #set training mode for models
    generatorBW.train()
    generatorColor.train()

    discriminatorBW.train()
    discriminatorColor.train()

    #Initially train the discriminator
    isDiscTurn = True

    generatorBWLoss = []
    generatorColorLoss = []

    discriminatorBWLoss = []
    discriminatorColorLoss = []

    for epoch in range(settings.NUM_EPOCHS):

        if settings.CHANGE_LR:
            if(epoch == settings.DECAY_EPOCH):
                decrease_lr(optimizerGen)
                decrease_lr(optimizerDisc)

        totalDiscBWLoss = 0.0
        totalDiscColorLoss = 0.0
        totalGenBWLoss = 0.0
        totalGenColorLoss = 0.0

        n = 0

        for i, (bw, color) in enumerate(tqdm(loader, leave=True, desc=f"Epoch: {epoch}")):
            n = i

            if isDiscTurn:
                stepLossBW, stepLossColor = discriminator_step(bw=bw, color=color, generatorBW=generatorBW, generatorColor=generatorColor, discriminatorBW=discriminatorBW, discriminatorColor=discriminatorColor, optimizer=optimizerDisc)
                totalDiscBWLoss += stepLossBW
                totalDiscColorLoss += stepLossColor
            else:
                stepLossBW, stepLossColor = generator_step(bw=bw, color=color, generatorBW=generatorBW, generatorColor=generatorColor, discriminatorBW=discriminatorBW, discriminatorColor=discriminatorColor, vgg16=vgg16, optimizer=optimizerGen)
                totalGenBWLoss += stepLossBW
                totalGenColorLoss += stepLossColor


            isDiscTurn = not isDiscTurn

            epochGenBWLoss = totalGenBWLoss / ( n // 2 + 1)
            epochGenColorLoss = totalGenColorLoss / ( n // 2 + 1)
            epochDiscBWLoss = totalDiscBWLoss / ( n // 2 + 1)
            epochDiscColorLoss = totalDiscColorLoss / ( n // 2 + 1)

            generatorBWLoss.append(epochGenBWLoss)
            generatorColorLoss.append(epochGenColorLoss)
            discriminatorBWLoss.append(epochDiscBWLoss)
            discriminatorColorLoss.append(epochDiscColorLoss)

    return generatorBWLoss, generatorColorLoss, discriminatorBWLoss, discriminatorColorLoss

def main() -> None:
    errors = None
    generatorBW, generatorColor, discriminatorBW, discriminatorColor, vgg16 = get_models()

    #use initialized weights
    if settings.USE_INITIALIZED_WEIGHTS:
        kaiming_initialization(generatorBW)
        kaiming_initialization(generatorColor)
        xaivier_initialization(discriminatorBW)
        xaivier_initialization(discriminatorColor)

    #dataset preparation
    dataset = BWColorMangaDataset(bw_manga_path=settings.TRAIN_BW_MANGA_PATH, color_manga_path=settings.TRAIN_COLOR_MANGA_PATH)
    loader = DataLoader(dataset=dataset, batch_size=settings.BATCH_SIZE, shuffle=True, num_workers=2)

    #Optimizers
    optimizerGen, optimizerDisc = get_optimizers(generatorBW, generatorColor, discriminatorBW, discriminatorColor)

    #Train
    try:
        generatorBWLoss, generatorColorLoss, discriminatorBWLoss, discriminatorColorLoss = train(generatorBW, generatorColor, discriminatorBW, discriminatorColor, vgg16, optimizerGen, optimizerDisc, loader)

        save_plots(generatorBWLoss, "GeneratorBW", generatorColorLoss, "GeneratorColor", "Total Generator Loss")
        save_plots(discriminatorBWLoss, "DiscriminatorBW", discriminatorColorLoss, "DiscriminatorColor", "Total Discriminator Loss")

    except Exception as e:
        errors = e

    if settings.SAVE_CHECKPOINTS:
        save_model(generatorColor, optimizerGen, CheckpointTypes.COLOR_GENERATOR)
        
    if errors is not None:
        raise errors
    

if __name__ == "__main__":
    main()