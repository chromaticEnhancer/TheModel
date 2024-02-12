import torchvision
from torch.optim import lr_scheduler

from themodel.generator import ResNet9Generator
from themodel.discriminator import PatchGan

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

def normal_initialization(model: nn.Module) -> None:
    for m in model.modules():
        if isinstance(m, nn.Conv2d or nn.Linear):
            nn.init.normal_(m.weight, mean=0, std=0.2)
            if(hasattr(m, 'bias') and m.bias is not None):
                nn.init.constant_(m.bias.data, 0.0)

def get_models() -> tuple[nn.Module, nn.Module, nn.Module, nn.Module]:

    generatorBW = ResNet9Generator(input_channels=3, output_channels=3).to(settings.DEVICE)
    generatorColor = ResNet9Generator(input_channels=3, output_channels=3).to(settings.DEVICE)
    
    discriminatorBW = PatchGan(input_channels=3).to(settings.DEVICE)
    discriminatorColor = PatchGan(input_channels=3).to(settings.DEVICE)

    return generatorBW, generatorColor, discriminatorBW, discriminatorColor


def get_optimizers(generatorBW: nn.Module, generatorColor: nn.Module, discriminatorBW: nn.Module, discriminatorColor: nn.Module) -> tuple[torch.optim.Optimizer, torch.optim.Optimizer]:
    optimizerGen = torch.optim.Adam(params=list(generatorBW.parameters()) + list(generatorColor.parameters()), lr=settings.GENERATOR_LR, betas=(0.5, 0.999))
    optimizerDisc = torch.optim.Adam(params=list(discriminatorBW.parameters()) + list(discriminatorColor.parameters()), lr=settings.DISCRIMINATOR_LR, betas=(0.5, 0.999))

    lr_scheduler.LambdaLR(optimizerGen, lr_lambda=lambda epoch: (1.0 - max(0, epoch + 2 - 100) / float(settings.DECAY_EPOCH + 1)))
    lr_scheduler.LambdaLR(optimizerDisc, lr_lambda=lambda epoch: (1.0 - max(0, epoch + 2 - 100) / float(settings.DECAY_EPOCH + 1)))

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

def shouldTrainModels(models: list[nn.Module], shouldTrain: bool) -> None:
    for model in models:
        for param in model.parameters():
            param.requires_grad = shouldTrain

def train(generatorBW: nn.Module, generatorColor: nn.Module, discriminatorBW: nn.Module, discriminatorColor: nn.Module, optimizerGen: torch.optim.Optimizer, optimizerDisc: torch.optim.Optimizer, loader: DataLoader) -> tuple[list[float], list[float], list[float], list[float]]:

    #set training mode for models
    generatorBW.train()
    generatorColor.train()

    discriminatorBW.train()
    discriminatorColor.train()

    for epoch in range(1, settings.NUM_EPOCHS):
  
        for i, (bw, color) in enumerate(tqdm(loader, leave=True, desc=f"Epoch: {epoch}")):
            fake_color = generatorColor(bw)
            recre_bw = generatorBW(fake_color)

            fake_bw = generatorBW(color)
            recre_color = generatorColor(fake_bw)

            # Train Generators
            shouldTrainModels([discriminatorBW, discriminatorColor], False)
            optimizerGen.zero_grad()

            ...

def main() -> None:
    errors = None
    generatorBW, generatorColor, discriminatorBW, discriminatorColor = get_models()

    #use initialized weights
    if settings.USE_INITIALIZED_WEIGHTS:
        normal_initialization(generatorBW)
        normal_initialization(generatorColor)
        normal_initialization(discriminatorBW)
        normal_initialization(discriminatorColor)

    #dataset preparation
    dataset = BWColorMangaDataset(bw_manga_path=settings.TRAIN_BW_MANGA_PATH, color_manga_path=settings.TRAIN_COLOR_MANGA_PATH)
    loader = DataLoader(dataset=dataset, batch_size=settings.BATCH_SIZE, shuffle=True, num_workers=2)

    #Optimizers
    optimizerGen, optimizerDisc = get_optimizers(generatorBW, generatorColor, discriminatorBW, discriminatorColor)

    #Train
    try:
    
        ...

    except Exception as e:
        errors = e

    if settings.SAVE_CHECKPOINTS:
        save_model(generatorColor, optimizerGen, CheckpointTypes.COLOR_GENERATOR)
        save_model(generatorBW, optimizerGen, CheckpointTypes.BW_GENERATOR)
        save_model(discriminatorColor, optimizerDisc, CheckpointTypes.COLOR_DISC)
        save_model(discriminatorBW, optimizerDisc, CheckpointTypes.BW_DISC)
        
    if errors is not None:
        raise errors
    
