from themodel.generator import UNet
from themodel.discriminator import PatchGAN

from themodel.config import settings
from themodel.dataset import BWColorMangaDataset
from themodel.losses import VGGPerceptualLoss, white_color_penalty
from themodel.utils import save_model, load_model, CheckpointTypes, make_deterministic, manage_loss

import torch
import torch.nn as nn
import torch.optim as optimizer

from tqdm import tqdm
from torch.utils.data import DataLoader


plot_ad_bw_disc = []
plot_ad_co_disc = []

plot_l1_co_gen = []
plot_l1_bw_gen = []

plot_per_co_gen = []
plot_per_bw_gen = []

plot_wh_co_gen = []

plot_cycle_co_gen = []
plot_cycle_bw_gen = []


# fmt: off
def train_model(
        bw_disc, co_disc, bw_gen, co_gen,
        optimizer_disc, optimizer_gen,
        l1, perceptual_loss, adverserial_loss, white_color_penalty_loss,
        train_loader,
        gen_scaler, disc_scaler,
        epoch_no
    ):
    #fmt: on
    
    loop = tqdm(train_loader, leave=True)


    for _, (bw, color) in enumerate(loop):
        bw = bw.to(settings.DEVICE)
        color = color.to(settings.DEVICE)



        #for discriminator we only we adverserial_loss
        with torch.cuda.amp.autocast(): #type:ignore
            generated_color = co_gen(bw)
            co_disc_res_for_color = co_disc(color)
            co_disc_res_for_generated = co_disc(generated_color.detach())

            generated_bw = bw_gen(color)
            bw_disc_res_for_bw = bw_disc(bw)
            bw_disc_res_for_generated = bw_disc(generated_bw.detach())

            color_disc_loss_color = adverserial_loss(co_disc_res_for_color, torch.ones_like(co_disc_res_for_color))
            color_disc_loss_generated = adverserial_loss(co_disc_res_for_generated, torch.zeros_like(co_disc_res_for_generated))

            bw_disc_loss_bw = adverserial_loss(bw_disc_res_for_bw, torch.ones_like(bw_disc_res_for_bw))
            bw_disc_loss_generatd = adverserial_loss(bw_disc_res_for_generated, torch.zeros_like(bw_disc_res_for_generated))

            color_disc_loss_total = color_disc_loss_color + color_disc_loss_generated
            bw_disc_loss_total = bw_disc_loss_bw + bw_disc_loss_generatd

            disc_loss = (color_disc_loss_total + bw_disc_loss_total) / 2

            plot_ad_co_disc.append(color_disc_loss_total)
            plot_ad_bw_disc.append(bw_disc_loss_total)


        optimizer_disc.zero_grad()
        disc_scaler.scale(disc_loss).backward()
        disc_scaler.step(optimizer_disc)
        disc_scaler.update()

        with torch.cuda.amp.autocast(): #type:ignore

            #adverserial loss for generators
            bw_disc_res_for_generated = bw_disc(generated_bw)
            color_disc_res_for_generated = co_disc(generated_color)
            bw_disc_loss_for_generated = adverserial_loss(bw_disc_res_for_generated, torch.ones_like(bw_disc_res_for_generated))
            color_disc_loss_for_generated = adverserial_loss(color_disc_res_for_generated, torch.ones_like(color_disc_res_for_generated))

            #l1 loss
            l1_bw_out = bw_gen(color)
            l1_color_out = co_gen(bw)
            l1_loss_for_bw = l1(l1_bw_out, bw)
            l1_loss_for_color = l1(l1_color_out, color)

            plot_l1_bw_gen.append(l1_loss_for_bw)
            plot_l1_co_gen.append(l1_loss_for_color)


            #perceptual loss
            per_bw_out = bw_gen(color)
            per_color_out = co_gen(bw)
            perceptual_loss_for_bw = perceptual_loss(per_bw_out, bw)
            perceptual_loss_for_color = perceptual_loss(per_color_out, color)

            plot_per_bw_gen.append(perceptual_loss_for_bw)
            plot_per_co_gen.append(perceptual_loss_for_color)

            #white color penalty loss
            white_color_out = co_gen(bw)
            white_penalty_loss_for_color = white_color_penalty_loss(color, white_color_out)

            plot_wh_co_gen.append(white_penalty_loss_for_color)

            #cycle consistency loss
            cycle_bw = bw_gen(generated_color)
            cycle_color = co_gen(generated_bw)
            cycle_bw_loss = l1(bw, cycle_bw)
            cycle_color_loss = l1(color, cycle_color)

            plot_cycle_bw_gen.append(cycle_bw_loss)
            plot_cycle_co_gen.append(cycle_color_loss)

            generator_loss = (
                bw_disc_loss_for_generated + color_disc_loss_for_generated
                + l1_loss_for_bw + l1_loss_for_color
                + perceptual_loss_for_bw + perceptual_loss_for_color
                + white_penalty_loss_for_color
                + cycle_bw_loss * settings.LAMBDA_CYCLE + cycle_color_loss * settings.LAMBDA_CYCLE
            )

        
        optimizer_gen.zero_grad()
        gen_scaler.scale(generator_loss).backward()
        gen_scaler.step(optimizer_gen)
        gen_scaler.update()

    # manage the losses
    manage_loss(plot_ad_bw_disc, epoch_no=epoch_no)
    manage_loss(plot_ad_co_disc, epoch_no=epoch_no)
    manage_loss(plot_l1_bw_gen, epoch_no=epoch_no)
    manage_loss(plot_l1_co_gen, epoch_no=epoch_no)
    manage_loss(plot_per_bw_gen, epoch_no=epoch_no)
    manage_loss(plot_per_co_gen, epoch_no=epoch_no)
    manage_loss(plot_wh_co_gen, epoch_no=epoch_no)
    manage_loss(plot_cycle_bw_gen, epoch_no=epoch_no)
    manage_loss(plot_cycle_co_gen, epoch_no=epoch_no)


    save_model(model=bw_disc, optimizer=optimizer_disc, checkpoint_type=CheckpointTypes.BW_DISC)
    save_model(model=co_disc, optimizer=optimizer_disc, checkpoint_type=CheckpointTypes.COLOR_DISC)
    save_model(model=co_gen, optimizer=optimizer_gen, checkpoint_type=CheckpointTypes.COLOR_GENERATOR)
    save_model(model=bw_gen, optimizer=optimizer_gen, checkpoint_type=CheckpointTypes.BW_GENERATOR)


        

            


def main():
    # make_deterministic()

    bw_disc = PatchGAN().to(settings.DEVICE)
    co_disc = PatchGAN().to(settings.DEVICE)

    bw_gen = UNet().to(settings.DEVICE)
    co_gen = UNet().to(settings.DEVICE)

    optimizer_disc = optimizer.Adam(
        params=list(bw_disc.parameters()) + list(co_disc.parameters()),
        lr=settings.LEARNING_RATE,
        betas=(0.5, 0.999),
    )

    optimizer_gen = optimizer.Adam(
        params=list(bw_gen.parameters()) + list(co_gen.parameters()),
        lr=settings.LEARNING_RATE,
        betas=(0.5, 0.999),
    )

    l1 = nn.L1Loss()
    perceptual_loss = VGGPerceptualLoss().to(settings.DEVICE)
    adverserial_loss = nn.BCEWithLogitsLoss()
    white_color_penalty_loss = white_color_penalty

    if settings.LOAD_CHECKPOINTS:
        load_model(
            checkpoint_type=CheckpointTypes.COLOR_GENERATOR,
            model=co_gen,
            optimizer=optimizer_gen,
            lr=settings.LEARNING_RATE,
        )

        load_model(
            checkpoint_type=CheckpointTypes.BW_GENERATOR,
            model=bw_gen,
            optimizer=optimizer_gen,
            lr=settings.LEARNING_RATE,
        )

        load_model(
            checkpoint_type=CheckpointTypes.COLOR_DISC,
            model=co_disc,
            optimizer=optimizer_disc,
            lr=settings.LEARNING_RATE,
        )

        load_model(
            checkpoint_type=CheckpointTypes.BW_DISC,
            model=bw_disc,
            optimizer=optimizer_disc,
            lr=settings.LEARNING_RATE,
        )

    train_dataset = BWColorMangaDataset(
        bw_manga_path=settings.TRAIN_BW_MANGA_PATH,
        color_manga_path=settings.TRAIN_COLOR_MANGA_PATH,
    )
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=settings.BATCH_SIZE,
        shuffle=True,
        num_workers=settings.NUM_WORKERS,
        pin_memory=True,
    )

    gen_scaler = torch.cuda.amp.GradScaler()  # type:ignore
    disc_scaler = torch.cuda.amp.GradScaler()  # type:ignore

    # fmt: off
    for epoch in range(settings.NUM_EPOCHS):
        train_model(
            bw_disc, co_disc, bw_gen, co_gen,
            optimizer_disc, optimizer_gen,
            l1, perceptual_loss, adverserial_loss, white_color_penalty_loss,
            train_loader,
            gen_scaler, disc_scaler,
            epoch
        )
    # fmt:on
