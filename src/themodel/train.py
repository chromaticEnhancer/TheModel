from themodel.generator import UNet
from themodel.discriminator import PatchGAN

from themodel.config import settings
from themodel.dataset import BWColorMangaDataset
from themodel.losses import VGGPerceptualLoss, white_color_penalty
from themodel.utils import (
    save_model,
    load_model,
    CheckpointTypes,
    make_deterministic,
    save_plots,
)

import torch
import torch.nn as nn
import torch.optim as optimizer

from tqdm import tqdm
from torch.utils.data import DataLoader


plot_ad_bw_disc = []
plot_ad_co_disc = []

plot_l1_co_gen = []
plot_l1_bw_gen = []

# plot_per_co_gen = []
# plot_per_bw_gen = []

# plot_wh_co_gen = []
plot_ad_co_gen = []
plot_ad_bw_gen = []

plot_cycle_co_gen = []
plot_cycle_bw_gen = []


# fmt: off
def train_model(
        bw_disc, co_disc, bw_gen, co_gen,
        optimizer_disc, optimizer_gen,
        l1, adverserial_loss, #white_color_penalty_loss,
        train_loader,
        epoch_no
    ):
    #fmt: on
    

    #avgerage loss
    avg_loss_ad_co_disc = []
    avg_loss_ad_bw_disc = []

    avg_loss_l1_co_gen = []
    avg_loss_l1_bw_gen = []

    # avg_loss_wh_pen_co_gen = []
    avg_loss_ad_co_gen = []
    avg_loss_ad_bw_gen = []

    avg_loss_cy_co_gen = []
    avg_loss_cy_bw_gen = []


    for bw, color in tqdm(train_loader, leave=True, desc=f'Epoch_no: {epoch_no}'):
        bw = bw.to(settings.DEVICE)
        color = color.to(settings.DEVICE)


        # Discriminator

        generated_color = co_gen(bw)
        co_disc_res_for_color = co_disc(color)
        co_disc_res_for_generated = co_disc(generated_color.detach())


        generated_bw = bw_gen(color)
        bw_disc_res_for_bw = bw_disc(bw)
        bw_disc_res_for_generated = bw_disc(generated_bw.detach())


        #for discriminator we only we adverserial_loss
        color_disc_loss_color = adverserial_loss(co_disc_res_for_color, torch.ones_like(co_disc_res_for_color))
        color_disc_loss_generated = adverserial_loss(co_disc_res_for_generated, torch.zeros_like(co_disc_res_for_generated))

        bw_disc_loss_bw = adverserial_loss(bw_disc_res_for_bw, torch.ones_like(bw_disc_res_for_bw))
        bw_disc_loss_generatd = adverserial_loss(bw_disc_res_for_generated, torch.zeros_like(bw_disc_res_for_generated))

        color_disc_loss_total = color_disc_loss_color + color_disc_loss_generated
        bw_disc_loss_total = bw_disc_loss_bw + bw_disc_loss_generatd

        disc_loss = (color_disc_loss_total + bw_disc_loss_total) / 2

        avg_loss_ad_co_disc.append(color_disc_loss_total.item())
        avg_loss_ad_bw_disc.append(bw_disc_loss_total.item())


        optimizer_disc.zero_grad()
        disc_loss.backward()
        optimizer_disc.step()
    
        #Generator

        generated_color_g = co_gen(bw)
        generated_bw_g = bw_gen(color)

        #adverserial loss for generators
        bw_disc_res_for_generated = bw_disc(generated_bw_g)
        color_disc_res_for_generated = co_disc(generated_color_g)
        bw_disc_loss_for_generated = adverserial_loss(bw_disc_res_for_generated, torch.ones_like(bw_disc_res_for_generated))
        color_disc_loss_for_generated = adverserial_loss(color_disc_res_for_generated, torch.ones_like(color_disc_res_for_generated))
        avg_loss_ad_co_gen.append(color_disc_loss_for_generated.item())
        avg_loss_ad_bw_gen.append(bw_disc_loss_for_generated.item())

        #l1 loss
        l1_loss_for_bw = l1(generated_bw_g, bw)
        l1_loss_for_color = l1(generated_color_g, color)

        avg_loss_l1_bw_gen.append(l1_loss_for_bw.item())
        avg_loss_l1_co_gen.append(l1_loss_for_color.item())


        #perceptual loss
        # per_bw_out = bw_gen(color)
        # per_color_out = co_gen(bw)
        # perceptual_loss_for_bw = perceptual_loss(generated_bw_g, bw)
        # perceptual_loss_for_color = perceptual_loss(generated_color_g, color)

        # plot_per_bw_gen.append(perceptual_loss_for_bw.item())
        # plot_per_co_gen.append(perceptual_loss_for_color.item())

        #white color penalty loss
        # white_color_out = co_gen(bw)
        # white_penalty_loss_for_color = white_color_penalty_loss(color, generated_color_g)
        # avg_loss_wh_pen_co_gen.append(white_penalty_loss_for_color.item())


        #cycle consistency loss
        cycle_bw = bw_gen(generated_color_g)
        cycle_color = co_gen(generated_bw_g)
        cycle_bw_loss = l1(bw, cycle_bw)
        cycle_color_loss = l1(color, cycle_color)

        avg_loss_cy_bw_gen.append(cycle_bw_loss.item())
        avg_loss_cy_co_gen.append(cycle_color_loss.item())

        generator_loss = (
            bw_disc_loss_for_generated + color_disc_loss_for_generated
            + l1_loss_for_bw + l1_loss_for_color
            # + perceptual_loss_for_bw + perceptual_loss_for_color
            # + white_penalty_loss_for_color
            + cycle_bw_loss * settings.LAMBDA_CYCLE + cycle_color_loss * settings.LAMBDA_CYCLE
        )

        
        optimizer_gen.zero_grad()
        generator_loss.backward()
        optimizer_gen.step()

    # manage the losses
    plot_ad_bw_disc.append(sum(avg_loss_ad_bw_disc) / len(avg_loss_ad_bw_disc))
    plot_ad_co_disc.append(sum(avg_loss_ad_co_disc) / len(avg_loss_ad_co_disc))
    plot_l1_bw_gen.append(sum(avg_loss_l1_bw_gen) / len(avg_loss_l1_bw_gen))
    plot_l1_co_gen.append(sum(avg_loss_l1_co_gen) / len(avg_loss_l1_co_gen))
    # plot_wh_co_gen.append(sum(avg_loss_wh_pen_co_gen) / len(avg_loss_wh_pen_co_gen))
    plot_ad_co_gen.append(sum(avg_loss_ad_co_gen) / len(avg_loss_ad_co_gen))
    plot_ad_bw_gen.append(sum(avg_loss_ad_bw_gen) / len(avg_loss_ad_bw_gen))
    plot_cycle_bw_gen.append(sum(avg_loss_cy_bw_gen) / len(avg_loss_cy_co_gen))
    plot_cycle_co_gen.append(sum(avg_loss_cy_co_gen) / len(avg_loss_cy_co_gen))


    if settings.SAVE_CHECKPOINTS:
        save_model(model=bw_disc, optimizer=optimizer_disc, checkpoint_type=CheckpointTypes.BW_DISC)
        save_model(model=co_disc, optimizer=optimizer_disc, checkpoint_type=CheckpointTypes.COLOR_DISC)
        save_model(model=co_gen, optimizer=optimizer_gen, checkpoint_type=CheckpointTypes.COLOR_GENERATOR)
        save_model(model=bw_gen, optimizer=optimizer_gen, checkpoint_type=CheckpointTypes.BW_GENERATOR)
            


def main():
    # make_deterministic()
    all_exceptions = None

    bw_disc = PatchGAN(in_channels=1).to(settings.DEVICE)
    co_disc = PatchGAN(in_channels=3).to(settings.DEVICE)

    bw_gen = UNet(in_channels=3, out_channels=1).to(settings.DEVICE)
    co_gen = UNet(in_channels=1, out_channels=3).to(settings.DEVICE)


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
    # perceptual_loss = VGGPerceptualLoss().to(settings.DEVICE)
    adverserial_loss = nn.BCEWithLogitsLoss()
    # white_color_penalty_loss = white_color_penalty

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

    try:
        # fmt: off
        for epoch in range(settings.NUM_EPOCHS):
            train_model(
                bw_disc, co_disc, bw_gen, co_gen,
                optimizer_disc, optimizer_gen,
                l1, adverserial_loss, #white_color_penalty_loss,
                train_loader,
                epoch
            )
    except Exception as e:
        all_exceptions = e
    
    # fmt:on
        
    save_model(co_gen, optimizer_gen, CheckpointTypes.COLOR_GENERATOR)

    save_plots(plot_ad_bw_disc, 'BW Discriminator', plot_ad_co_disc, 'Color Discriminator', 'Adverserial Loss Generator')
    save_plots(plot_l1_bw_gen, 'BW Generator', plot_l1_co_gen, 'Color Generator', 'L1 Loss')
    # save_plots(plot_per_bw_gen, 'BW Generator', plot_per_co_gen, 'Color Generator', 'Perceptual Loss')
    # save_plots(plot_wh_co_gen, 'Color Generator', None, None, 'White Color Penalty Loss')
    save_plots(plot_ad_bw_gen, 'BW Generator', plot_ad_co_gen, 'Color Generator', 'Adverserial Loss Generator')
    save_plots(plot_cycle_bw_gen, 'BW Generator', plot_cycle_co_gen, 'Color Generator', 'Cycle Consistency Loss')


    if all_exceptions is not None:
        raise all_exceptions



if __name__ == "__main__":
    main()
