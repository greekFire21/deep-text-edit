import torch
from loguru import logger as info_logger
from src.disk import disk
from pathlib import Path
from src.logger.simple import Logger
from src.losses.vgg import VGGLoss
from src.data.baseline import BaselineDataset
from src.losses.vgg import VGGLoss
from src.utils.download import download_dataset
from src.models.embedders import ContentResnet, StyleResnet
from src.models.nlayer_discriminator import NLayerDiscriminator
from src.models.stylegan_mask import StyleBased_Generator
from src.training.stylegan_mask import StyleGanMaskTrainer
from src.storage.simple import Storage
from src.losses.STRFL import OCRLoss
from src.losses.typeface_perceptual import TypefacePerceptualLoss

from datasets import load_dataset
from src.data.dataloader import HFDataLoaderForStyleGANMask


class Config:
    def __init__(self):
        disk.login()

        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        info_logger.info(f'Using device: {device}')
        style_dir = Path('data/IMGUR5K')
        
        batch_size = 64
        dataset = load_dataset("jhc90/webtoon_text_conversion_data_v2", cache_dir="/content/drive/MyDrive/textStyleBrush/webtoon_text_conversion_data_v2")

        train_dataloader = HFDataLoaderForStyleGANMask(dataset['train'], shuffle=True, batch_size=batch_size, num_workers=8).make_dataloader()
        val_dataloader = HFDataLoaderForStyleGANMask(dataset['valid'], shuffle=False, batch_size=batch_size, num_workers=8).make_dataloader()

        total_epochs = 500

        weights_folder_name = 'Stylegan (pretrained on content)'
        weights_folder = f'models/{weights_folder_name}'

        model_G = StyleBased_Generator(dim_latent=512)
        model_G.to(device)

        style_embedder = StyleResnet().to(device) 
        style_embedder.load_state_dict(torch.load(f'{weights_folder}/style_embedder'))

        content_embedder = ContentResnet().to(device)
        content_embedder.load_state_dict(torch.load(f'{weights_folder}/content_embedder'))

        model_D = NLayerDiscriminator(input_nc=3, ndf=64, n_layers=3, norm_layer=(lambda x : torch.nn.Identity()))
        model_D.to(device)

        optimizer_G = torch.optim.AdamW(
            list(model_G.parameters()) +
            list(style_embedder.parameters()) +
            list(content_embedder.parameters()),
            lr=1e-3,
            weight_decay=1e-6
        )
        scheduler_G = torch.optim.lr_scheduler.ExponentialLR(
            optimizer_G,
            gamma=0.9
        )

        optimizer_D = torch.optim.AdamW(model_D.parameters(), lr=1e-4)
        scheduler_D = torch.optim.lr_scheduler.ExponentialLR(
            optimizer_D,
            gamma=0.9
        )

        dir_coef = 1.0
        perc_coef = 25.0
        tex_coef = 7.0
        adv_coef = 0.06

        checkpoint_folder = 'stylegan(pretrained_on_content)_mse_128x128'
        storage = Storage(f'checkpoints/{checkpoint_folder}')

        logger = Logger(
            image_freq=100,
            project_name='deep-text-edit',
            entity='greekfire21',
            config={
                'img_size': (128, 128)
            }
        )

        self.trainer = StyleGanMaskTrainer(
            model_G,
            model_D,
            style_embedder,
            content_embedder,
            optimizer_G,
            optimizer_D,
            scheduler_G,
            scheduler_D,
            train_dataloader,
            val_dataloader,
            storage,
            logger,
            total_epochs,
            device,
            dir_coef,
            perc_coef,
            tex_coef,
            adv_coef,
            VGGLoss(),
            torch.nn.MSELoss()
        )

    def run(self):
        self.trainer.run()
