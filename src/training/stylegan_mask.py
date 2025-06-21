import torch
from src.logger.simple import Logger
from src.storage.simple import Storage
from src.utils.draw import draw_word, img_to_tensor
from torch import nn, optim
from torch.utils.data import DataLoader
from loguru import logger
from tqdm import tqdm


class StyleGanMaskTrainer:
    def __init__(self,
                 model_G: nn.Module,
                 style_embedder: nn.Module,
                 content_embedder: nn.Module,
                 optimizer_G: optim.Optimizer,
                 scheduler_G: optim.lr_scheduler._LRScheduler,
                 train_dataloader: DataLoader,
                 val_dataloader: DataLoader,
                 storage: Storage,
                 logger: Logger,
                 total_epochs: int,
                 device: str,
                 loss: nn.Module,
                 ):

        self.device = device
        self.model_G = model_G
        self.optimizer_G = optimizer_G
        self.scheduler_G = scheduler_G
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.total_epochs = total_epochs
        self.logger = logger
        self.storage = storage
        self.loss = loss.to(device)
        self.style_embedder = style_embedder
        self.content_embedder = content_embedder
    
    def set_requires_grad(self, net: nn.Module, requires_grad: bool = False):
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad

    def train(self):
        logger.info('Start training')
        self.model_G.train()
        self.content_embedder.train()
        self.style_embedder.train()

        for batch in tqdm(self.train_dataloader):
            style_imgs = batch["style_img"]
            content_imgs = batch["content_img"]
            mask_imgs = batch["mask_img"]
            desired_imgs = batch["desired_img"]
            
            self.optimizer_G.zero_grad()

            style_imgs = style_imgs.to(self.device)
            content_imgs = content_imgs.to(self.device)
            mask_imgs = mask_imgs.to(self.device)
            desired_imgs = desired_imgs.to(self.device)

            style_embeds = self.style_embedder(style_imgs)
            content_embeds = self.content_embedder(content_imgs)

            ### calculate G losses
            self.set_requires_grad(self.model_G, True) 
            
            preds_rgb, preds_mask = self.model_G(content_embeds, style_embeds)
            desired_imgs_loss = self.loss(preds_rgb, desired_imgs)
            mask_imgs_loss = self.loss(preds_mask, mask_imgs)

            loss_G = desired_imgs_loss + mask_imgs_loss

            loss_G.backward()

            self.optimizer_G.step()

            self.logger.log_train(
                losses={
                    'desired_imgs_loss': desired_imgs_loss.item(),
                    'mask_imgs_loss': mask_imgs_loss.item(),
                    'full_loss': loss_G.item()},
                images={
                    'style': style_imgs,
                    'content': content_imgs,
                    'result': preds_rgb,
                    'mask_result': preds_mask})

    def validate(self, epoch: int):
        self.model_G.eval()
        self.content_embedder.eval()
        self.style_embedder.eval()

        for batch in tqdm(self.val_dataloader):
            style_imgs = batch["style_img"]
            content_imgs = batch["content_img"]
            mask_imgs = batch["mask_img"]
            desired_imgs = batch["desired_img"]

            self.optimizer_G.zero_grad()

            style_imgs = style_imgs.to(self.device)
            content_imgs = content_imgs.to(self.device)
            mask_imgs = mask_imgs.to(self.device)
            desired_imgs = desired_imgs.to(self.device)
            style_embeds = self.style_embedder(style_imgs)
            content_embeds = self.content_embedder(content_imgs)

            preds_rgb, preds_mask = self.model_G(content_embeds, style_embeds)
            desired_imgs_loss = self.loss(preds_rgb, desired_imgs)
            mask_imgs_loss = self.loss(preds_mask, mask_imgs)

            loss_G = desired_imgs_loss + mask_imgs_loss

            self.logger.log_val(
                losses={
                    'desired_imgs_loss': desired_imgs_loss.item(),
                    'mask_imgs_loss': mask_imgs_loss.item(),
                    'full_loss': loss_G.item()},
                images={
                    'style': style_imgs,
                    'content': content_imgs,
                    'result': preds_rgb,
                    'mask_result': preds_mask})

        avg_losses, _ = self.logger.end_val()
        self.storage.save(epoch,
                          {'model_G': self.model_G,
                           'content_embedder': self.content_embedder,
                           'style_embedder': self.style_embedder,
                           'optimizer_G': self.optimizer_G,
                           'scheduler_G': self.scheduler_G,},
                          avg_losses['full_loss'])

    def run(self):
        for epoch in range(self.total_epochs):
            logger.info(f'epoch {epoch}')
            self.train()
            with torch.no_grad():
                self.validate(epoch)
            if self.scheduler_G is not None:
                self.scheduler_G.step()
