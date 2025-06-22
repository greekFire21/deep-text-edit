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
                 model_D: nn.Module,
                 style_embedder: nn.Module,
                 content_embedder: nn.Module,
                 optimizer_G: optim.Optimizer,
                 optimizer_D: optim.Optimizer,
                 scheduler_G: optim.lr_scheduler._LRScheduler,
                 scheduler_D: optim.lr_scheduler._LRScheduler,
                 train_dataloader: DataLoader,
                 val_dataloader: DataLoader,
                 storage: Storage,
                 logger: Logger,
                 total_epochs: int,
                 device: str,
                 dir_coef: float,
                 perc_coef: float,
                 tex_coef: float,
                 adv_coef: float,
                 ocr_coef: float,
                 perc_loss: nn.Module,
                 dir_loss: nn.Module,
                 ocr_loss: nn.Module,
                 ):

        self.device = device
        self.model_G = model_G
        self.model_D = model_D
        self.optimizer_G = optimizer_G
        self.optimizer_D = optimizer_D
        self.scheduler_G = scheduler_G
        self.scheduler_D = scheduler_D
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.total_epochs = total_epochs
        self.logger = logger
        self.storage = storage
        self.dir_coef = dir_coef
        self.perc_coef = perc_coef
        self.tex_coef = tex_coef
        self.adv_coef = adv_coef
        self.ocr_coef = ocr_coef
        self.dir_loss = dir_loss.to(device)
        self.perc_loss = perc_loss.to(device)
        self.ocr_loss = ocr_loss.to(device)
        self.style_embedder = style_embedder
        self.content_embedder = content_embedder
    
    def set_requires_grad(self, net: nn.Module, requires_grad: bool = False):
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad

    def model_D_adv_loss(self, 
                         style_imgs: torch.Tensor, 
                         content_embeds: torch.Tensor, 
                         style_embeds: torch.Tensor
                         ):
        pred_rgb, _ = self.model_G(content_embeds, style_embeds)
        pred_D_fake = self.model_D(pred_rgb.detach())
        pred_D_real = self.model_D(style_imgs)
        fake = torch.tensor(0.).expand_as(pred_D_fake).to(self.device)
        real = torch.tensor(1.).expand_as(pred_D_real).to(self.device)
        return (self.dir_loss(pred_D_real, real) + self.dir_loss(pred_D_fake, fake)) / 2.

    def model_G_adv_loss(self, preds: torch.Tensor):
        pred_D_fake = self.model_D(preds)
        valid = torch.tensor(1.).expand_as(pred_D_fake).to(self.device)
        return self.dir_loss(pred_D_fake, valid)
    
    def train(self):
        logger.info('Start training')
        self.model_G.train()
        self.model_D.train()
        self.content_embedder.train()
        self.style_embedder.train()

        for batch in tqdm(self.train_dataloader):
            style_imgs = batch["style_img"]
            content_imgs = batch["content_img"]
            mask_imgs = batch["mask_img"]
            desired_imgs = batch["desired_img"]
            
            self.optimizer_G.zero_grad()
            self.optimizer_D.zero_grad()
            style_imgs = style_imgs.to(self.device)
            content_imgs = content_imgs.to(self.device)
            mask_imgs = mask_imgs.to(self.device)
            desired_imgs = desired_imgs.to(self.device)

            style_embeds = self.style_embedder(style_imgs)
            content_embeds = self.content_embedder(content_imgs)

            ### calculate D loss
            self.set_requires_grad(self.model_D, True)
            self.set_requires_grad(self.model_G, False)
            loss_D = self.model_D_adv_loss(style_imgs, content_embeds, style_embeds)

            ### calculate G losses
            self.set_requires_grad(self.model_D, False)
            self.set_requires_grad(self.model_G, True) 
            
            preds_rgb, preds_mask = self.model_G(content_embeds, style_embeds)
            desired_imgs_dir_loss = self.dir_loss(preds_rgb, desired_imgs)
            mask_imgs_dir_loss = self.dir_loss(preds_mask, mask_imgs)

            perc_loss, tex_loss = self.perc_loss(desired_imgs, preds_rgb)

            adv_loss = self.model_G_adv_loss(preds_rgb)

            ocr_loss, recognized = self.ocr_loss(preds_rgb, batch["content"], return_recognized=True)
            word_images = torch.stack(list(map(lambda word: img_to_tensor(draw_word(word)), recognized)))

            loss_G = \
                self.dir_coef * desired_imgs_dir_loss + \
                self.dir_coef * mask_imgs_dir_loss + \
                self.perc_coef * perc_loss + \
                self.tex_coef * tex_loss + \
                self.adv_coef * adv_loss + \
                self.ocr_coef * ocr_loss

            # update models
            self.set_requires_grad(self.model_D, True)
            
            loss_G.backward()
            loss_D.backward()

            self.optimizer_D.step()
            self.optimizer_G.step()

            self.logger.log_train(
                losses={
                    'desired_imgs_dir_loss': desired_imgs_dir_loss.item(),
                    'mask_imgs_dir_loss': mask_imgs_dir_loss.item(),
                    'perc_loss': perc_loss.item(),
                    'tex_loss': tex_loss.item(),
                    'adv_loss': adv_loss.item(),
                    'disc_loss': loss_D.item(),
                    'ocr_loss': ocr_loss.item(),
                    'full_loss': loss_G.item()},
                images={
                    'style': style_imgs,
                    'content': content_imgs,
                    'result': preds_rgb,
                    'mask_result': preds_mask,
                    'recognized': word_images})

    def validate(self, epoch: int):
        self.model_G.eval()
        self.model_D.eval()
        self.content_embedder.eval()
        self.style_embedder.eval()

        for batch in tqdm(self.val_dataloader):
            style_imgs = batch["style_img"]
            content_imgs = batch["content_img"]
            mask_imgs = batch["mask_img"]
            desired_imgs = batch["desired_img"]

            self.optimizer_G.zero_grad()
            self.optimizer_D.zero_grad()
            style_imgs = style_imgs.to(self.device)
            content_imgs = content_imgs.to(self.device)
            mask_imgs = mask_imgs.to(self.device)
            desired_imgs = desired_imgs.to(self.device)
            style_embeds = self.style_embedder(style_imgs)
            content_embeds = self.content_embedder(content_imgs)

            preds_rgb, preds_mask = self.model_G(content_embeds, style_embeds)
            desired_imgs_dir_loss = self.dir_loss(preds_rgb, desired_imgs)
            mask_imgs_dir_loss = self.dir_loss(preds_mask, mask_imgs)

            perc_loss, tex_loss = self.perc_loss(desired_imgs, preds_rgb)

            adv_loss = self.model_G_adv_loss(preds_rgb)

            ocr_loss, recognized = self.ocr_loss(preds_rgb, batch["content"], return_recognized=True)
            word_images = torch.stack(list(map(lambda word: img_to_tensor(draw_word(word)), recognized)))

            loss_G = \
                self.dir_coef * desired_imgs_dir_loss + \
                self.dir_coef * mask_imgs_dir_loss + \
                self.perc_coef * perc_loss + \
                self.tex_coef * tex_loss + \
                self.adv_coef * adv_loss + \
                self.ocr_coef * ocr_loss

            self.logger.log_val(
                losses={
                    'desired_imgs_dir_loss': desired_imgs_dir_loss.item(),
                    'mask_imgs_dir_loss': mask_imgs_dir_loss.item(),
                    'perc_loss': perc_loss.item(),
                    'tex_loss': tex_loss.item(),
                    'adv_loss': adv_loss.item(),
                    'ocr_loss': ocr_loss.item(),
                    'full_loss': loss_G.item()},
                images={
                    'style': style_imgs,
                    'content': content_imgs,
                    'result': preds_rgb,
                    'mask_result': preds_mask,
                    'recognized': word_images})

        avg_losses, _ = self.logger.end_val()
        self.storage.save(epoch,
                          {'model_G': self.model_G,
                           'model_D': self.model_D,
                           'content_embedder': self.content_embedder,
                           'style_embedder': self.style_embedder,
                           'optimizer_G': self.optimizer_G,
                           'optimizer_D': self.optimizer_D,
                           'scheduler_G': self.scheduler_G,
                           'scheduler_D': self.scheduler_D},
                          avg_losses['full_loss'])

    def run(self):
        for epoch in range(self.total_epochs):
            logger.info(f'epoch {epoch}')
            self.train()
            with torch.no_grad():
                self.validate(epoch)
            if self.scheduler_G is not None:
                self.scheduler_G.step()
            if self.scheduler_D is not None:
                self.scheduler_D.step()
