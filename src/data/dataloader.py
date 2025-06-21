from torch.utils.data import DataLoader
from torchvision import transforms
import random
from src.utils.draw import draw_word


class HFDataLoader():
    def __init__(self, dataset, return_style_labels, shuffle, batch_size, num_workers):
        self.dataset = dataset
        self.return_style_labels = return_style_labels
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.words = list(self.dataset["content_style"])
        self.allowed_symbols = '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!\"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((64, 192)),
        ])
        self.augment = transforms.Compose([
            transforms.RandomInvert(),
        ])

        self.dataset.set_transform(self.transform_batch)

    def transform_batch(self, batch):
        batch["img_style"] = [self.transform(img_style) for img_style in batch["img_style"]]
        batch["img_style"] = [self.augment(img_style) for img_style in batch["img_style"]]

        batch["content"] = [self.choose_content() for _ in range(len(batch["img_style"]))]
        batch["img_content"] = [self.transform(draw_word(content)) for content in batch["content"]]

        batch["content_style"] = [self.preprocess_content_style(content_style) for content_style in batch["content_style"]]
        batch["img_content_style"] = [self.transform(draw_word(content_style)) for content_style in batch["content_style"]]

        if not self.return_style_labels:
            batch.pop("content_style")

        return batch
    
    def choose_content(self):
        content = random.choice(self.words)
        content = ''.join([i for i in content if i in self.allowed_symbols])
        while not content:
            content = random.choice(self.words)
            content = ''.join([i for i in content if i in self.allowed_symbols])

        return content
    
    def preprocess_content_style(self, content_style):
        content_style = ''.join([i for i in content_style if i in self.allowed_symbols])
        if not content_style:
            content_style = 'o'

        return content_style
    
    def make_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=self.shuffle, num_workers=self.num_workers)
    

class HFDataLoaderForStyleGANMask():
    def __init__(self, dataset, shuffle, batch_size, num_workers):
        self.dataset = dataset
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((128, 128)),
        ])

        self.dataset.set_transform(self.transform_batch)

    def transform_batch(self, batch):
        batch["style_img"] = [self.transform(style_img) for style_img in batch["style_img"]]
        batch["content_img"] = [self.transform(draw_word(content)) for content in batch["content"]]

        batch["mask_img"] = [self.transform(mask_img) for mask_img in batch["mask_img"]]
        batch["desired_img"] = [self.transform(desired_img) for desired_img in batch["desired_img"]]

        return batch
    
    def make_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=self.shuffle, num_workers=self.num_workers)