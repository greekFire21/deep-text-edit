import cv2
import numpy as np
import torch
import json
import random
import string
import time

from loguru import logger
from torch.utils.data import Dataset
from src.utils.draw import draw_word
from torchvision import transforms as T
from pathlib import Path


class BaselineDataset(Dataset):
    def __init__(self, style_dir: Path, return_style_labels: bool = False, part: str = ""):
        '''
            root_dir - directory with 2 subdirectories - root_dir/style, root_dir/content
            Images in root_dir/content(hard-coded): 64 x 256
            Images in root_dir/style: arbitrary - need to be resized to 256x256?
        '''
        self.style_dir = style_dir
        # self.style_files = list(self.style_dir.glob('*.png'))
        with open(self.style_dir.parent / f'style_files_{part}.json', 'r') as json_file:
            self.style_files = json.load(json_file)
        self.style_files = list(map(lambda x: Path(x), self.style_files))
        self.return_style_labels = return_style_labels
        json_path = style_dir.parent / f'words_{part}.json'
        with open(json_path, 'r', encoding='utf-8') as json_file:
            self.words = json.load(json_file)
        logger.info(f'Total Files: {len(self.style_files) }')
        self.transform = T.Compose([
            T.ToTensor(),
            T.Resize((64, 192)),
        ])
        self.augment = T.Compose([
            T.RandomInvert(),
        ])

        self.max_retries = 10
        self.retry_delay = 1  # seconds

    def __len__(self):
        return len(self.style_files)

    def __getitem__(self, index):
        try:
            for attempt in range(self.max_retries):
                img_style = cv2.imread(str(self.style_files[index]), cv2.IMREAD_COLOR)
                if img_style is not None:
                    break  # 성공적으로 읽었으면 루프 탈출
                else:
                    # print(f"[Retry {attempt + 1}] Failed to read image: {self.style_files[index]}")
                    time.sleep(self.retry_delay)
            else:
                # 반복 다 해도 실패하면 예외 발생
                raise Exception
            # img_style = cv2.imread(str(self.style_files[index]), cv2.IMREAD_COLOR)
            # if img_style is None:
            #     raise Exception
            img_style = self.transform(img_style)
            img_style = self.augment(img_style)

            content = random.choice(list(self.words.values()))
            allowed_symbols = '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!\"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
            content = ''.join([i for i in content if i in allowed_symbols])
            while not content:
                content = random.choice(list(self.words.values()))
                content = ''.join([i for i in content if i in allowed_symbols])
            img_content = self.transform(draw_word(content))

            content_style = self.words[self.style_files[index].stem]
            content_style = ''.join([i for i in content_style if i in allowed_symbols])
            if not content_style:
                content_style = 'o'
            img_content_style = self.transform(draw_word(content_style))

            if self.return_style_labels:
                return img_style, img_content, content, img_content_style, content_style
            
            return img_style, img_content, content, img_content_style

        except Exception as e:
            logger.error(f'Exception at {self.style_files[index]}, {e}')
            raise e
