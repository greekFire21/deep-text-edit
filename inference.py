import torch
from torchvision import transforms
from torchvision.utils import save_image
import cv2

from src.models.stylegan import StyleBased_Generator
from src.models.embedders import ContentResnet, StyleResnet
from src.utils.draw import draw_word


weights_folder = f"/content/drive/MyDrive/textStyleBrush/deep-text-edit/checkpoints/stylegan(pretrained_on_content)_typeface_ocr_adv_192x64/7"
input_image_path = "/content/drive/MyDrive/textStyleBrush/sample_input.jpg"
output_image_path = "/content/drive/MyDrive/textStyleBrush/sample_output.jpg"
input_text, output_text = "hahaha", "World"
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

model_G = StyleBased_Generator(dim_latent=512)
model_G.load_state_dict(torch.load(f'{weights_folder}/model_G'))
model_G.to(device)

style_embedder = StyleResnet()
style_embedder.load_state_dict(torch.load(f'{weights_folder}/style_embedder'))
style_embedder.to(device)

content_embedder = ContentResnet()
content_embedder.load_state_dict(torch.load(f'{weights_folder}/content_embedder'))
content_embedder.to(device)

model_G.eval()
style_embedder.eval()
content_embedder.eval()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((64, 192)),
])

img_style = cv2.imread(input_image_path, cv2.IMREAD_COLOR)
img_style = transform(img_style).unsqueeze(0).to(device)
print(img_style.shape)

img_content = draw_word(input_text)
img_content = transform(img_content).unsqueeze(0).to(device)
print(img_content.shape)

style_embeds = style_embedder(img_style)
content_embeds = content_embedder(img_content)

preds = model_G(content_embeds, style_embeds)

save_image(preds, output_image_path)
