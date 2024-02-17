import torch
import io
from utils import *
from PIL import Image, ImageDraw, ImageFont

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model checkpoints
srresnet_checkpoint = "./resnet/checkpoint_srresnet2x_735.pth.tar"

# Load models
srresnet = torch.load(srresnet_checkpoint)['model'].to(device)
srresnet.eval()

def visualize_sr(img):
    

    lr_img = img
    #change image format
    #img_bytes = io.BytesIO()
    #hr_img.save(img_bytes, "png")
    #hr_img = Image.open(img_bytes, mode="r")
    
    
    lr_img = lr_img.convert('RGB')


    # Super-resolution (SR) with SRResNet
    sr_img_srresnet = srresnet(convert_image(lr_img, source='pil', target='imagenet-norm').unsqueeze(0).to(device))
    sr_img_srresnet = sr_img_srresnet.squeeze(0).cpu().detach()
    sr_img_srresnet = convert_image(sr_img_srresnet, source='[-1, 1]', target='pil')

    return sr_img_srresnet
