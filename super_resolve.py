import torch
import io
from utils import *
from PIL import Image, ImageDraw, ImageFont
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model checkpoints
srresnet_checkpoint = "./resnet/checkpoint_srresnet2x_1050.pth.tar"

# Load models
srresnet = torch.load(srresnet_checkpoint)['model'].to(device)
srresnet.eval()
for param in srresnet.parameters():
    param.grad = None

def visualize_sr(img):
    
    #parameters for division on big images
    HEIGHT = 400
    WIDTH = 400
    MAX_SIZE = WIDTH * HEIGHT # (WIDTH, HEIGHT)
    
    image_frags = []

    lr_img = img
    lr_img = lr_img.convert('RGB')

    if lr_img.size[0] * lr_img.size[1] > MAX_SIZE:
        imgwidth, imgheight = lr_img.size
        for i in range(0,imgheight,HEIGHT):
            for j in range(0,imgwidth,WIDTH):
                box = (j, i, j+WIDTH, i+HEIGHT)
                a = lr_img.crop(box)
                image_frags.append(a)

    # Super-resolution (SR) with SRResNet
                
    if len(image_frags) > 0:
        #Image pating calculations
        new_sr_img = Image.new("RGB", (lr_img.size[0]*2, lr_img.size[1]*2), "white")
        width_capacity = math.ceil(lr_img.size[0]/WIDTH)
        print("image_frags:", len(image_frags))
        for i in range(len(image_frags)):
            print(f"Image: {i}")
            sr_img_srresnet = srresnet(convert_image(image_frags[i], source='pil', target='imagenet-norm').unsqueeze(0).to(device))
            sr_img_srresnet = sr_img_srresnet.squeeze(0).cpu().detach()
            sr_img_srresnet = convert_image(sr_img_srresnet, source='[-1, 1]', target='pil')
            new_sr_img.paste(sr_img_srresnet, ((i%width_capacity)*WIDTH*2, i//width_capacity*HEIGHT*2))
            del sr_img_srresnet

        return new_sr_img
    else:
        sr_img_srresnet = srresnet(convert_image(lr_img, source='pil', target='imagenet-norm').unsqueeze(0).to(device))
        sr_img_srresnet = sr_img_srresnet.squeeze(0).cpu().detach()
        sr_img_srresnet = convert_image(sr_img_srresnet, source='[-1, 1]', target='pil')

        return sr_img_srresnet
