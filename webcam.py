import cv2
import torch
import net
import argparse
import torch.nn as nn
from PIL import Image
from function import adaptive_instance_normalization, coral
from torchvision import transforms
import time
import numpy as np

# Device
device = ("cuda" if torch.cuda.is_available() else "cpu")

STYLE_PATH = "input/style/mosaic.jpg"
PRESERVE_COLOR = False
WIDTH = 1280
HEIGHT = 720

# Preprocessing ~ Image to Tensor
def itot(img):
    return torch.Tensor(np.array(img).transpose(2,0,1)).unsqueeze(dim=0).to(device)

# Preprocessing ~ Tensor to Image
def ttoi(tensor):

    # Remove the batch_size dimension
    tensor = tensor.squeeze()
    #img = ttoi_t(tensor)
    img = tensor.cpu().numpy()
    
    # Transpose from [C, H, W] -> [H, W, C]
    img = img.transpose(1, 2, 0)
    return img

def style_transfer(vgg, decoder, content, style, alpha=1.0,
                   interpolation_weights=None):
    
    assert (0.0 <= alpha <= 1.0)
    content_f = vgg(content)
    style_f = vgg(style)
    if interpolation_weights:
        _, C, H, W = content_f.size()
        feat = torch.FloatTensor(1, C, H, W).zero_().to(device)
        base_feat = adaptive_instance_normalization(content_f, style_f)
        for i, w in enumerate(interpolation_weights):
            feat = feat + w * base_feat[i:i + 1]
        content_f = content_f[0:1]
    else:
        feat = adaptive_instance_normalization(content_f, style_f)
    feat = feat * alpha + content_f * (1 - alpha)
    return decoder(feat)

def webcam(style_path, width=1280, height=720):
    """
    Captures and saves an image, perform style transfer, and again saves the styled image.
    Reads the styled image and show in window. 
    """

    # Load Transformer Network
    print("Loading Network")
    decoder = net.decoder
    vgg = net.vgg

    decoder.eval()
    vgg.eval()

    decoder.load_state_dict(torch.load('models/decoder.pth'))
    vgg.load_state_dict(torch.load('models/vgg_normalized.pth'))
    vgg = nn.Sequential(*list(vgg.children())[:31])

    vgg.to(device)
    decoder.to(device)
    print("Done Loading Network")

    # Set webcam settings
    cam = cv2.VideoCapture(0)
    cam.set(3, width)
    cam.set(4, height)

    # Load style image and convert it to tensor
    style_img=cv2.imread(style_path)
    style_img=cv2.resize(style_img,(int(style_img.shape[1]/style_img.shape[0]*400),400))
    style_tensor=itot(style_img)

    # Main loop
    with torch.no_grad():
        while True:
            startTime=time.time()

            # Get webcam input
            ret_val, content_img = cam.read()

            # Mirror and resize
            content_img = cv2.flip(content_img, 1)
            content_img = cv2.resize(content_img,(0,0),fx=0.8,fy=0.8,interpolation=cv2.INTER_CUBIC)

            # Free-up unneeded cuda memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Generate image
            content_tensor = itot(content_img)

            with torch.no_grad():
                generated_tensor = style_transfer(vgg, decoder, content_tensor, style_tensor,alpha=0.6)

            generated_img = ttoi(generated_tensor.detach())
            generated_img = generated_img / 255

            # concatenate original image and generated image
            factor=0.4*content_img.shape[0]/style_img.shape[0]
            resized_style_img = cv2.resize(style_img,(0,0),fx=factor,fy=factor, interpolation = cv2.INTER_CUBIC)
            content_img[0:resized_style_img.shape[0],0:resized_style_img.shape[1],:]=resized_style_img
            output=np.concatenate((content_img/255,generated_img),axis=1)

            # Show webcam
            cv2.namedWindow('Demo webcam',cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Demo webcam',width,int(0.5*height))
            cv2.imshow('Demo webcam', output)
            if cv2.waitKey(1) == 27: 
                break  # esc to quit

            print("Generation time of last frame: {}".format(time.time()-startTime))
            
    # Free-up memories
    cam.release()
    cv2.destroyAllWindows()

webcam(STYLE_PATH, WIDTH, HEIGHT)

