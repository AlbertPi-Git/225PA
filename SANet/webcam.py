import time
import argparse
import os
import torch
import torch.nn as nn
import cv2
import numpy as np
import model

parser = argparse.ArgumentParser()
parser.add_argument('--style', type=str,
                    help='File path to the style image')

args= parser.parse_args()

WIDTH = 1280
HEIGHT = 720
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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

def webcam(style_path, width=1280, height=720):
    """
    Captures and saves an image, perform style transfer, and again saves the styled image.
    Reads the styled image and show in window. 
    """

    # Load model
    decoder = model.decoder
    transform = model.Transform(in_planes = 512)
    vgg = model.vgg

    decoder.eval()
    transform.eval()
    vgg.eval()

    decoder.load_state_dict(torch.load("decoder_iter_500000.pth"))
    transform.load_state_dict(torch.load("transformer_iter_500000.pth"))
    vgg.load_state_dict(torch.load("vgg_normalised.pth"))

    norm = nn.Sequential(*list(vgg.children())[:1])
    enc_1 = nn.Sequential(*list(vgg.children())[:4])  # input -> relu1_1
    enc_2 = nn.Sequential(*list(vgg.children())[4:11])  # relu1_1 -> relu2_1
    enc_3 = nn.Sequential(*list(vgg.children())[11:18])  # relu2_1 -> relu3_1
    enc_4 = nn.Sequential(*list(vgg.children())[18:31])  # relu3_1 -> relu4_1
    enc_5 = nn.Sequential(*list(vgg.children())[31:44])  # relu4_1 -> relu5_1

    norm.to(device)
    enc_1.to(device)
    enc_2.to(device)
    enc_3.to(device)
    enc_4.to(device)
    enc_5.to(device)
    transform.to(device)
    decoder.to(device)

    # Load style image
    style_img= cv2.imread(style_path)
    style_img=cv2.resize(style_img,(int(style_img.shape[1]/style_img.shape[0]*400),400))
    style_tensor=itot(style_img)

    # Set webcam settings
    cam = cv2.VideoCapture(0)
    cam.set(3, width)
    cam.set(4, height)

    # Main loop
    with torch.no_grad():
        while True:
            startTime=time.time()

            # Get webcam input
            ret_val, content_img = cam.read()

            # Mirror and resize
            content_img = cv2.flip(content_img, 1)
            content_img = cv2.resize(content_img,(0,0),fx=0.8,fy=0.8,interpolation=cv2.INTER_CUBIC)
            content_tensor=itot(content_img)

            # Free-up unneeded cuda memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # stylize image
            Content4_1 = enc_4(enc_3(enc_2(enc_1(content_tensor))))
            Content5_1 = enc_5(Content4_1)
            Style4_1 = enc_4(enc_3(enc_2(enc_1(style_tensor))))
            Style5_1 = enc_5(Style4_1)
            stylized_tensor = decoder(transform(Content4_1, Style4_1, Content5_1, Style5_1))

            stylized_tensor.clamp(0, 255)
            stylized_img=ttoi(stylized_tensor.detach())
            stylized_img=stylized_img/255


            # concatenate original image and generated image
            factor=0.4*content_img.shape[0]/style_img.shape[0]
            resized_style_img = cv2.resize(style_img,(0,0),fx=factor,fy=factor, interpolation = cv2.INTER_CUBIC)
            content_img[0:resized_style_img.shape[0],0:resized_style_img.shape[1],:]=resized_style_img
            output=np.concatenate((content_img/255,stylized_img),axis=1)

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

webcam(args.style, WIDTH, HEIGHT)

