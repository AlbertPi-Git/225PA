import cv2
import torch
import argparse
import torch.nn as nn
from torchvision import transforms
import time
import numpy as np
from network import AvatarNet

# Device
device = ("cuda" if torch.cuda.is_available() else "cpu")

STYLE_PATH = "sample_images/style/starry_night.jpg"
MODEL_CHECKPOINT = "trained_models/check_point.pth"
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

def webcam(style_path, width=1280, height=720):
    """
    Captures and saves an image, perform style transfer, and again saves the styled image.
    Reads the styled image and show in window. 
    """

    # load check point
    if torch.cuda.is_available():
        check_point = torch.load(MODEL_CHECKPOINT)
    else:
        check_point = torch.load(MODEL_CHECKPOINT,map_location=torch.device('cpu'))

    # load network
    network = AvatarNet([1, 6, 11, 20])
    network.load_state_dict(check_point['state_dict'])
    network = network.to(device)

    # Load style image and convert it to tensor
    style_img=cv2.imread(style_path)
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

            print(content_tensor.shape)
            print(style_tensor.shape)
            
            # stylize image
            with torch.no_grad():
                stylized_tensor =  network(content_tensor, [style_tensor], 0.3, 3, 1,
                        None, None, False)
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

webcam(STYLE_PATH, WIDTH, HEIGHT)

