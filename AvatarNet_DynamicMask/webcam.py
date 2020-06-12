"""
Captures and saves an image, perform style transfer, and again saves the styled image.
Reads the styled image and show in window. 
"""

import cv2
import time
import torch
import argparse
import numpy as np
import torch.nn as nn
from models import UNet
from network import AvatarNet
from base import VideoInference
from torchvision import transforms

# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--person_style', type=str,
                    help='File path to the person style image')
parser.add_argument('--background_style', type=str,
                    help='File path to the background style image')
parser.add_argument('--ratio', type=float, default=0.5,
                    help='Style strength ratio')
args= parser.parse_args()

# Device
device = ("cuda" if torch.cuda.is_available() else "cpu")

# Path of pretrained weights of AvatarNet for style transfer
STYLE_NET_CHECKPOINT = "trained_models/AvatarNet.pth"
# Path of pretrained weights of UNet for segmentation
SEGMENTATION_NET_CHECKPOINT = "trained_models/UNet_ResNet18.pth"

# Expected resolution of output video
WIDTH = int(1280/1)
HEIGHT = int(720/1)

# Normalize the torch image (tensor) since the pretrained weights is trained using normalized input
def normalize(img):
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]    
    normalizer=transforms.Compose([transforms.Normalize(mean=MEAN, std=STD)])
    return normalizer(img)

# Reverse the normalized image to normal distribution for displaying
def denormalize(img):
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]
    MEAN = [-mean/std for mean, std in zip(MEAN, STD)]
    STD = [1/std for std in STD]    
    denormalizer=transforms.Compose([transforms.Normalize(mean=MEAN, std=STD)])
    return denormalizer(img)

# Convert OpenCV imread image to RGB style and 0 ~ 1 range. The further convert it to normalized torch tensor with batch channel.
def toTensor(img):
    img =cv2.cvtColor(img,cv2.COLOR_BGR2RGB)/255
    # Transpose from [H, W, C] -> [C, H, W]
    tensor = torch.Tensor(np.array(img).transpose(2,0,1))
    tensor = normalize(tensor)
    # Add the batch_size dimension
    tensor = tensor.unsqueeze(0).to(device)
    return tensor

# Convert torch tensor back to OpenCV image for displaying
def toImage(tensor):
    # Remove the batch_size dimension
    tensor = tensor.squeeze()
    tensor=denormalize(tensor)
    # Transpose from [C, H, W] -> [H, W, C]
    img = tensor.cpu().numpy().transpose(1, 2, 0)
    img =np.array(cv2.cvtColor(img,cv2.COLOR_RGB2BGR)*255,dtype=np.uint8)
    return img

# Load segmentation net model
model = UNet(backbone="resnet18", num_classes=2)
if torch.cuda.is_available():
    trained_dict = torch.load(SEGMENTATION_NET_CHECKPOINT)['state_dict']
else:
    trained_dict = torch.load(SEGMENTATION_NET_CHECKPOINT, map_location="cpu")['state_dict']
model.load_state_dict(trained_dict, strict=False)
model.to(device)
model.eval()

# Create segmentation object
segmentObj = VideoInference(
    model=model,
    video_path=0,
    input_size=320,
    height=HEIGHT,
    width=WIDTH,
    use_cuda=torch.cuda.is_available(),
    draw_mode='matting',
)

# Load style transfer net model
network = AvatarNet([1, 6, 11, 20])
if torch.cuda.is_available():
    trained_dict = torch.load(STYLE_NET_CHECKPOINT)['state_dict']
else:
    trained_dict = torch.load(STYLE_NET_CHECKPOINT,map_location=torch.device('cpu'))['state_dict']
network.load_state_dict(trained_dict, strict=False)
network = network.to(device)

# Load person style image and convert it to tensor
person_style_img=cv2.imread(args.person_style)
person_style_img=cv2.resize(person_style_img,(int(person_style_img.shape[1]/person_style_img.shape[0]*400),400))
person_style_tensor=toTensor(person_style_img)

# Load background style image and convert it to tensor
background_style_img=cv2.imread(args.background_style)
background_style_img=cv2.resize(background_style_img,(int(background_style_img.shape[1]/background_style_img.shape[0]*400),400))
background_style_tensor=toTensor(background_style_img)


# Set webcam settings
cam = cv2.VideoCapture(0)
cam.set(3, 1280)
cam.set(4, 720)     ## !!Minimum width OpenCV can set for camera is 480, size smaller than 480 won't have effect


####################################### Main loop ########################################
with torch.no_grad():
    while True:
        startTime=time.time()

        # Get webcam input
        ret_val, content_img = cam.read()

        # Mirror and resize video frame to exprected size
        content_img = cv2.flip(content_img, 1)
        content_img = cv2.resize(content_img,(WIDTH,HEIGHT),interpolation=cv2.INTER_CUBIC)
        content_tensor=toTensor(content_img)

        # generate person_mask and background_mask
        # shape of mask image is (H,W) and range is 0 ~ 1
        person_mask=np.round(segmentObj.run(content_img))
        background_mask=1-person_mask

        # unsqueeze twice to make form (B,C,H,W) tensor format
        person_mask_tensor=torch.Tensor(person_mask).unsqueeze(0).unsqueeze(0).to(device)
        # person_mask_tensor.unsqueeze_(0)
        background_mask_tensor=torch.Tensor(background_mask).unsqueeze(0).unsqueeze(0).to(device)
        # background_mask_tensor.unsqueeze_(0)

        # Free-up unneeded cuda memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # stylize image
        with torch.no_grad():
            stylized_tensor =  network(content_tensor, [person_style_tensor,background_style_tensor], args.ratio, 3, 1,
                    [person_mask_tensor,background_mask_tensor], None, False)
        
        stylized_img=toImage(stylized_tensor.detach())
        stylized_img = cv2.resize(stylized_img,(WIDTH,HEIGHT),interpolation=cv2.INTER_CUBIC)
        
        # concatenate original image and stylized image
        factor=0.4*content_img.shape[0]/person_style_img.shape[0]
        resized_style_img = cv2.resize(person_style_img,(0,0),fx=factor,fy=factor, interpolation = cv2.INTER_CUBIC)
        content_img[0:resized_style_img.shape[0],0:resized_style_img.shape[1],:]=resized_style_img
        output=np.array(255*np.concatenate((content_img/255,stylized_img/255),axis=1),dtype=np.uint8)

        # cv2.imwrite('webcam.jpg',stylized_img)
        # cv2.imwrite('webcam_comp.jpg',output)

        # Show webcam
        cv2.namedWindow('Demo webcam',cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Demo webcam',WIDTH,int(0.5*HEIGHT))
        cv2.imshow('Demo webcam', output)
        if cv2.waitKey(1) == 27: 
            break  # esc to quit

        print("Generation time of last frame: {}".format(time.time()-startTime))
        
# Free-up memories
cam.release()
cv2.destroyAllWindows()













# cv2.imwrite('selfie.jpg',content_img)
# cv2.imwrite('mask1.jpg',cv2.cvtColor(np.array(255*person_mask,dtype=np.uint8),cv2.COLOR_GRAY2BGR))
# cv2.imwrite('mask2.jpg',cv2.cvtColor(np.array(255*background_mask,dtype=np.uint8),cv2.COLOR_GRAY2BGR))

# person=cv2.bitwise_and(content_img,content_img,mask=np.array(255*person_mask,dtype=np.uint8))/255   
# stylized_img+=person


