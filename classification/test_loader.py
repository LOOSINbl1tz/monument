import os.path as osp
import os
import glob
import cv2
import numpy as np
import torch
import RRDBNet_arch as arch
from PIL import Image

def esrgan(path,q,i):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    img = img * 1.0 / 255
    img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
    img_LR = img.unsqueeze(0)
    img_LR = img_LR.to(device)
    base = os.path.join('data','test_res',q)
    base_img = os.path.join('data','test_res',q,i)
    if not os.path.exists(base):
        os.mkdir(base)

    print(base_img)

    with torch.no_grad():
        output = model(img_LR).data.squeeze().float().cpu().clamp_(0, 1).numpy()

    output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
    output = (output * 255.0).round()
    cv2.imwrite(base_img, output)


model_path = 'gan/RRDB_ESRGAN_x4.pth' 
device = torch.device('cuda')  

test_img_folder = os.path.join('data','test')

model = arch.RRDBNet(3, 3, 64, 23, gc=32)
model.load_state_dict(torch.load(model_path), strict=True)
model.eval()
model = model.to(device)

for q in os.listdir(test_img_folder):
    for i in os.listdir(os.path.join(test_img_folder,q)):
        image_path = os.path.join(test_img_folder,q,i)
        esrgan(image_path,q,i)
        