import os.path as osp
import glob
import cv2
import numpy as np
import torch
import RRDBNet_arch as arch
from PIL import Image

model_path = 'models/RRDB_ESRGAN_x4.pth'  
device = torch.device('cuda')  

test_img_folder = 'LR/*'

model = arch.RRDBNet(3, 3, 64, 23, gc=32)
model.load_state_dict(torch.load(model_path), strict=True)
model.eval()
model = model.to(device)

print('Model path {:s}. \nTesting...'.format(model_path))

idx = 0
for path in glob.glob(test_img_folder):
    idx += 1
    base = osp.splitext(osp.basename(path))[0]
    print(idx, base)
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    img = img * 1.0 / 255
    img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
    img_LR = img.unsqueeze(0)
    img_LR = img_LR.to(device)

    with torch.no_grad():
        output = model(img_LR).data.squeeze().float().cpu().clamp_(0, 1).numpy()
    output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
    output = (output * 255.0).round()
    cv2.imwrite('results/{:s}_rlt.png'.format(base), output)
    image = Image.open('results/{:s}_rlt.png'.format(base))
    width = 800
    height = 800
    new_size = (width, height)
    resized_image = image.resize(new_size, Image.LANCZOS)

    # Save as PNG to maintain quality
    resized_image.save('results/{:s}_rlt.png'.format(base))