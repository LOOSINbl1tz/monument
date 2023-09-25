from PIL import Image
import os

dir = 'cropped'
data = os.listdir(dir)
print(data)

width = 1734
heigth = 684
j=1
dim = (221,203,1683,851)
for i in data:
    img = Image.open(os.path.join(dir,i))
    img = img.crop(dim)
    img.save(os.path.join('test',j),format='PNG')
    
