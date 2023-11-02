from keras.models import load_model
import os
import matplotlib.image as mpimg
import numpy as np
from matplotlib import pyplot as plt
from data_gen import TestGen

path = os.path.join('model','best','resnet50_2023-10-23-39-27.h5')

model = load_model(path)

test_loss, test_accuracy = model.evaluate(TestGen().load())
print(test_loss,test_accuracy)
preds = model.predict(TestGen().load())
mapper = {0:['GatewayOfIndia','Gateway Of India'],1:['CharMinar','Char Minar'],2:['lotusTemple','Lotus Temple']}

path_dir = os.path.join('data','test')
path_dir_res = os.path.join('data','test_res')

for path_idx,i in enumerate(preds):
    val = np.argmax(i)
    monument_name = mapper[val][1]
    monument_dir = mapper[val][0]

    img_path1 = os.path.join(path_dir,monument_dir,os.listdir(os.path.join(path_dir,monument_dir))[0])
    img_path2 = os.path.join(path_dir_res,monument_dir,os.listdir(os.path.join(path_dir_res,monument_dir))[0])

    img1 = mpimg.imread(img_path1)  
    plt.subplot(1, 3, 1)  
    plt.imshow(img1) 
    res_h = str(img1.shape[0])
    res_w = str(img1.shape[1])
    plt.title('Low Res'+" "+res_w+'x'+res_h)
    plt.axis('off')

    img2 = mpimg.imread(img_path2) 
    plt.subplot(1, 3, 2) 
    plt.imshow(img2) 
    res_h = str(img2.shape[0])
    res_w = str(img2.shape[1])
    plt.title('High Res'+" "+res_w+'x'+res_h)
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.text(0.5, 0.5, 'Predicted: '+monument_name, fontsize=12, ha='center')
    plt.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join('preds',monument_name+'.png'))
    plt.show()
    