from keras.models import load_model
import os
from tensorflow.keras.preprocessing import image
import numpy as np
from keras.applications.vgg19 import preprocess_input, decode_predictions

path = os.path.join('model','best','resnet50_2023-10-23-39-27.h5')

model = load_model(path)
img_path = os.path.join('data','test','lotusTemple','1.png')
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

predictions = model.predict(x)

mapper = {0:'Gateway Of India',1:'Char Minar',2:'Lotus Temple'}

for val,i in enumerate(predictions[0]):
    if i == 1.0:
        print(mapper[val])



