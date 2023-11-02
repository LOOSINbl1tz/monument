from model import VGG16
from data_gen import TrainGen,TestGen
import os
from keras.callbacks import TensorBoard
import datetime

n = 10

data = TrainGen()
train_data, val_data = data.load()

input_shape = (224,224,3)

vgg16_model = VGG16(input_shape,3)
vgg16_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

log_dir = os.path.join('logs','fit')  
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1, write_graph=True)

vgg16_model.fit(train_data,epochs=n,validation_data=val_data,callbacks=[tensorboard_callback])
test_loss, test_accuracy = vgg16_model.evaluate(TestGen().load())
print(test_loss,test_accuracy)

time = datetime.datetime.now()
extracted_time = [time.year,time.month,time.hour,time.minute,time.second]
extracted_time = [str(i) for i in extracted_time]

model_name = 'vgg16_'+'-'.join(extracted_time)+'.h5'
vgg16_model.save(os.path.join('model',model_name))