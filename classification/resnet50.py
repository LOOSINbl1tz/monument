from model import ResNet50
from data_gen import TrainGen,TestGen
import os
from keras.callbacks import TensorBoard
import datetime

def train_resnet50(n):
    # n = 10

    data = TrainGen()
    train_data, val_data = data.load()

    input_shape = (224,224,3)

    log_dir = os.path.join('logs','fit','resnet50')  
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1, write_graph=True)

    resnet50_model = ResNet50(input_shape,3)
    resnet50_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    resnet50_model.fit(train_data,epochs=n,validation_data=val_data,callbacks=[tensorboard_callback])
    test_loss, test_accuracy = resnet50_model.evaluate(TestGen().load())
    print(test_loss,test_accuracy)

    time = datetime.datetime.now()
    extracted_time = [time.year,time.month,time.hour,time.minute,time.second]
    extracted_time = [str(i) for i in extracted_time]

    model_name = 'resnet50_'+'-'.join(extracted_time)+'.h5'
    resnet50_model.save(os.path.join('model',model_name))
