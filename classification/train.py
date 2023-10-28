from model import VGG19
from data_gen import TrainGen,TestGen
import os

n = 10

data = TrainGen()
train_data, val_data = data.load()

input_shape = (896,896,3)
model = VGG19(input_shape,3)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(train_data,epochs=n,validation_data=val_data)
test_loss, test_accuracy = model.evaluate(TestGen().load())
print(test_loss,test_accuracy)

model.save(os.path.join('model','save.h5'))
