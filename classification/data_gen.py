from keras.preprocessing.image import ImageDataGenerator
import os

target_size = (896,896)
batch_size = 4  
train_datagen = ImageDataGenerator(rescale=1.0/255.0)
test_datagen = ImageDataGenerator(rescale=1.0/127.5 - 1)

class TrainGen:
    def __init__(self) -> None:
        self.train_dir = os.path.join('data','train')
        self.val_dir = os.path.join('data','val')

    def load(self):
        train_generator = train_datagen.flow_from_directory(
                                            self.train_dir,
                                            target_size=target_size,
                                            batch_size=batch_size,
                                            class_mode='categorical'
                                        )
        val_generator = train_datagen.flow_from_directory(
        self.val_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical'
        )
        return train_generator, val_generator

class TestGen:
    def __init__(self) -> None:
        self.test_dir = os.path.join('data','test')

    def load(self):
        test_generator = test_datagen.flow_from_directory(
            self.test_dir,
            target_size=target_size,
            batch_size=batch_size,
            class_mode='categorical'
            )
        return test_generator