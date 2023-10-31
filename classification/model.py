from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.layers import BatchNormalization, Activation, GlobalAveragePooling2D, Dense, Add
from keras.models import Model, Sequential

# Define the VGG19 architecture
def VGG19(input_shape, num_classes):
    # Input layer
    input_tensor = Input(shape=input_shape)

    # Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(input_tensor)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv4')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv4')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv4')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    # Fully connected layers
    x = Flatten(name='flatten')(x)
    x = Dense(128, activation='relu', name='fc1')(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu', name='fc2')(x)
    x = Dropout(0.5)(x)
    x = Dense(512, activation='relu', name='fc3')(x)
    x = Dropout(0.5)(x)
    x = Dense(512, activation='relu', name='fc4')(x)
    x = Dropout(0.5)(x)
    x = Dense(512, activation='relu', name='fc5')(x)
    x = Dropout(0.5)(x)
    x = Dense(512, activation='relu', name='fc6')(x)
    x = Dropout(0.5)(x)
    x = Dense(num_classes, activation='softmax', name='predictions')(x)

    # Create the model
    model = Model(inputs=input_tensor, outputs=x, name='vgg19')

    return model

def residual_unit(x, filters, stride=1):
    identity = x
    
    x = Conv2D(filters, (1, 1), strides=(stride, stride), padding='valid')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    x = Conv2D(filters, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    x = Conv2D(filters * 4, (1, 1))(x)
    x = BatchNormalization()(x)
    
    if stride != 1 or identity.shape[-1] != filters * 4:
        identity = Conv2D(filters * 4, (1, 1), strides=(stride, stride))(identity)
    
    x = Add()([x, identity])
    x = Activation('relu')(x)
    
    return x

def ResNet50(input_shape=(224, 224, 3),classes = 3):
    input_tensor = Input(shape=input_shape)
    
    x = Conv2D(64, (7, 7), strides=(2, 2), padding='same')(input_tensor)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    
    x = residual_unit(x, 64)
    x = residual_unit(x, 64)
    x = residual_unit(x, 64)
    
    x = residual_unit(x, 128, stride=2)
    x = residual_unit(x, 128)
    x = residual_unit(x, 128)
    x = residual_unit(x, 128)
    
    x = residual_unit(x, 256, stride=2)
    x = residual_unit(x, 256)
    x = residual_unit(x, 256)
    x = residual_unit(x, 256)
    x = residual_unit(x, 256)
    
    x = residual_unit(x, 512, stride=2)
    x = residual_unit(x, 512)
    x = residual_unit(x, 512)
    
    x = GlobalAveragePooling2D()(x)
    x = Dense(classes, activation='softmax')(x)
    
    model = Model(inputs=input_tensor, outputs=x)
    
    return model

def VGG16(input_shape=(224, 224, 3), num_classes=3):
    model = Sequential()
    
    # Block 1
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=input_shape, name='block1_conv1'))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool'))
    
    # Block 2
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1'))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool'))
    
    # Block 3
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1'))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2'))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool'))
    
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1'))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2'))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool'))
    
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1'))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2'))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool'))
    
    model.add(Flatten(name='flatten'))
    model.add(Dense(128, activation='relu', name='fc1'))
    model.add(Dense(256, activation='relu', name='fc2'))
    model.add(Dense(512, activation='relu', name='fc3'))
    model.add(Dense(num_classes, activation='softmax', name='predictions'))
    
    return model