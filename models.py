from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.applications.densenet import DenseNet201
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.applications.nasnet import NASNetLarge, NASNetMobile
from tensorflow.keras.applications.resnet_v2 import ResNet152V2
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Flatten

def get_model(model_name='inception', num_class=4):
    if model_name == 'inception':
        return get_model_inception(num_class)
    if model_name == 'xception':
        return get_model_xception(num_class)
    if model_name == 'nasnet_mobile':
        return get_model_nasnet_mobile(num_class)
    if model_name == 'nasnet_large':
        return get_model_nasnet_large(num_class)
    if model_name == 'densenet':
        return get_model_densenet(num_class)
    if model_name == 'resnet':
        return get_model_resnet(num_class)
    if model_name == 'mobilenet':
        return get_model_mobilenet(num_class)
    if model_name == 'inception_resnet':
        return get_model_inception_resnet(num_class)
    else:
        return get_model_vgg(num_class)


def get_custom_top(x, num_class):
    x = GlobalAveragePooling2D(name='avg_pool')(x)
    #x = Dense(1024, activation='relu')(x)
    return Dense(num_class, activation='softmax', name='predictions')(x)

def fine_tune(base_model, model, layer_to_train):
    base_model_depth = len(base_model.layers)

    for layer in base_model.layers:
        layer.trainable = False

    if layer_to_train == 0:
        pass
    elif layer_to_train == 'all':
        for layer in base_model.layers:
            layer.trainable = True
    else:
        top_trainable = base_model_depth - layer_to_train
        for layer in model.layers[:top_trainable]:
            layer.trainable = False
        for layer in model.layers[top_trainable:]:
            layer.trainable = True


def get_model_vgg(num_class, layer_to_train=0):
    base_model = VGG16(weights='imagenet', include_top=False)
    x = base_model.output

    # custom top
    x = Flatten(name='flatten')(x)
    x = Dense(4096, activation='relu', name='fc1')(x)
    x = Dense(4096, activation='relu', name='fc2')(x)
    predictions = Dense(num_class, activation='softmax', name='predictions')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    fine_tune(base_model, model, layer_to_train)

    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def get_model_inception(num_class):
    base_model = InceptionV3(weights='imagenet', include_top=False)
    x = base_model.output

    # custom top
    predictions = get_custom_top(x, num_class)

    # this is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)
    # first: train only the top layers (which were randomly initialized)
    # i.e. freeze all convolutional InceptionV3 layers
    for layer in base_model.layers:
        layer.trainable = False

    # compile the model (should be done *after* setting layers to non-trainable)
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def get_model_xception(num_class):
    base_model = Xception(weights='imagenet', include_top=False)
    x = base_model.output

    # custom top
    predictions = get_custom_top(x, num_class)

    model = Model(inputs=base_model.input, outputs=predictions)
    for layer in base_model.layers:
        layer.trainable = False
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def get_model_resnet(num_class):
    base_model = ResNet152V2(weights='imagenet', include_top=False)
    x = base_model.output

    # custom top
    predictions = get_custom_top(x, num_class)

    model = Model(inputs=base_model.input, outputs=predictions)
    for layer in base_model.layers:
        layer.trainable = False
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

    return model



def get_model_inception_resnet(num_class):
    base_model = InceptionResNetV2(weights='imagenet', include_top=False)
    x = base_model.output

    # custom top
    predictions = get_custom_top(x, num_class)

    model = Model(inputs=base_model.input, outputs=predictions)
    for layer in base_model.layers:
        layer.trainable = False
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def get_model_mobilenet(num_class):
    base_model = MobileNetV2(weights='imagenet', include_top=False)
    x = base_model.output

    # custom top
    predictions = get_custom_top(x, num_class)

    model = Model(inputs=base_model.input, outputs=predictions)
    for layer in base_model.layers:
        layer.trainable = False
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def get_model_densenet(num_class):
    base_model = DenseNet201(weights='imagenet', include_top=False)
    x = base_model.output

    # custom top
    predictions = get_custom_top(x, num_class)

    model = Model(inputs=base_model.input, outputs=predictions)
    for layer in base_model.layers:
        layer.trainable = False
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def get_model_nasnet_mobile(num_class):
    base_model = NASNetMobile(weights='imagenet', include_top=False)
    x = base_model.output

    # custom top
    predictions = get_custom_top(x, num_class)

    model = Model(inputs=base_model.input, outputs=predictions)
    for layer in base_model.layers:
        layer.trainable = False
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def get_model_nasnet_large(num_class):
    base_model = NASNetLarge(weights='imagenet', include_top=False)
    x = base_model.output

    # custom top
    predictions = get_custom_top(x, num_class)

    model = Model(inputs=base_model.input, outputs=predictions)
    for layer in base_model.layers:
        layer.trainable = False
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

