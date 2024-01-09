from keras.models import Model
from keras.layers import (
    Input, Conv2D, BatchNormalization, ReLU, Concatenate,
    Add, GlobalAveragePooling2D, Dense, AveragePooling2D
)
from keras.regularizers import l2

def BasicBlock(inputs, filters, stride=1):
    x = Conv2D(filters, kernel_size=(3, 3), strides=stride, padding='same', kernel_initializer='he_normal')(inputs)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = Conv2D(filters, kernel_size=(3, 3), strides=1, padding='same', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)

    shortcut = inputs
    if stride != 1 or inputs.shape[-1] != filters:
        shortcut = Conv2D(filters, kernel_size=(1, 1), strides=stride, padding='same', kernel_initializer='he_normal')(inputs)
        shortcut = BatchNormalization()(shortcut)

    x = Add()([x, shortcut])
    x = ReLU()(x)
    return x

def Root(inputs, out_channels, kernel_size=1):
    x = inputs if isinstance(inputs, list) else [inputs]  # Convert inputs to a list if not already
    x = Concatenate(axis=-1)(x)
    x = Conv2D(out_channels, kernel_size, padding='same', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    return x

def Tree(inputs, block, in_channels, out_channels, level=1, stride=1):
    if level == 1:
        root = Root(inputs, out_channels)
        left_node = block(inputs, out_channels, stride=stride)
        right_node = block(left_node, out_channels, stride=1)
        return root, left_node, right_node
    else:
        root = Root(inputs, out_channels * (level + 2))
        levels = []
        for i in reversed(range(1, level)):
            subtree = Tree(inputs, block, in_channels, out_channels, level=i, stride=stride)
            levels.append(subtree)
        prev_root = block(inputs, out_channels, stride=stride)
        left_node = block(prev_root, out_channels, stride=1)
        right_node = block(left_node, out_channels, stride=1)
        return root, levels, prev_root, left_node, right_node

def DLA(block, num_classes=10):
    inputs = Input(shape=(None, None, 3))

    base = Conv2D(16, kernel_size=(3, 3), strides=1, padding='same', kernel_initializer='he_normal')(inputs)
    base = BatchNormalization()(base)
    base = ReLU()(base)

    layer1 = Conv2D(16, kernel_size=(3, 3), strides=1, padding='same', kernel_initializer='he_normal')(base)
    layer1 = BatchNormalization()(layer1)
    layer1 = ReLU()(layer1)

    layer2 = Conv2D(32, kernel_size=(3, 3), strides=1, padding='same', kernel_initializer='he_normal')(layer1)
    layer2 = BatchNormalization()(layer2)
    layer2 = ReLU()(layer2)

    layer3 = Tree(layer2, block, 32, 64, level=1, stride=1)
    layer4 = Tree(layer3[-1], block, 64, 128, level=2, stride=2)
    layer5 = Tree(layer4[-1], block, 128, 256, level=2, stride=2)
    layer6 = Tree(layer5[-1], block, 256, 512, level=1, stride=2)

    out = GlobalAveragePooling2D()(layer6[-1])
    out = Dense(num_classes, activation='softmax', kernel_regularizer=l2(1e-4))(out)

    model = Model(inputs=inputs, outputs=out)
    return model

# Create the model
def define_model():
    model = DLA(BasicBlock, num_classes=10)    
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
    return model

