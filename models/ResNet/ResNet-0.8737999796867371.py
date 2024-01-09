def residual_block(x, filters, kernel_size=3, strides=1):
    # Shortcut connection
    shortcut = x
    
    # Main path
    x = Conv2D(filters=filters,
               kernel_size=kernel_size,
               strides=strides,
               padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    
    x = Conv2D(filters=filters,
               kernel_size=kernel_size,
               strides=strides,
               padding='same')(x)
    x = BatchNormalization()(x)
    
    # Skip connection
    x = Add()([x, shortcut])
    x = ReLU()(x)
    return x

# Define the ResNet9 model using Keras Functional API
def define_resnet9():
    input = Input(shape=(None, None, 3))
    
    # Conv1
    x = Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same')(input)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    
    # Conv2
    x = Conv2D(128, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    
    # Res1
    x = residual_block(x, filters=128)
    x = residual_block(x, filters=128)
    
    # Conv3
    x = Conv2D(256, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    
    # Conv4
    x = Conv2D(512, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    
    # Res2
    x = residual_block(x, filters=512)
    x = residual_block(x, filters=512)
    
    # Classifier
    x = GlobalAveragePooling2D()(x)
    x = Dense(10, activation='softmax')(x)
    
    model = Model(inputs=input, outputs=x)
    
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
    return model