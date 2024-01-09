def define_model():
	model = Sequential()
	model.add(Conv2D(32,  kernel_size = 3,kernel_initializer='he_normal', activation='relu', input_shape = (32, 32, 3)))
	model.add(BatchNormalization())
	
	model.add(Dropout(0.2))
	
	model.add(Conv2D(64, kernel_size = 3, kernel_initializer='he_normal', strides=1, activation='relu'))
	model.add(BatchNormalization())
	
	model.add(MaxPooling2D((2, 2)))
	model.add(Conv2D(128, kernel_size = 3, strides=1, kernel_initializer='he_normal' ,padding='same', activation='relu'))
	model.add(BatchNormalization())
	
	model.add(MaxPooling2D((2, 2)))
	model.add(Conv2D(64, kernel_size = 3,kernel_initializer='he_normal', activation='relu'))
	model.add(BatchNormalization())
	
	model.add(MaxPooling2D((4, 4)))
	model.add(Dropout(0.4))

	model.add(Flatten())
	model.add(Dense(256,kernel_initializer='he_normal', activation = "relu"))
	model.add(Dropout(0.5))
	model.add(Dense(10, kernel_initializer='glorot_uniform', activation = "softmax"))


	# Compile the model
	model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['acc'])
	return model