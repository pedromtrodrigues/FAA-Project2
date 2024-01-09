def define_model():
	model = Sequential()

	model.add(Conv2D(32, (3,3), padding='same', activation='relu', input_shape=(32,32,3)))
	model.add(BatchNormalization())
	model.add(Conv2D(32, (3,3), padding='same', activation='relu'))
	model.add(BatchNormalization())
	model.add(MaxPooling2D(pool_size=(2,2)))
	model.add(Dropout(0.3))

	model.add(Conv2D(64, (3,3), padding='same', activation='relu'))
	model.add(BatchNormalization())
	model.add(Conv2D(64, (3,3), padding='same', activation='relu'))
	model.add(BatchNormalization())
	model.add(MaxPooling2D(pool_size=(2,2)))
	model.add(Dropout(0.5))

	model.add(Conv2D(128, (3,3), padding='same', activation='relu'))
	model.add(BatchNormalization())
	model.add(Conv2D(128, (3,3), padding='same', activation='relu'))
	model.add(BatchNormalization())
	model.add(MaxPooling2D(pool_size=(2,2)))
	model.add(Dropout(0.5))

	model.add(Flatten())
	model.add(Dense(128, activation='relu'))
	model.add(BatchNormalization())
	model.add(Dropout(0.5))
	model.add(Dense(10, activation='softmax'))


	# Compile the model
	model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['acc'])
	return model