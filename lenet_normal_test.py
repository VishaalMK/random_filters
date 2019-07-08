'''Trains LeNet on the MNIST dataset.
'''

from __future__ import print_function
import keras
import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import initializers
from keras import regularizers
from keras import backend as K
import matplotlib.pyplot as plt
import cPickle as pickle

np.random.seed(1337) # for reproducibility

batch_size = 128
num_classes = 10
epochs = 1
number_of_runs = 2


# input image dimensions
img_rows, img_cols = 28, 28

# the data, shuffled and split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# split the validation data from the training data
x_val = x_train[:10000,:,:]
x_train = x_train[10000:,:,:]
y_val = y_train[:10000]
y_train = y_train[10000:]

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_val = x_val.reshape(x_val.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_val = x_val.reshape(x_val.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_val = x_val.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_val /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_val.shape[0], 'validation samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_val = keras.utils.to_categorical(y_val, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

test_accuracy = []

for r in range(number_of_runs):
        
	print("Starting run number %d" %(r+1))
	# Model definition
	model = Sequential()
	model.add(Conv2D(4, kernel_size=(5, 5),
		         activation='relu',
		         input_shape=input_shape,kernel_initializer='truncated_normal',kernel_regularizer=regularizers.l2(0.01)))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Conv2D(6, (5, 5), activation='relu',kernel_initializer='truncated_normal',kernel_regularizer=regularizers.l2(0.01)))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Flatten())
	model.add(Dense(100, activation='relu',kernel_regularizer=regularizers.l2(0.01)))
	model.add(Dropout(0.5))
	model.add(Dense(num_classes, activation='softmax',kernel_regularizer=regularizers.l2(0.01)))



	model.compile(loss=keras.losses.categorical_crossentropy,
		      optimizer=keras.optimizers.Adadelta(),
		      metrics=['accuracy'])

	# Tensorboard Visualization
	#tb_callback = keras.callbacks.TensorBoard(log_dir='./logs/project_1/lenet/basic/run_1', histogram_freq=1, batch_size=32, 
	#                            write_graph=True, write_grads=False, write_images=False, embeddings_freq=0, 
	#                            embeddings_layer_names=None, embeddings_metadata=None)

	# Early stopping
	early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.002, patience=3, verbose=0, mode='auto')

	# Training the model
	history = model.fit(x_train, y_train,
		  	    batch_size=batch_size,
		  	    epochs=epochs,
		 	    verbose=1,
		  	    validation_data=(x_val, y_val),
		  	    callbacks=[early_stop])

	# Plots
	# list all data in history
	#print(history.history.keys())
	# summarize history for accuracy
	#plt.plot(history.history['acc'])
	#plt.plot(history.history['val_acc'])
	#plt.title('model accuracy')
	#plt.ylabel('accuracy')
	#plt.xlabel('epoch')
	#plt.legend(['train', 'test'], loc='upper left')
	#plt.show()
	# summarize history for loss
	#plt.plot(history.history['loss'])
	#plt.plot(history.history['val_loss'])
	#plt.title('model loss')
	#plt.ylabel('loss')
	#plt.xlabel('epoch')
	#plt.legend(['train', 'test'], loc='upper left')
	#plt.show()

	# Model evaluation against the test set
	score = model.evaluate(x_test, y_test, verbose=0)
	print('Test loss:', score[0])
	print('Test accuracy:', score[1])
	test_accuracy.append(score[1])

print(test_accuracy)

with open('test.p', 'wb') as f:
    pickle.dump(test_accuracy,f)

#plt.boxplot(test_accuracy)
#plt.title('test accuracy')
#plt.ylabel('accuracy')
#plt.xticks([1, 2, 3], ['run_1', 'run_2', 'run_3'])
#plt.show()

fig = plt.figure()
#fig.suptitle('bold figure suptitle', fontsize=14, fontweight='bold')

ax = fig.add_subplot(111)
ax.boxplot(test_accuracy)

ax.set_title('Test accuracy')
ax.set_xlabel('NN Treatments')
ax.set_ylabel('test_accuracy')
plt.xticks([1], ['basic'])
plt.show()
