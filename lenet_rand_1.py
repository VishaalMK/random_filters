'''Trains LeNet on the MNIST dataset.
   The number of layers to be set random are specified in the freezing layer step 
'''

from __future__ import print_function
import keras
import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import initializers
from keras import backend as K

np.random.seed(1337) # for reproducibility

batch_size = 75
num_classes = 10
epochs = 14

# Input image dimensions
img_rows, img_cols = 28, 28

# The data, shuffled and split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Split the validation data from the training data
x_val = x_train[:10000,:,:]
x_train = x_train[10000:,:,:]
y_val = y_train[:10000]
y_train = y_train[10000:]

# Assimilate data in necessary fashion
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

# Convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_val = keras.utils.to_categorical(y_val, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# Build model
model = Sequential()
model.add(Conv2D(4, kernel_size=(5, 5),
                 activation='relu',
                 input_shape=input_shape,kernel_initializer='truncated_normal'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(6, (5, 5), activation='relu',kernel_initializer='truncated_normal'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

# Freeze(not included in training/weights not updated) specific number of layers in the model
# model.layers = [Conv,Pool,Conv,Pool,Flatten,Dense,Dropout,Dense]
for layer in  model.layers[:7]:
	layer.trainable = False

# Compile model
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

# Tensorboard Visualization
tb_callback = keras.callbacks.TensorBoard(log_dir='./logs/project_1/lenet/rand_3/run_2', histogram_freq=1, batch_size=32, 
                            write_graph=True, write_grads=False, write_images=False, embeddings_freq=0, 
                            embeddings_layer_names=None, embeddings_metadata=None)

# Early stopping
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.02, patience=3, verbose=0, mode='auto')

# Fit model
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_val, y_val),
          callbacks=[tb_callback,early_stop])

# Evaluation on test set
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

