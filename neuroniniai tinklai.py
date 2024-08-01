import numpy as np
import pathlib
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import os

#loading data

train_input = pathlib.Path(r'C:\Users\egle0\Documents\pythono projektai\d\train')
valid_input = pathlib.Path(r'C:\Users\egle0\Documents\pythono projektai\d\valid')
test_input = pathlib.Path(r'C:\Users\egle0\Documents\pythono projektai\d\test')

#hot encoding
data_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)
#generating data
train_data = data_generator.flow_from_directory(train_input,
                                                target_size=(64, 64),
                                                batch_size=50,
                                                shuffle=True,
                                                class_mode='categorical',
                                                classes=['ace of clubs', 'ace of diamonds',
                                                         'ace of hearts','ace of spades','eight of clubs','eight of diamonds',
                                                         'eight of hearts','eight of spades','five of clubs','five of diamonds',
                                                         'five of hearts', 'five of spades', 'four of clubs', 'four of diamonds',
                                                         'four of hearts', 'four of spades', 'jack of clubs', 'jack of diamonds',
                                                         'jack of hearts', 'jack of spades', 'joker', 'king of clubs', 'king of diamonds',
                                                         'king of hearts', 'king of spades', 'nine of clubs', 'nine of diamonds',
                                                         'nine of hearts', 'nine of spades', 'queen of clubs', 'queen of diamonds',
                                                         'queen of hearts', 'queen of spades', 'seven of clubs', 'seven of diamonds',
                                                         'seven of hearts', 'seven of spades', 'six of clubs', 'six of diamonds',
                                                         'six of hearts', 'six of spades', 'ten of clubs', 'ten of diamonds',
                                                         'ten of hearts', 'ten of spades', 'three of clubs', 'three of diamonds',
                                                         'three of hearts', 'three of spades', 'two of clubs', 'two of diamonds',
                                                         'two of hearts', 'two of spades'],
                                                seed=42)

test_data = data_generator.flow_from_directory(test_input,
                                               target_size=(64, 64),
                                               batch_size=50,
                                               shuffle=True,
                                               class_mode='categorical',
                                               classes=['ace of clubs', 'ace of diamonds',
                                                        'ace of hearts', 'ace of spades', 'eight of clubs',
                                                        'eight of diamonds',
                                                        'eight of hearts', 'eight of spades', 'five of clubs',
                                                        'five of diamonds',
                                                        'five of hearts', 'five of spades', 'four of clubs',
                                                        'four of diamonds',
                                                        'four of hearts', 'four of spades', 'jack of clubs',
                                                        'jack of diamonds',
                                                        'jack of hearts', 'jack of spades', 'joker', 'king of clubs',
                                                        'king of diamonds',
                                                        'king of hearts', 'king of spades', 'nine of clubs',
                                                        'nine of diamonds',
                                                        'nine of hearts', 'nine of spades', 'queen of clubs',
                                                        'queen of diamonds',
                                                        'queen of hearts', 'queen of spades', 'seven of clubs',
                                                        'seven of diamonds',
                                                        'seven of hearts', 'seven of spades', 'six of clubs',
                                                        'six of diamonds',
                                                        'six of hearts', 'six of spades', 'ten of clubs',
                                                        'ten of diamonds',
                                                        'ten of hearts', 'ten of spades', 'three of clubs',
                                                        'three of diamonds',
                                                        'three of hearts', 'three of spades', 'two of clubs',
                                                        'two of diamonds',
                                                        'two of hearts', 'two of spades'],
                                               seed=42)

valid_data = data_generator.flow_from_directory(valid_input,
                                                target_size=(64, 64),
                                                batch_size= 50,
                                                shuffle=True,
                                                class_mode='categorical',
                                                classes=['ace of clubs', 'ace of diamonds',
                                                         'ace of hearts','ace of spades','eight of clubs','eight of diamonds',
                                                         'eight of hearts','eight of spades','five of clubs','five of diamonds',
                                                         'five of hearts', 'five of spades', 'four of clubs', 'four of diamonds',
                                                         'four of hearts', 'four of spades', 'jack of clubs', 'jack of diamonds',
                                                         'jack of hearts', 'jack of spades', 'joker', 'king of clubs', 'king of diamonds',
                                                         'king of hearts', 'king of spades', 'nine of clubs', 'nine of diamonds',
                                                         'nine of hearts', 'nine of spades', 'queen of clubs', 'queen of diamonds',
                                                         'queen of hearts', 'queen of spades', 'seven of clubs', 'seven of diamonds',
                                                         'seven of hearts', 'seven of spades', 'six of clubs', 'six of diamonds',
                                                         'six of hearts', 'six of spades', 'ten of clubs', 'ten of diamonds',
                                                         'ten of hearts', 'ten of spades', 'three of clubs', 'three of diamonds',
                                                         'three of hearts', 'three of spades', 'two of clubs', 'two of diamonds',
                                                         'two of hearts', 'two of spades'],
                                                seed=42)

class_names = train_data.class_indices
print(class_names)
print(len(class_names))
print('Training shape: ', train_data.image_shape)
print('Testing shape: ', test_data.image_shape)

#model parameters

epochs = 20
num_classes = 53

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, 3, padding='same', activation='linear', input_shape=(64, 64, 3)),
    tf.keras.layers.LeakyReLU(alpha=0.1),
    tf.keras.layers.MaxPooling2D((2, 2), padding='same'),
    tf.keras.layers.Conv2D(64,  3, activation='linear',padding='same'),
    tf.keras.layers.MaxPooling2D((2, 2), padding='same'),
    tf.keras.layers.LeakyReLU(alpha=0.1),
    tf.keras.layers.Conv2D(128,  3, activation='linear',padding='same'),
    tf.keras.layers.MaxPooling2D((2, 2), padding='same'),
    tf.keras.layers.LeakyReLU(alpha=0.1),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='linear'),
    tf.keras.layers.LeakyReLU(alpha=0.1),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(num_classes, activation='softmax')])

model.compile(loss='categorical_crossentropy',
              optimizer='adam',  # (learning_rate=0.001), pasirinkti skirtingus optimaizerius!!!!
              metrics=['accuracy'])
model.summary()



history = model.fit(train_data, validation_data=valid_data, epochs=epochs)

model.save("C:/Users/egle0/Documents/pythono projektai/d/modelisEP.h5")


image_path = "C:/Users/egle0/Documents/pythono projektai/d/grafikas.jpg"
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
pd.DataFrame(history.history).plot(grid=True)
plt.show()

plt.savefig(image_path)

prediction = model.predict(valid_data)
print(prediction)

model.evaluate(test_data)