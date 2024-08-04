import joblib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras import Sequential
from keras import optimizers
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout, Flatten
from keras.layers import LeakyReLU

dir = "C:/Users/egle0/OneDrive/Dokumentai/duomenuanalizevirusu/DATA/"
test_dir = "C:/Users/egle0/Documents/pythono projektai/test/"
# hot encoding
# Neural network prefers a range between 0 and 1. You can convert the datasets, this is grace scale from 0 -black to 255 withe
data_generator = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1. / 255,
    validation_split=0.3,
    zoom_range=0.1
)

data_augmentation = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1. / 255,
    validation_split=0.3

)

# generating data
train_data = data_augmentation.flow_from_directory(dir,
                                                   target_size=(64, 64),
                                                   batch_size=50,
                                                   color_mode='grayscale',
                                                   shuffle=True,
                                                   class_mode='categorical',
                                                   classes=['Golf', 'bmw serie 1', 'chevrolet spark', 'chevroulet aveo',
                                                            'clio', 'duster',
                                                            'hyundai i10', 'hyundai tucson', 'logan', 'megane',
                                                            'mercedes class a',
                                                            'nemo citroen', 'octavia', 'picanto', 'polo', 'sandero',
                                                            'seat ibiza',
                                                            'symbol', 'toyota corolla', 'volkswagen tiguan'],
                                                   subset='training',  # Use 'training' subset
                                                   seed=42)

test_data = data_augmentation.flow_from_directory(dir,
                                                  target_size=(64, 64),
                                                  batch_size=50,
                                                  color_mode='grayscale',
                                                  shuffle=True,
                                                  class_mode='categorical',
                                                  classes=['Golf', 'bmw serie 1', 'chevrolet spark', 'chevroulet aveo',
                                                           'clio', 'duster',
                                                           'hyundai i10', 'hyundai tucson', 'logan', 'megane',
                                                           'mercedes class a',
                                                           'nemo citroen', 'octavia', 'picanto', 'polo', 'sandero',
                                                           'seat ibiza',
                                                           'symbol', 'toyota corolla', 'volkswagen tiguan'],
                                                  subset='validation',  # Use 'test' subset
                                                  seed=42)

validation_data = data_generator.flow_from_directory(dir,
                                                     target_size=(64, 64),
                                                     batch_size=50,
                                                     color_mode='grayscale',
                                                     shuffle=True,
                                                     class_mode='categorical',
                                                     classes=['Golf', 'bmw serie 1', 'chevrolet spark',
                                                              'chevroulet aveo', 'clio', 'duster',
                                                              'hyundai i10', 'hyundai tucson', 'logan', 'megane',
                                                              'mercedes class a',
                                                              'nemo citroen', 'octavia', 'picanto', 'polo', 'sandero',
                                                              'seat ibiza',
                                                              'symbol', 'toyota corolla', 'volkswagen tiguan'],
                                                     subset='validation',  # Use 'test' subset
                                                     seed=42)

# cheeking images augmentation
images, labels = next(test_data)
imagesau, labelsau = next(validation_data)


# Plot the images to check the augmentation
def plot_images(images_arr):
    fig, axes = plt.subplots(1, 3, figsize=(10, 10))
    axes = axes.flatten()
    for img, ax in zip(images_arr, axes):
        ax.imshow(np.squeeze(img), cmap='gray')
        ax.axis('off')
    plt.tight_layout()
    plt.show()


plot_images(images[:3])
plt.suptitle("Before data augmentation:")
plot_images(imagesau[:3])
plt.suptitle("After data augmentation:")

# how much diffrent clasess we have
# def klases(labels):
#    n = {}
#    for i in labels:
#        if i in n:
#             n[i]+=1
#        else:
#            n[i] = 1
#    return n

# n = klases(labels)
# sugrupuota = sorted(n.items(), key=lambda item: item[1])
# print(sugrupuota)


epochs = 100
num_classes = 20

# creating model
model = Sequential()
model.add(Conv2D(64, kernel_size=(3, 3), activation='ReLU', input_shape=(64, 64, 1), padding='same'))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D((2, 2), padding='same'))
model.add(Conv2D(128, (3, 3), activation='ReLU', padding='same'))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D((2, 2), padding='same'))
model.add(Conv2D(128, (3, 3), activation='ReLU', padding='same'))
model.add(LeakyReLU(alpha=0.1))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(128, activation='ReLU'))
model.add(LeakyReLU(alpha=0.1))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(), metrics=['accuracy'])
model.summary()

# training model
train = model.fit(train_data, validation_data=test_data, epochs=epochs)
test_eval = model.evaluate(validation_data, verbose=0)
print("Losst: ", test_eval[0], "\nAccuracy:", test_eval[1])
predict = model.predict(validation_data, verbose=1)

# saving some model
filename2 = 'C:/Users/egle0/OneDrive/Dokumentai/duomenuanalizevirusu/1_model.sav'
joblib.dump(model, filename2)

# accuracy and losst curves
accuracy = train.history['accuracy']
loss = train.history['loss']
vaccuracy = train.history['val_accuracy']
vloss = train.history['val_loss']
epochs = range(len(accuracy))
plt.plot(epochs, accuracy, 'bo', label='Training accuracy')
plt.plot(epochs, vaccuracy, 'b', label='Test accuracy v')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Trainig loss')
plt.plot(epochs, vloss, 'b', label='Test loss v')
plt.title('Training and validation loss')
plt.legend()
plt.show()

# Generate predicted classes
predicted_classes = np.argmax(predictions, axis=1)

# Get true classes
true_classes = test_data.classes
class_labels = list(test_data.class_indices.keys())

# Generate the confusion matrix
conf_matrix = confusion_matrix(true_classes, predicted_classes)

# Generate and print the confusion matrix
conf_matrix = confusion_matrix(true_classes, predicted_classes)
cm_display = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=class_labels)

#
plt.title('Confusion Matrix with Highlighted Largest Values')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()
