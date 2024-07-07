import pathlib
import pandas as pd
import os
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
import keras
from keras.layers import LeakyReLU
from keras import Sequential,Input,Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import BatchNormalization



dir = "C:/Users/egle0/OneDrive/Dokumentai/duomenuanalizevirusu/DATA/"

y = ['Golf','bmw serie 1','chevrolet spark','chevroulet aveo','clio','duster','hyundai i10','hyundai tucson','logan',
     'megane','mercedes class a','nemo citroen','octavia','picanto','polo','sandero','seat ibiza','symbol','toyota corolla',
     'volkswagen tiguan']

images = 64
df = []
labels = []

for category in y:
 category_dir = os.path.join(dir, category)

 for filename in os.listdir(category_dir):
     # Check if the file is an image (you might want to add more sophisticated checks)
     if filename.endswith(".jpg") or filename.endswith(".png"):
         # Load image with error handling
         try:
             image_path = os.path.join(category_dir, filename)
             image = Image.open(image_path)

             # Resize image and convert to grayscale
             image = image.resize((images, images)).convert('L')

             #convert to grace scale
             # Define transform
             transform = transforms.Grayscale()
             # pilka spalva
             datai = transform(image)

             # Convert image to numpy array
             image_array = np.array(datai)

             # Append the image array to the data list
             df.append(image_array)

             # Append the label corresponding to the category
             labels.append(y.index(category))  # Use index as label (assuming categories are ordered)
         except Exception as e:
             print(f"Error loading image {filename}: {e}")

# Convert data and labels to numpy arrays
df = np.array(df)
labels = np.array(labels)
class_names = [ 'Golf','bmw serie 1','chevrolet spark','chevroulet aveo','clio','duster','hyundai i10','hyundai tucson','logan',
     'megane','mercedes class a','nemo citroen','octavia','picanto','polo','sandero','seat ibiza','symbol','toyota corolla',
     'volkswagen tiguan' ]



# Set random seed for purposes of reproducibility
seed = 21

plt.figure(figsize=(10,10))
for i in range(10):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(df[i],  cmap=plt.cm.binary)
    plt.xlabel(class_names[labels[i] == labels[2]])
plt.show()

#spliting to train and test
X_train, X_test, y_train, y_test = train_test_split(df, labels, test_size=0.2, random_state=42)


#how much diffrent clasess we have
def klases(labels):
    n = {}
    for i in labels:
        if i in n:
             n[i]+=1
        else:
            n[i] = 1
    return n

n = klases(labels)
sugrupuota = sorted(n.items(), key=lambda item: item[1])
print(sugrupuota)


batch_size = 32
epochs = 30
num_classes = 20

#creating model
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),activation='linear',input_shape=(64,64,1),padding='same'))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D((2, 2),padding='same'))
model.add(Conv2D(128, (3, 3), activation='linear',padding='same'))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
model.add(Conv2D(128, (3, 3), activation='linear',padding='same'))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
model.add(Flatten())
model.add(Dense(128, activation='linear'))
model.add(LeakyReLU(alpha=0.1))
model.add(Dense(num_classes, activation='softmax'))


model.compile(loss=keras.losses.sparse_categorical_crossentropy, optimizer=keras.optimizers.Adam(),metrics=['accuracy'])
model.summary()

#cheaking shape
X_train.shape
y_train.shape
X_test.shape
y_test.shape
#training model
train = model.fit(X_train,y_train, batch_size=batch_size,epochs=epochs,validation_split=0.2)
test_eval = model.evaluate(X_test, y_test, verbose=0)