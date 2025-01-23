import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

dataset_path = "C:/Users/jusar/Documents/GitHub/AI-Powered-Emotional-Intelligence-for-Autonomous-Vehicles/dataset/"

labels_path = "C:/Users/jusar/Documents/GitHub/AI-Powered-Emotional-Intelligence-for-Autonomous-Vehicles/dataset/labels.csv"  

images = []
labels = []

emotion_dict = {0: 'anger', 1: 'contempt', 2: 'disgust', 3: 'fear', 4: 'happy', 5: 'neutral', 6: 'sad', 7: 'surprise'}

for emotion_id, emotion in emotion_dict.items():
    emotion_folder = os.path.join(dataset_path, emotion)
    for img_name in os.listdir(emotion_folder):
        img_path = os.path.join(emotion_folder, img_name)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  
        img = cv2.resize(img, (48, 48))  
        images.append(img)
        labels.append(emotion_id)

images = np.array(images)
labels = np.array(labels)

images = images / 255.0

X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)

X_train = X_train.reshape(-1, 48, 48, 1)
X_val = X_val.reshape(-1, 48, 48, 1)


# CNN
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

model = Sequential()

# First convolutional layer + Max Pooling
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Second convolutional layer + Max Pooling
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Third convolutional layer + Max Pooling
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Flatten the features before feeding into the Dense layers
model.add(Flatten())

# Fully connected (Dense) layer
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))  # Dropout for regularization

# Output layer (8 classes for the 8 emotions)
model.add(Dense(8, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Model summary to review architecture
model.summary()

# Training
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_val, y_val))

#Evaluation
val_loss, val_acc = model.evaluate(X_val, y_val)
print(f'Validation Loss: {val_loss}')
print(f'Validation Accuracy: {val_acc}')

#Plot
import matplotlib.pyplot as plt

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.show()

#Save
model.save('emotion_detection_model.h5')
