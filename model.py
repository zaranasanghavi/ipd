import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam

# Define paths
train_directory = 'skin-disease-datasaet/test_set'  # Replace with your training data path
test_directory = 'skin-disease-datasaet/test_set'      # Replace with your testing data path

# Parameters
height, width = 150, 150  # Set image dimensions
batch_size = 32           # Set batch size

# Create ImageDataGenerator for training and testing
train_datagen = ImageDataGenerator(rescale=1.0/255.0)
test_datagen = ImageDataGenerator(rescale=1.0/255.0)

# Generate batches of tensor image data
train_generator = train_datagen.flow_from_directory(
    train_directory,
    target_size=(height, width),
    batch_size=batch_size,
    class_mode='categorical'  # Use 'categorical' for multi-class classification
)

test_generator = test_datagen.flow_from_directory(
    test_directory,
    target_size=(height, width),
    batch_size=batch_size,
    class_mode='categorical'  # Same as above
)

# Build the model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(height, width, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(8, activation='softmax'))  # 8 classes

# Compile the model
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(
    train_generator,
    epochs=10,
    validation_data=test_generator
)

# Save the model
model.save('disease_classification_model.keras')
