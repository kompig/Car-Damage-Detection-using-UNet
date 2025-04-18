import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose, concatenate, Input, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau

# Define directories 
image_dir = r'C:\Users\SujithVarmaMudunuruI\OneDrive - Manipal University Jaipur\Desktop\ACADEMICS\UNET MODEL\UNET_TRAINING\DATA\unet train'  # Replace with your image directory
mask_dir = r'C:\Users\SujithVarmaMudunuruI\OneDrive - Manipal University Jaipur\Desktop\ACADEMICS\UNET MODEL\UNET_TRAINING\DATA\unet masks'  # Replace with your segmentation mask directory

# Parameters
img_size = 256  # Resize images to 128x128
batch_size = 1
num_classes = 9  # Based on your class labels
epochs = 80

# Function to load and preprocess images
def load_data(image_dir, mask_dir, img_size):
    images = []
    masks = []
    
    image_files = sorted(os.listdir(image_dir))
    mask_files = sorted(os.listdir(mask_dir))

    for img_file, mask_file in zip(image_files, mask_files):
        # Load and preprocess image
        img = load_img(os.path.join(image_dir, img_file), target_size=(img_size, img_size))
        img = img_to_array(img) / 255.0
        images.append(img)

        # Load and preprocess mask
        mask = load_img(os.path.join(mask_dir, mask_file), color_mode='grayscale', target_size=(img_size, img_size))
        mask = img_to_array(mask).astype('uint8')
        mask = np.clip(mask, 0, num_classes - 1)  # Clip values to valid range
        masks.append(mask)

    return np.array(images), np.array(masks)

# Load the data
images, masks = load_data(image_dir, mask_dir, img_size)
masks = tf.keras.utils.to_categorical(masks, num_classes=num_classes)

# Split data into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(images, masks, test_size=0.2, random_state=42)

# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)

datagen.fit(X_train)

# Ensure data is converted to tensors with the correct dtype
X_train = tf.convert_to_tensor(X_train, dtype=tf.float32)
y_train = tf.convert_to_tensor(y_train, dtype=tf.float32)
X_val = tf.convert_to_tensor(X_val, dtype=tf.float32)
y_val = tf.convert_to_tensor(y_val, dtype=tf.float32)

# Debugging input structure
print("X_train shape:", X_train.shape, "| dtype:", X_train.dtype)
print("y_train shape:", y_train.shape, "| dtype:", y_train.dtype)
print("X_val shape:", X_val.shape, "| dtype:", X_val.dtype)
print("y_val shape:", y_val.shape, "| dtype:", y_val.dtype)

# Define U-Net model
def build_unet(img_size, num_classes):
    inputs = Input((img_size, img_size, 3))

    # Encoder
    c1 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(inputs)
    c1 = Dropout(0.2)(c1)
    c1 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
    c2 = Dropout(0.2)(c2)
    c2 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
    c3 = Dropout(0.3)(c3)
    c3 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
    p3 = MaxPooling2D((2, 2))(c3)

    c4 = Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
    c4 = Dropout(0.3)(c4)
    c4 = Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
    p4 = MaxPooling2D(pool_size=(2, 2))(c4)

    c5 = Conv2D(1024, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
    c5 = Dropout(0.4)(c5)
    c5 = Conv2D(1024, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)

    # Decoder
    u6 = Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
    c6 = Dropout(0.3)(c6)
    c6 = Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)

    u7 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
    c7 = Dropout(0.3)(c7)
    c7 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)

    u8 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
    c8 = Dropout(0.2)(c8)
    c8 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)

    u9 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = concatenate([u9, c1], axis=3)
    c9 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
    c9 = Dropout(0.2)(c9)
    c9 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)

    outputs = Conv2D(num_classes, (1, 1), activation='softmax')(c9)

    model = Model(inputs=[inputs], outputs=[outputs])
    return model

# Compile the model
model = build_unet(img_size, num_classes)
model.compile(optimizer=Adam(learning_rate=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

# Define learning rate scheduler
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss', 
    factor=0.5, 
    patience=5, 
    min_lr=1e-6
)

# Train the model
history = model.fit(
    x=X_train,
    y=y_train,
    validation_data=(X_val, y_val),
    batch_size=batch_size,
    epochs=epochs,
    verbose=1,
    callbacks=[reduce_lr]
)

# Save the model
model.save('unet_car_damage_final.keras')

val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=1)
print(f"Validation Loss: {val_loss:.4f}")
print(f"Validation Accuracy: {val_accuracy:.4f}")

def calculate_iou(y_true, y_pred):
    y_true = tf.argmax(y_true, axis=-1).numpy().flatten()
    y_pred = tf.argmax(y_pred, axis=-1).numpy().flatten()
    ious = []

    for cls in range(num_classes):
        true_cls = y_true == cls
        pred_cls = y_pred == cls
        intersection = np.logical_and(true_cls, pred_cls).sum()
        union = np.logical_or(true_cls, pred_cls).sum()
        if union == 0:
            ious.append(np.nan)
        else:
            ious.append(intersection / union)

    return ious

# Predict on validation data
y_pred = model.predict(X_val, verbose=1)

# Calculate IoU
ious = calculate_iou(y_val, y_pred)
mean_iou = np.nanmean(ious)

print("Class-wise IoU:", ious)
print(f"Mean IoU: {mean_iou:.4f}")


def dice_coefficient(y_true, y_pred, smooth=1e-6):
    y_true_f = tf.reshape(y_true, [-1, num_classes])
    y_pred_f = tf.reshape(y_pred, [-1, num_classes])
    intersection = tf.reduce_sum(y_true_f * y_pred_f, axis=0)
    denominator = tf.reduce_sum(y_true_f + y_pred_f, axis=0)
    dice = (2. * intersection + smooth) / (denominator + smooth)
    return dice.numpy()

dice_scores = dice_coefficient(y_val, y_pred)
mean_dice = np.mean(dice_scores)

print("Class-wise Dice Coefficients:", dice_scores)
print(f"Mean Dice Coefficient: {mean_dice:.4f}")
