import os.path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from pathlib import Path
from tensorflow.keras import layers
from sklearn.metrics import confusion_matrix, roc_auc_score
from keras.models import model_from_json

train_num = 4500
test_num = 500
random_state = 1
labels = ['20-24', '25-29', '30-34', '35-49', '40-44', '45-50']

# Data preparation
image_dir_test = Path('../input/age-prediction/20-50/20-50/test')
image_dir_train = Path('../input/age-prediction/20-50/20-50/train')


def preprocessing(image_dir):
    filepaths = pd.Series(list(image_dir.glob(r'**/*.jpg')),
                          name='Filepath').astype(str)
    ages = filepaths.apply(
        lambda x: os.path.split(os.path.split(x)[0])[1]).astype(np.int16)
    ranged_ages = []
    for age in ages:
        if 20 <= age <= 24:
            ranged_ages.append('20-24')
        elif 25 <= age <= 29:
            ranged_ages.append('25-29')
        elif 30 <= age <= 34:
            ranged_ages.append('30-34')
        elif 35 <= age <= 39:
            ranged_ages.append('35-49')
        elif 40 <= age <= 44:
            ranged_ages.append('40-44')
        elif 45 <= age <= 50:
            ranged_ages.append('45-50')
    s_ages = pd.Series(ranged_ages, name='Age').astype(str)
    images = pd.concat([filepaths, s_ages],
                       axis=1).sample(frac=1).reset_index(drop=True)
    return images


train_df = preprocessing(image_dir_train).sample(
    train_num, random_state=random_state).reset_index(drop=True)
test_df = preprocessing(image_dir_test).sample(
    test_num, random_state=random_state).reset_index(drop=True)

# Loading images
train_generator = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1. / 255,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2)
test_generator = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1. / 255,
    horizontal_flip=True,
    zoom_range=0.2,
)
train_images = train_generator.flow_from_dataframe(dataframe=train_df,
                                                   x_col='Filepath',
                                                   y_col='Age',
                                                   target_size=(120, 120),
                                                   color_mode='rgb',
                                                   class_mode='categorical',
                                                   batch_size=32,
                                                   shuffle=True,
                                                   seed=42,
                                                   subset='training')
val_images = train_generator.flow_from_dataframe(dataframe=train_df,
                                                 x_col='Filepath',
                                                 y_col='Age',
                                                 target_size=(120, 120),
                                                 color_mode='rgb',
                                                 class_mode='categorical',
                                                 batch_size=32,
                                                 shuffle=True,
                                                 seed=42,
                                                 subset='validation')
test_images = test_generator.flow_from_dataframe(dataframe=test_df,
                                                 x_col='Filepath',
                                                 y_col='Age',
                                                 target_size=(120, 120),
                                                 color_mode='rgb',
                                                 class_mode='categorical',
                                                 batch_size=32,
                                                 shuffle=False)


# Functions
def compile_model(model):
    opt = tf.keras.optimizers.Adam(learning_rate=0.0003)
    model.compile(optimizer=opt,
                  loss='categorical_crossentropy',
                  metrics=['acc'])
    print(model.name, 'compiled')


def fit_model(model):
    history = model.fit(train_images,
                        validation_data=val_images,
                        epochs=20,
                        callbacks=[
                            tf.keras.callbacks.EarlyStopping(
                                monitor='val_loss',
                                patience=5,
                                restore_best_weights=True)
                        ])
    print(model.name, 'fitted')
    return history


def plot_history(history):
    history_frame = pd.DataFrame(history.history)
    history_frame.loc[:, ['loss', 'val_loss']].plot()


def evaluate_model(model):
    predicted_emotions_squeezed = np.squeeze(model.predict(test_images))
    true_emotions = test_images.labels
    predicted_emotions = []
    for pred in predicted_emotions_squeezed:
        predicted_emotions.append(np.argmax(pred))
    roc = roc_auc_score(true_emotions,
                        predicted_emotions_squeezed,
                        multi_class='ovr')
    print("Test ROC Score: {:.5f}".format(roc))
    plot_cf_matrix(true_emotions, predicted_emotions)


def plot_cf_matrix(true_emotions, predicted_emotions):
    cf_mat = confusion_matrix(true_emotions, predicted_emotions)
    plt.figure(figsize=(10, 10))
    ax = sns.heatmap(cf_mat, annot=True, cmap='Blues', fmt="d")
    ax.set_title('Confusion Matrix\n\n')
    ax.set_xlabel('\nPredicted Values')
    ax.set_ylabel('Actual Values ')
    ax.xaxis.set_ticklabels(labels)
    ax.yaxis.set_ticklabels(labels)
    plt.show()


def train_model(model):
    compile_model(model)
    history = fit_model(model)
    plot_history(history)
    evaluate_model(model)


def save_model(model):
    model_json = model.to_json()
    with open(model.name + ".json", "w") as json_file:
        json_file.write(model_json)
    model.save_weights(model.name + ".h5")
    print(model.name + " is saved")


def load_model(model):
    json_file = open(model.name + ".json", "r")
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(model.name + ".h5")
    print(model.name + " is loaded")
    return loaded_model


# Model
model = tf.keras.Sequential(
    [
        layers.InputLayer(input_shape=[120, 120, 3]),

        # Block One
        layers.Conv2D(filters=64, kernel_size=3, strides=2, padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPool2D(strides=2),
        layers.Dropout(0.187),

        # Block Two
        layers.Conv2D(filters=128, kernel_size=3, padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPool2D(strides=2),
        layers.Dropout(0.187),

        # Block Three
        layers.Conv2D(filters=256, kernel_size=3, padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPool2D(strides=2),
        layers.Dropout(0.187),

        # Block Four
        layers.Conv2D(filters=512, kernel_size=3, padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPool2D(strides=2),
        layers.Dropout(0.187),

        # Block Five
        layers.Conv2D(filters=1024, kernel_size=3, padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(0.187),

        # Head
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dense(6, activation='softmax')
    ],
    name='model')

# Train and test
train_model(model)
save_model(model)
