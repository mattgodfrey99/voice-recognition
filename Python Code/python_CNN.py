import librosa
import os
import json

## Get Data ##################################################################

# example data uploaded is just to show the format of folders for it to work,
# the data uploaded is quite small so shouldn't get a very high accuracy from
# it.

DATASET_PATH = "example_dataset" # dataset folder location
JSON_PATH = "data.json" # json file to store new data
SECONDS = 1 # length of time to sample
SAMPLES_TO_CONSIDER = 22050*SECONDS # number of seconds to sample


def preprocess_dataset(dataset_path, json_path, num_mfcc=13, n_fft=2048, hop_length=512):
    # dataset_path (str): Path to dataset
    # json_path (str): Path to json file
    # num_mfcc (int): Number of coefficients to extract
    # n_fft (int): Interval we consider to apply FFT. Measured in # of samples
    # hop_length (int): Sliding window for FFT. Measured in # of samples to return
    
    # dictionary where we'll store mapping, labels, MFCCs and filenames
    data = {
        "mapping": [],
        "labels": [],
        "MFCCs": [],
        "files": []
    }

    # loop through all sub-dirs
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):

        # ensure we're at sub-folder level
        if dirpath is not dataset_path:

            # save label (i.e., sub-folder name) in the mapping
            label = dirpath.split("/")[-1]
            data["mapping"].append(label)
            print("\nProcessing: '{}'".format(label))

            # process all audio files in sub-dir and store MFCCs
            for f in filenames:
                file_path = os.path.join(dirpath, f)

                # load audio file and slice it to ensure length consistency among different files
                signal, sample_rate = librosa.load(file_path)

                # drop audio files with less than pre-decided number of samples
                if len(signal) >= SAMPLES_TO_CONSIDER:

                    # ensure consistency of the length of the signal
                    signal = signal[:SAMPLES_TO_CONSIDER]

                    # extract MFCCs
                    MFCCs = librosa.feature.mfcc(signal, sample_rate, n_mfcc=num_mfcc, n_fft=n_fft,
                                                 hop_length=hop_length)

                    # store data for analysed track
                    data["MFCCs"].append(MFCCs.T.tolist())
                    data["labels"].append(i-1)
                    data["files"].append(file_path)
                    print("{}: {}".format(file_path, i-1))
                    
                
    # save data in json file
    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)


preprocess_dataset(DATASET_PATH, JSON_PATH) #run to actually process data
    
## CNN #######################################################################
    
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

DATA_PATH = "data.json"
SAVED_MODEL_PATH = "model.h5"
EPOCHS = 60 # 60 worked well for chosen data
BATCH_SIZE = 32
LEARNING_RATE = 0.0001 # how quickly model adapts to data

def load_data(data_path):
    
    # Loads training dataset from json file.
    # data_path (str): Path to json file containing data
    # X (ndarray): Inputs
    # y (ndarray): Targets
    
    with open(data_path, "r") as fp:
        data = json.load(fp)

    X = np.array(data["MFCCs"])
    y = np.array(data["labels"])
    print("Training sets loaded!")
    return X, y


def prepare_dataset(data_path, test_size=0.2):
    
    # data_path (str): Path to json file containing data
    # test_size (flaot): Percentage of dataset used for testing
    # X_train (ndarray): Inputs for the train set
    # y_train (ndarray): Targets for the train set
    # X_test (ndarray): Inputs for the test set
    # X_test (ndarray): Targets for the test set
    

    # load dataset
    X, y = load_data(data_path)

    # create train, test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

    # add an axis to nd array
    X_train = X_train[..., np.newaxis]
    X_test = X_test[..., np.newaxis]

    return X_train, y_train, X_test, y_test


def build_model(input_shape, loss="sparse_categorical_crossentropy", learning_rate=0.0001):
    
    # input_shape (tuple): Shape of array representing a sample train. E.g.: (44, 13, 1)
    # loss (str): Loss function to use
    # learning_rate (float):
    # model: TensorFlow model of CNN
    

    # build network architecture using convolutional layers
    model = tf.keras.models.Sequential()

    # 1st conv layer
    model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu', input_shape=input_shape,
                                     kernel_regularizer=tf.keras.regularizers.l2(0.001)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D((3, 3), strides=(2,2), padding='same'))

    # 2nd conv layer
    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu',
                                     kernel_regularizer=tf.keras.regularizers.l2(0.001)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D((3, 3), strides=(2,2), padding='same'))

    # 3rd conv layer
    model.add(tf.keras.layers.Conv2D(32, (2, 2), activation='relu',
                                     kernel_regularizer=tf.keras.regularizers.l2(0.001)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D((2, 2), strides=(2,2), padding='same'))

    # flatten output and feed into dense layer
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    tf.keras.layers.Dropout(0.3)

    # softmax output layer
    model.add(tf.keras.layers.Dense(10, activation='softmax')) # no. outputs for no. of labels

    optimiser = tf.optimizers.Adam(learning_rate=learning_rate)

    # compile model
    model.compile(optimizer=optimiser,
                  loss=loss,
                  metrics=["accuracy"])

    # print model parameters on console
    model.summary()

    return model


def train(model, epochs, batch_size, X_train, y_train, X_test, y_test):
    
    # epochs (int): Num training epochs
    # batch_size (int): Samples per batch
    # X_train (ndarray): Inputs for the train set
    # y_train (ndarray): Targets for the train set
    # history: Training history
   

    # train model
    history = model.fit(X_train,
                        y_train,
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_data=(X_test, y_test))
                        #callbacks=[earlystop_callback])
    return history


def plot_history(history):
    
    # all plots of accuracy and loss

    fig, axs = plt.subplots(2)

    # create accuracy subplot
    axs[0].plot(history.history["accuracy"], label="accuracy")
    axs[0].plot(history.history['val_accuracy'], label="val_accuracy")
    axs[0].set_ylabel("Accuracy")
    axs[0].legend(loc="lower right")
    axs[0].set_title("Accuracy evaluation")

    # create loss subplot
    axs[1].plot(history.history["loss"], label="loss")
    axs[1].plot(history.history['val_loss'], label="val_loss")
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("Loss")
    axs[1].legend(loc="upper right")
    axs[1].set_title("Loss evaluation")

    plt.show()

    
def main():
    # generate train, validation and test sets
    X_train, y_train, X_test, y_test = prepare_dataset(DATA_PATH)

    # create network
    input_shape = (X_train.shape[1], X_train.shape[2], 1)
    model = build_model(input_shape, learning_rate=LEARNING_RATE)

    # train network
    history = train(model, EPOCHS, BATCH_SIZE, X_train, y_train, X_test, y_test)

    # plot accuracy/loss for training/validation set as a function of the epochs
    plot_history(history)

    # save model
    model.save(SAVED_MODEL_PATH)
    
    accuracy = history.history["accuracy"]
    accuracy_val = history.history["val_accuracy"]
    loss = history.history["loss"]
    loss_val = history.history["val_loss"]
    
    return accuracy, accuracy_val, loss, loss_val
    
accuracy,accuracy_val,loss,loss_val = main()
# run whole network + save the accuracy and loss (tested and trained)

  
    