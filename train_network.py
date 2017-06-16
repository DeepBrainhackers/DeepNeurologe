from sklearn.model_selection import train_test_split
import numpy as np
import os.path as osp
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, CSVLogger, TensorBoard

from data_handling import load_data, load_labels, get_network


def run(data_folder, batch_size=1):
    y = load_labels(data_folder)
    X = load_data(data_folder)
    network = get_network(n_classes=np.unique(y).size)

    X_train, X_valid, X_test, y_train, y_valid, y_test = create_train_valid_test(X, y)
    mean_train = X_train.mean(axis=0, keepdims=True)
    std_train = X_train.std(axis=0, keepdims=True)
    X_train = (X_train - mean_train)/std_train
    X_valid = (X_valid - mean_train)/std_train
    X_test = (X_test - mean_train)/std_train

    y_train = to_categorical(y_train, num_classes=2)
    y_valid = to_categorical(y_valid, num_classes=2)
    y_test = to_categorical(y_test, num_classes=2)

    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.01, patience=200, verbose=1, mode='min')
    lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=1, mode='min', min_lr=1e-5)
    csv_logger = CSVLogger(osp.join(data_folder, 'training.log'))
    tensorboard = TensorBoard(log_dir=osp.join(data_folder, 'tensorboard'), histogram_freq=1, write_graph=True,
                              write_images=True)

    network.fit(x=X_train, y=y_train, batch_size=batch_size, epochs=1000, verbose=1, validation_data=(X_valid, y_valid),
                shuffle=True, callbacks=[early_stopping, lr_scheduler, csv_logger, tensorboard])

    network.evaluate(x=X_test, y=y_test, batch_size=1, verbose=1)


def create_train_valid_test(X, y):
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=60, stratify=y)
    X_valid, X_test, y_valid, y_test = train_test_split(X_valid, y_valid, test_size=30, stratify=y_valid)
    return X_train, X_valid, X_test, y_train, y_valid, y_test


if __name__ == '__main__':
    data_dir = '/home/paulgpu/git/DeepNeurologe'
    run(data_dir)


