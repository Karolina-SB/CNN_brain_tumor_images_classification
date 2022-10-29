# TODO: jak już wybiorę fajny model na 100 (słaby, średni i mocny) to optymalki a następnie to do 100Full i augmentacja i zobaczyć czy da lepsze wyniki
# TODO: znormalizować colorbar w matrixach
# TODO: znormalizować oś y wykresów loss
# TODO: k fold cross valid
# TODO: wydrukować mapy cech (czego uczą się kolejne warstwy)


import multiprocessing
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import warnings
from tensorflow.keras.backend import clear_session
import tensorflow as tf
from datetime import datetime
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import StratifiedKFold
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import gc
from contextlib import redirect_stdout
import os
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

warnings.filterwarnings(action="ignore")

set_size = (224, 224)
batch_size = 32

epochs = 40
mode = 'rgb'
class_mode = "binary"
loss_fun = "binary_crossentropy"
activation_fun = "sigmoid"
output_units = 1
pat = 'bez tego'  # 2
minlr = 'bez tego'  # 0.000001
my_callbacks = []  # [tf.keras.callbacks.ReduceLROnPlateau(patience=pat, min_lr=minlr)]


# tf.keras.callbacks.EarlyStopping(monitor="val_loss", min_delta=0.0, patience=12, mode="auto", restore_best_weights=False)


def print_plots(acc, val_acc, loss, val_loss, y_test, pred, label_a, label_b, results_path, fold):
    fig1, [ax1, ax2] = plt.subplots(nrows=2, ncols=1, figsize=(8, 9))
    plt.tight_layout(pad=5, h_pad=8, w_pad=0.1)

    ax1.plot(acc, label="Trening")
    ax1.plot(val_acc, label="Walidacja")
    ax1.set_ylim([0, 1])
    ax1.tick_params(axis='both', labelsize=12)
    ax1.legend(fontsize=12)
    ax1.set_xlabel("Epoka", fontsize=14, labelpad=8.0)
    ax1.set_ylabel("Dokładność", fontsize=14, labelpad=8.0)
    ax1.set_title("Wykres dokładności na przestrzeni epok", fontsize=17, pad=20.0)

    ax2.plot(loss, label="Trening")  # Training loss
    ax2.plot(val_loss, label="Walidacja")  # Validation Loss
    ax2.legend(fontsize=12)
    ax2.set_ylim([0, max(ax2.get_ylim())])
    ax2.tick_params(axis='both', labelsize=12)
    ax2.set_xlabel("Epoka", fontsize=14, labelpad=8.0)
    ax2.set_ylabel("Strata", fontsize=14, labelpad=8.0)
    ax2.set_title("Wykres straty na przestrzeni epok", fontsize=17, pad=20.0)

    plt.savefig(results_path + 'acc_loss_' + str(fold) + '.png', format='png')
    plt.savefig(results_path + 'acc_loss_' + str(fold) + '.pdf', format='pdf')
    # plt.show()

    fig2, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(5, 4), constrained_layout=True)
    cm = confusion_matrix(y_test, pred, normalize='true')
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', linewidths=0.1, linecolor='black', clip_on=False, annot_kws={"size": 13})
    ax1.set_xlabel('Wynik klasyfikacji', fontsize=14, labelpad=8.0)
    ax1.set_ylabel('Stan faktyczny', fontsize=14, labelpad=8.0)
    ax1.tick_params(axis='both', labelsize=12)
    ax1.xaxis.set_ticklabels([label_a, label_b])
    ax1.yaxis.set_ticklabels([label_a, label_b])
    ax1.set_title('Macierz pomyłek', fontsize=17, pad=20.0)
    plt.savefig(results_path + 'conf_matrix_' + str(fold) + '.png', format='png')
    plt.savefig(results_path + 'conf_matrix_' + str(fold) + '.pdf', format='pdf')
    # plt.show()
    return


def run_model_3_2_kfold(datestamp, name):
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import MaxPooling2D, Dense, Flatten, Conv2D
    from sklearn.metrics import classification_report, recall_score, precision_score, f1_score

    root_results = 'C:/Dane/Wyniki/KFold_Unbalanced/'
    root_data = r'C:/Dane/unbalanced_kfold_train/'
    test_data = r'C:/Dane/unbalanced_kfold_test'
    train_data = pd.read_csv('C:/Users/karol/Documents/STUDIA/Untitled Folder/mgrmgr/unbalanced_train_labels.csv')

    model_name = name + '/'
    results_path = root_results + model_name
    path = os.path.join(results_path, datestamp)
    os.makedirs(path)
    results_path = str(path) + '/'

    def get_model(size):
        tf.keras.backend.clear_session()
        lr = 0.0001  # 0.001 (default)
        opt = Adam(learning_rate=lr)

        model = Sequential()
        model.add(Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), activation="relu", padding="valid", input_shape=(size[0], size[1], 3)))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), activation="relu", padding="valid"))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), activation="relu", padding="valid"))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dense(64, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(output_units, activation=activation_fun))

        model.compile(optimizer=opt, loss=loss_fun, metrics=["accuracy"])
        with open(results_path + 'summary.txt', 'w') as f:
            with redirect_stdout(f):
                model.summary()
        return model

    Y = train_data[['label']]
    skf = StratifiedKFold(n_splits=5, random_state=7, shuffle=True)
    idg = ImageDataGenerator(rescale=(1.0 / 255.0))
    test_data_generator = idg.flow_from_directory(directory=test_data, target_size=set_size,
                                                  class_mode=class_mode, color_mode=mode, batch_size=batch_size, shuffle=False)

    idg_train = ImageDataGenerator(rescale=(1.0/255.0), rotation_range=90, brightness_range=[0.8, 1.2], zoom_range=0.2)

    TEST_ACCURACY = []
    TEST_LOSS = []
    TEST_PRECISION = []
    TEST_RECALL = []
    TEST_F1 = []
    AVR_TIME = []

    count = 0
    for root_dir, cur_dir, files in os.walk(root_data):
        count += len(files)
    print('file count:', count)

    fold_var = 0

    for train_index, val_index in skf.split(np.zeros(count), Y):
        fold_var += 1
        training_data = train_data.iloc[train_index]
        validation_data = train_data.iloc[val_index]

        train_data_generator = idg_train.flow_from_dataframe(training_data,
                                                       x_col="filename", y_col="label", batch_size=batch_size,
                                                       class_mode="binary", target_size=set_size, shuffle=True)
        valid_data_generator = idg.flow_from_dataframe(validation_data,
                                                       x_col="filename", y_col="label", batch_size=batch_size,
                                                       class_mode="binary", target_size=set_size, shuffle=False)

        unique = np.unique(train_data_generator.classes, return_counts=True)
        labels_dict = dict(zip(unique[0], unique[1]))
        weights_for_unbalanced = {0: float(labels_dict[1]/labels_dict[0]), 1: 1.0}
        print("Wagi klas: {}".format(weights_for_unbalanced))




        model = get_model(set_size)

        print("\n--------------------------    Fitting on fold {}    --------------------------\n".format(fold_var))
        start = time.time()
        history = model.fit(train_data_generator, epochs=epochs, callbacks=my_callbacks,
                            validation_data=valid_data_generator, verbose=2, class_weight=weights_for_unbalanced)
        end = time.time()
        elapsed = end - start

        results = model.evaluate(test_data_generator, verbose=2)
        pred = model.predict(test_data_generator)
        pred = np.round(pred).tolist()  # FOR BINARY
        y_test = test_data_generator.labels  # set y_test to the expected output
        recall = recall_score(y_true=y_test, y_pred=pred, average='binary')
        precision = precision_score(y_true=y_test, y_pred=pred, average='binary')
        f1 = f1_score(y_true=y_test, y_pred=pred, average='binary')

        with open(results_path + 'classification_report_' + str(fold_var) + '.txt', 'w') as f:
            with redirect_stdout(f):
                print(classification_report(y_test, pred, digits=4))
                print("\n\n")
                print("Trening trwał: {:.3f} sekund".format(elapsed))
                print("\n")
                print("Test Loss: {:.4f}".format(results[0]))

        results = dict(zip(model.metrics_names, results))
        TEST_ACCURACY.append(results['accuracy'])
        TEST_LOSS.append(results['loss'])
        TEST_PRECISION.append(precision)
        TEST_RECALL.append(recall)
        TEST_F1.append(f1)
        AVR_TIME.append(elapsed)

        acc = history.history["accuracy"]  # report of model
        val_acc = history.history["val_accuracy"]  # history of validation data
        loss = history.history["loss"]  # Training loss
        val_loss = history.history["val_loss"]  # validation loss

        label_a = list(train_data_generator.class_indices.keys())[list(train_data_generator.class_indices.values()).index(0)]
        label_b = list(train_data_generator.class_indices.keys())[list(train_data_generator.class_indices.values()).index(1)]

        print_plots(acc, val_acc, loss, val_loss, y_test, pred, label_a, label_b, results_path, fold_var)

        gc.collect()



    print("Test Accuracy:  {:.4f} (+/- {:.4f})".format(np.mean(TEST_ACCURACY), np.std(TEST_ACCURACY)))
    print("Test Loss:      {:.4f}  (+/- {:.4f})".format(np.mean(TEST_LOSS), np.std(TEST_LOSS)))
    print("Test Precision: {:.4f} (+/- {:.4f})".format(np.mean(TEST_PRECISION), np.std(TEST_PRECISION)))
    print("Test Recall:    {:.4f} (+/- {:.4f})".format(np.mean(TEST_RECALL), np.std(TEST_RECALL)))
    print("Test F1:        {:.4f} (+/- {:.4f})".format(np.mean(TEST_F1), np.std(TEST_F1)))
    print("Test Time:      {:.3f}  (+/- {:.3f})".format(np.mean(AVR_TIME), np.std(AVR_TIME)))

    with open(results_path + 'GENERAL_report.txt', 'w') as f:
        with redirect_stdout(f):
            print("Test Accuracy:  {:.4f} (+/- {:.4f})".format(np.mean(TEST_ACCURACY), np.std(TEST_ACCURACY)))
            print("Test Loss:      {:.4f}  (+/- {:.4f})".format(np.mean(TEST_LOSS), np.std(TEST_LOSS)))
            print("Test Precision: {:.4f} (+/- {:.4f})".format(np.mean(TEST_PRECISION), np.std(TEST_PRECISION)))
            print("Test Recall:    {:.4f} (+/- {:.4f})".format(np.mean(TEST_RECALL), np.std(TEST_RECALL)))
            print("Test F1:        {:.4f} (+/- {:.4f})".format(np.mean(TEST_F1), np.std(TEST_F1)))
            print("\n")
            print("tf.keras.callbacks.ReduceLROnPlateau(patience={}, min_lr={})".format(pat, minlr))

    tf.keras.backend.clear_session()

    return


def run_model_4_3_kfold(datestamp, name):
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import MaxPooling2D, Dense, Flatten, Conv2D
    from sklearn.metrics import classification_report, recall_score, precision_score, f1_score

    root_results = 'C:/Dane/Wyniki/KFold_Unbalanced/'
    root_data = r'C:/Dane/unbalanced_kfold_train/'
    test_data = r'C:/Dane/unbalanced_kfold_test'
    train_data = pd.read_csv('C:/Users/karol/Documents/STUDIA/Untitled Folder/mgrmgr/unbalanced_train_labels.csv')

    model_name = name + '/'
    results_path = root_results + model_name
    path = os.path.join(results_path, datestamp)
    os.makedirs(path)
    results_path = str(path) + '/'

    def get_model(size):
        tf.keras.backend.clear_session()
        lr = 0.0001  # 0.001 (default)
        opt = Adam(learning_rate=lr)

        model = Sequential()
        model.add(Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), activation="relu", padding="valid", input_shape=(set_size[0], set_size[1], 3)))
        model.add(Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), activation="relu", padding="valid"))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), activation="relu", padding="valid"))
        model.add(Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), activation="relu", padding="valid"))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), activation="relu", padding="valid"))
        model.add(Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), activation="relu", padding="valid"))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dense(64, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(output_units, activation=activation_fun))

        model.compile(optimizer=opt, loss=loss_fun, metrics=["accuracy"])
        with open(results_path + 'summary.txt', 'w') as f:
            with redirect_stdout(f):
                model.summary()
        return model

    Y = train_data[['label']]
    skf = StratifiedKFold(n_splits=5, random_state=7, shuffle=True)
    idg = ImageDataGenerator(rescale=(1.0 / 255.0))
    test_data_generator = idg.flow_from_directory(directory=test_data, target_size=set_size,
                                                  class_mode=class_mode, color_mode=mode, batch_size=batch_size, shuffle=False)

    idg_train = ImageDataGenerator(rescale=(1.0/255.0), rotation_range=150, brightness_range=[0.3, 1.5], zoom_range=0.4)


    TEST_ACCURACY = []
    TEST_LOSS = []
    TEST_PRECISION = []
    TEST_RECALL = []
    TEST_F1 = []
    AVR_TIME = []

    count = 0
    for root_dir, cur_dir, files in os.walk(root_data):
        count += len(files)
    print('file count:', count)

    fold_var = 0

    for train_index, val_index in skf.split(np.zeros(count), Y):
        fold_var += 1
        training_data = train_data.iloc[train_index]
        validation_data = train_data.iloc[val_index]

        train_data_generator = idg_train.flow_from_dataframe(training_data,
                                                       x_col="filename", y_col="label", batch_size=batch_size,
                                                       class_mode="binary", target_size=set_size, shuffle=True)
        valid_data_generator = idg.flow_from_dataframe(validation_data,
                                                       x_col="filename", y_col="label", batch_size=batch_size,
                                                       class_mode="binary", target_size=set_size, shuffle=False)

        unique = np.unique(train_data_generator.classes, return_counts=True)
        labels_dict = dict(zip(unique[0], unique[1]))
        weights_for_unbalanced = {0: float(labels_dict[1] / labels_dict[0]), 1: 1.0}
        print("Wagi klas: {}".format(weights_for_unbalanced))

        model = get_model(set_size)

        print("\n--------------------------    Fitting on fold {}    --------------------------\n".format(fold_var))
        start = time.time()
        history = model.fit(train_data_generator, epochs=epochs, callbacks=my_callbacks,
                            validation_data=valid_data_generator, verbose=2, class_weight=weights_for_unbalanced)
        end = time.time()
        elapsed = end - start

        results = model.evaluate(test_data_generator, verbose=2)
        pred = model.predict(test_data_generator)
        pred = np.round(pred).tolist()  # FOR BINARY
        y_test = test_data_generator.labels  # set y_test to the expected output
        recall = recall_score(y_true=y_test, y_pred=pred, average='binary')
        precision = precision_score(y_true=y_test, y_pred=pred, average='binary')
        f1 = f1_score(y_true=y_test, y_pred=pred, average='binary')

        with open(results_path + 'classification_report_' + str(fold_var) + '.txt', 'w') as f:
            with redirect_stdout(f):
                print(classification_report(y_test, pred, digits=4))
                print("\n\n")
                print("Trening trwał: {:.3f} sekund".format(elapsed))
                print("\n")
                print("Test Loss: {:.4f}".format(results[0]))

        results = dict(zip(model.metrics_names, results))
        TEST_ACCURACY.append(results['accuracy'])
        TEST_LOSS.append(results['loss'])
        TEST_PRECISION.append(precision)
        TEST_RECALL.append(recall)
        TEST_F1.append(f1)
        AVR_TIME.append(elapsed)

        acc = history.history["accuracy"]  # report of model
        val_acc = history.history["val_accuracy"]  # history of validation data
        loss = history.history["loss"]  # Training loss
        val_loss = history.history["val_loss"]  # validation loss

        label_a = list(train_data_generator.class_indices.keys())[list(train_data_generator.class_indices.values()).index(0)]
        label_b = list(train_data_generator.class_indices.keys())[list(train_data_generator.class_indices.values()).index(1)]

        print_plots(acc, val_acc, loss, val_loss, y_test, pred, label_a, label_b, results_path, fold_var)

        gc.collect()

    print("Test Accuracy:  {:.4f} (+/- {:.4f})".format(np.mean(TEST_ACCURACY), np.std(TEST_ACCURACY)))
    print("Test Loss:      {:.4f}  (+/- {:.4f})".format(np.mean(TEST_LOSS), np.std(TEST_LOSS)))
    print("Test Precision: {:.4f} (+/- {:.4f})".format(np.mean(TEST_PRECISION), np.std(TEST_PRECISION)))
    print("Test Recall:    {:.4f} (+/- {:.4f})".format(np.mean(TEST_RECALL), np.std(TEST_RECALL)))
    print("Test F1:        {:.4f} (+/- {:.4f})".format(np.mean(TEST_F1), np.std(TEST_F1)))
    print("Test Time:      {:.3f}  (+/- {:.3f})".format(np.mean(AVR_TIME), np.std(AVR_TIME)))

    with open(results_path + 'GENERAL_report.txt', 'w') as f:
        with redirect_stdout(f):
            print("Test Accuracy:  {:.4f} (+/- {:.4f})".format(np.mean(TEST_ACCURACY), np.std(TEST_ACCURACY)))
            print("Test Loss:      {:.4f}  (+/- {:.4f})".format(np.mean(TEST_LOSS), np.std(TEST_LOSS)))
            print("Test Precision: {:.4f} (+/- {:.4f})".format(np.mean(TEST_PRECISION), np.std(TEST_PRECISION)))
            print("Test Recall:    {:.4f} (+/- {:.4f})".format(np.mean(TEST_RECALL), np.std(TEST_RECALL)))
            print("Test F1:        {:.4f} (+/- {:.4f})".format(np.mean(TEST_F1), np.std(TEST_F1)))
            print("Test Time:      {:.3f}  (+/- {:.3f})".format(np.mean(AVR_TIME), np.std(AVR_TIME)))
            print("\n")
            print("tf.keras.callbacks.ReduceLROnPlateau(patience={}, min_lr={})".format(pat, minlr))

    tf.keras.backend.clear_session()

    return



# -----------------------------------------------------------------------------------------------------------------------


def run_model_4_3_kfold_dropout(datestamp, name):
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import MaxPooling2D, Dense, Flatten, Conv2D, Dropout
    from sklearn.metrics import classification_report, recall_score, precision_score, f1_score

    root_results = 'C:/Dane/Wyniki/KFold_Unbalanced/'
    root_data = r'C:/Dane/unbalanced_kfold_train/'
    test_data = r'C:/Dane/unbalanced_kfold_test'
    train_data = pd.read_csv('C:/Users/karol/Documents/STUDIA/Untitled Folder/mgrmgr/unbalanced_train_labels.csv')

    model_name = name + '/'
    results_path = root_results + model_name
    path = os.path.join(results_path, datestamp)
    os.makedirs(path)
    results_path = str(path) + '/'

    def get_model(size, rate1):
        tf.keras.backend.clear_session()
        lr = 0.0001  # 0.001 (default)
        opt = Adam(learning_rate=lr)
        model = Sequential()
        model.add(Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), activation="relu", padding="valid", input_shape=(size[0], size[1], 3)))
        model.add(Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), activation="relu", padding="valid"))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), activation="relu", padding="valid"))
        model.add(Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), activation="relu", padding="valid"))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), activation="relu", padding="valid"))
        model.add(Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), activation="relu", padding="valid"))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(rate1))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(output_units, activation=activation_fun))
        model.compile(optimizer=opt, loss=loss_fun, metrics=["accuracy"])

        with open(results_path + 'summary.txt', 'w') as f:
            with redirect_stdout(f):
                model.summary()
        return model

    Y = train_data[['label']]
    skf = StratifiedKFold(n_splits=5, random_state=7, shuffle=True)
    idg = ImageDataGenerator(rescale=(1.0 / 255.0))
    test_data_generator = idg.flow_from_directory(directory=test_data, target_size=set_size,
                                                  class_mode=class_mode, color_mode=mode, batch_size=batch_size, shuffle=False)

    idg_train = ImageDataGenerator(rescale=(1.0/255.0), rotation_range=150, brightness_range=[0.3, 1.5], zoom_range=0.4)

    TEST_ACCURACY = []
    TEST_LOSS = []
    TEST_PRECISION = []
    TEST_RECALL = []
    TEST_F1 = []
    AVR_TIME = []

    count = 0
    for root_dir, cur_dir, files in os.walk(root_data):
        count += len(files)
    print('file count:', count)

    params_dict = {'rate1': [0.3]}

    for r1 in params_dict['rate1']:
        print("R1: {}".format(r1))
        folder_name = "Drop({})".format(r1)
        param_path = results_path + folder_name
        os.makedirs(param_path)
        param_path = param_path + '/'
        fold_var = 0
        for train_index, val_index in skf.split(np.zeros(count), Y):
            fold_var += 1
            training_data = train_data.iloc[train_index]
            validation_data = train_data.iloc[val_index]
            train_data_generator = idg_train.flow_from_dataframe(training_data,
                                                           x_col="filename", y_col="label", batch_size=batch_size,
                                                           class_mode="binary", target_size=set_size, shuffle=True)
            valid_data_generator = idg.flow_from_dataframe(validation_data,
                                                           x_col="filename", y_col="label", batch_size=batch_size,
                                                           class_mode="binary", target_size=set_size, shuffle=False)

            unique = np.unique(train_data_generator.classes, return_counts=True)
            labels_dict = dict(zip(unique[0], unique[1]))
            weights_for_unbalanced = {0: float(labels_dict[1] / labels_dict[0]), 1: 1.0}
            print("Wagi klas: {}".format(weights_for_unbalanced))

            model = get_model(set_size, r1)
            print("\n--------------------------    Fitting on fold {}    --------------------------\n".format(fold_var))
            start = time.time()
            history = model.fit(train_data_generator, epochs=epochs, callbacks=my_callbacks,
                                validation_data=valid_data_generator, verbose=2, class_weight=weights_for_unbalanced)
            end = time.time()
            elapsed = end - start
            results = model.evaluate(test_data_generator, verbose=2)
            pred = model.predict(test_data_generator)
            pred = np.round(pred).tolist()  # FOR BINARY
            y_test = test_data_generator.labels  # set y_test to the expected output
            recall = recall_score(y_true=y_test, y_pred=pred, average='binary')
            precision = precision_score(y_true=y_test, y_pred=pred, average='binary')
            f1 = f1_score(y_true=y_test, y_pred=pred, average='binary')
            with open(param_path + 'classification_report_' + str(fold_var) + '.txt', 'w') as f:
                with redirect_stdout(f):
                    print(classification_report(y_test, pred, digits=4))
                    print("\n\n")
                    print("Trening trwał: {:.3f} sekund".format(elapsed))
                    print("\n")
                    print("Test Loss: {:.4f}".format(results[0]))
            results = dict(zip(model.metrics_names, results))
            TEST_ACCURACY.append(results['accuracy'])
            TEST_LOSS.append(results['loss'])
            TEST_PRECISION.append(precision)
            TEST_RECALL.append(recall)
            TEST_F1.append(f1)
            AVR_TIME.append(elapsed)
            acc = history.history["accuracy"]  # report of model
            val_acc = history.history["val_accuracy"]  # history of validation data
            loss = history.history["loss"]  # Training loss
            val_loss = history.history["val_loss"]  # validation loss
            label_a = list(train_data_generator.class_indices.keys())[list(train_data_generator.class_indices.values()).index(0)]
            label_b = list(train_data_generator.class_indices.keys())[list(train_data_generator.class_indices.values()).index(1)]
            print_plots(acc, val_acc, loss, val_loss, y_test, pred, label_a, label_b, param_path, fold_var)
            gc.collect()
        print("Test Accuracy:  {:.4f} (+/- {:.4f})".format(np.mean(TEST_ACCURACY), np.std(TEST_ACCURACY)))
        print("Test Loss:      {:.4f}  (+/- {:.4f})".format(np.mean(TEST_LOSS), np.std(TEST_LOSS)))
        print("Test Precision: {:.4f} (+/- {:.4f})".format(np.mean(TEST_PRECISION), np.std(TEST_PRECISION)))
        print("Test Recall:    {:.4f} (+/- {:.4f})".format(np.mean(TEST_RECALL), np.std(TEST_RECALL)))
        print("Test F1:        {:.4f} (+/- {:.4f})".format(np.mean(TEST_F1), np.std(TEST_F1)))
        print("Test Time:      {:.3f}  (+/- {:.3f})".format(np.mean(AVR_TIME), np.std(AVR_TIME)))
        with open(param_path + 'GENERAL_report.txt', 'w') as f:
            with redirect_stdout(f):
                print("Test Accuracy:  {:.4f} (+/- {:.4f})".format(np.mean(TEST_ACCURACY), np.std(TEST_ACCURACY)))
                print("Test Loss:      {:.4f}  (+/- {:.4f})".format(np.mean(TEST_LOSS), np.std(TEST_LOSS)))
                print("Test Precision: {:.4f} (+/- {:.4f})".format(np.mean(TEST_PRECISION), np.std(TEST_PRECISION)))
                print("Test Recall:    {:.4f} (+/- {:.4f})".format(np.mean(TEST_RECALL), np.std(TEST_RECALL)))
                print("Test F1:        {:.4f} (+/- {:.4f})".format(np.mean(TEST_F1), np.std(TEST_F1)))
                print("Test Time:      {:.3f}  (+/- {:.3f})".format(np.mean(AVR_TIME), np.std(AVR_TIME)))
                print("\n")
                print("tf.keras.callbacks.ReduceLROnPlateau(patience={}, min_lr={})".format(pat, minlr))
        tf.keras.backend.clear_session()

    return


def run_model_4_3_kfold_all_regulations(datestamp, name):
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import MaxPooling2D, Dense, Flatten, Conv2D, BatchNormalization, Dropout
    from sklearn.metrics import classification_report, recall_score, precision_score, f1_score

    root_results = 'C:/Dane/Wyniki/KFold_Unbalanced/'
    root_data = r'C:/Dane/unbalanced_kfold_train/'
    test_data = r'C:/Dane/unbalanced_kfold_test'
    train_data = pd.read_csv('C:/Users/karol/Documents/STUDIA/Untitled Folder/mgrmgr/unbalanced_train_labels.csv')

    model_name = name + '/'
    results_path = root_results + model_name
    path = os.path.join(results_path, datestamp)
    os.makedirs(path)
    results_path = str(path) + '/'

    def get_model(size):
        tf.keras.backend.clear_session()
        lr = 0.0001  # 0.001 (default)
        opt = Adam(learning_rate=lr)
        model = Sequential()
        model.add(Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), activation="relu", padding="valid", input_shape=(size[0], size[1], 3)))
        model.add(BatchNormalization())
        model.add(Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), activation="relu", padding="valid"))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), activation="relu", padding="valid"))
        model.add(BatchNormalization())
        model.add(Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), activation="relu", padding="valid"))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), activation="relu", padding="valid"))
        model.add(BatchNormalization())
        model.add(Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), activation="relu", padding="valid"))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dense(64, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.3))
        model.add(Dense(32, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dense(output_units, activation=activation_fun))

        model.compile(optimizer=opt, loss=loss_fun, metrics=["accuracy"])

        with open(results_path + 'summary.txt', 'w') as f:
            with redirect_stdout(f):
                model.summary()
        return model

    Y = train_data[['label']]
    skf = StratifiedKFold(n_splits=5, random_state=7, shuffle=True)
    idg = ImageDataGenerator(rescale=(1.0 / 255.0))
    test_data_generator = idg.flow_from_directory(directory=test_data, target_size=set_size,
                                                  class_mode=class_mode, color_mode=mode, batch_size=batch_size, shuffle=False)

    idg_train = ImageDataGenerator(rescale=(1.0/255.0), rotation_range=150, brightness_range=[0.3, 1.5], zoom_range=0.4)


    TEST_ACCURACY = []
    TEST_LOSS = []
    TEST_PRECISION = []
    TEST_RECALL = []
    TEST_F1 = []
    AVR_TIME = []

    count = 0
    for root_dir, cur_dir, files in os.walk(root_data):
        count += len(files)
    print('file count:', count)

    fold_var = 0

    for train_index, val_index in skf.split(np.zeros(count), Y):
        fold_var += 1
        training_data = train_data.iloc[train_index]
        validation_data = train_data.iloc[val_index]

        train_data_generator = idg_train.flow_from_dataframe(training_data,
                                                       x_col="filename", y_col="label", batch_size=batch_size,
                                                       class_mode="binary", target_size=set_size, shuffle=True)
        valid_data_generator = idg.flow_from_dataframe(validation_data,
                                                       x_col="filename", y_col="label", batch_size=batch_size,
                                                       class_mode="binary", target_size=set_size, shuffle=False)

        unique = np.unique(train_data_generator.classes, return_counts=True)
        labels_dict = dict(zip(unique[0], unique[1]))
        weights_for_unbalanced = {0: float(labels_dict[1] / labels_dict[0]), 1: 1.0}
        print("Wagi klas: {}".format(weights_for_unbalanced))

        model = get_model(set_size)

        print("\n--------------------------    Fitting on fold {}    --------------------------\n".format(fold_var))
        start = time.time()
        history = model.fit(train_data_generator, epochs=epochs, callbacks=my_callbacks,
                            validation_data=valid_data_generator, verbose=2, class_weight=weights_for_unbalanced)
        end = time.time()
        elapsed = end - start

        results = model.evaluate(test_data_generator, verbose=2)
        pred = model.predict(test_data_generator)
        pred = np.round(pred).tolist()  # FOR BINARY
        y_test = test_data_generator.labels  # set y_test to the expected output
        recall = recall_score(y_true=y_test, y_pred=pred, average='binary')
        precision = precision_score(y_true=y_test, y_pred=pred, average='binary')
        f1 = f1_score(y_true=y_test, y_pred=pred, average='binary')

        with open(results_path + 'classification_report_' + str(fold_var) + '.txt', 'w') as f:
            with redirect_stdout(f):
                print(classification_report(y_test, pred, digits=4))
                print("\n\n")
                print("Trening trwał: {:.3f} sekund".format(elapsed))
                print("\n")
                print("Test Loss: {:.4f}".format(results[0]))

        results = dict(zip(model.metrics_names, results))
        TEST_ACCURACY.append(results['accuracy'])
        TEST_LOSS.append(results['loss'])
        TEST_PRECISION.append(precision)
        TEST_RECALL.append(recall)
        TEST_F1.append(f1)
        AVR_TIME.append(elapsed)

        acc = history.history["accuracy"]  # report of model
        val_acc = history.history["val_accuracy"]  # history of validation data
        loss = history.history["loss"]  # Training loss
        val_loss = history.history["val_loss"]  # validation loss

        label_a = list(train_data_generator.class_indices.keys())[list(train_data_generator.class_indices.values()).index(0)]
        label_b = list(train_data_generator.class_indices.keys())[list(train_data_generator.class_indices.values()).index(1)]

        print_plots(acc, val_acc, loss, val_loss, y_test, pred, label_a, label_b, results_path, fold_var)

        gc.collect()

    print("Test Accuracy:  {:.4f} (+/- {:.4f})".format(np.mean(TEST_ACCURACY), np.std(TEST_ACCURACY)))
    print("Test Loss:      {:.4f}  (+/- {:.4f})".format(np.mean(TEST_LOSS), np.std(TEST_LOSS)))
    print("Test Precision: {:.4f} (+/- {:.4f})".format(np.mean(TEST_PRECISION), np.std(TEST_PRECISION)))
    print("Test Recall:    {:.4f} (+/- {:.4f})".format(np.mean(TEST_RECALL), np.std(TEST_RECALL)))
    print("Test F1:        {:.4f} (+/- {:.4f})".format(np.mean(TEST_F1), np.std(TEST_F1)))
    print("Test Time:      {:.3f}  (+/- {:.3f})".format(np.mean(AVR_TIME), np.std(AVR_TIME)))

    with open(results_path + 'GENERAL_report.txt', 'w') as f:
        with redirect_stdout(f):
            print("Test Accuracy:  {:.4f} (+/- {:.4f})".format(np.mean(TEST_ACCURACY), np.std(TEST_ACCURACY)))
            print("Test Loss:      {:.4f}  (+/- {:.4f})".format(np.mean(TEST_LOSS), np.std(TEST_LOSS)))
            print("Test Precision: {:.4f} (+/- {:.4f})".format(np.mean(TEST_PRECISION), np.std(TEST_PRECISION)))
            print("Test Recall:    {:.4f} (+/- {:.4f})".format(np.mean(TEST_RECALL), np.std(TEST_RECALL)))
            print("Test F1:        {:.4f} (+/- {:.4f})".format(np.mean(TEST_F1), np.std(TEST_F1)))
            print("Test Time:      {:.3f}  (+/- {:.3f})".format(np.mean(AVR_TIME), np.std(AVR_TIME)))
            print("\n")
            print("tf.keras.callbacks.ReduceLROnPlateau(patience={}, min_lr={})".format(pat, minlr))

    tf.keras.backend.clear_session()

    return


# ------------------------------------------------------------------------------------------------------------------------

def transfer_VGG19(datestamp, name):
    from tensorflow.keras.applications import VGG19
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Flatten, BatchNormalization, Dropout
    from sklearn.metrics import classification_report, recall_score, precision_score, f1_score

    #epochs = 11

    root_results = 'C:/Dane/Wyniki/KFold_Unbalanced/'
    root_data = r'C:/Dane/unbalanced_kfold_train/'
    test_data = r'C:/Dane/unbalanced_kfold_test'
    train_data = pd.read_csv('C:/Users/karol/Documents/STUDIA/Untitled Folder/mgrmgr/unbalanced_train_labels.csv')

    model_name = name + '/'
    results_path = root_results + model_name
    path = os.path.join(results_path, datestamp)
    os.makedirs(path)
    results_path = str(path) + '/'

    def get_model(size):
        tf.keras.backend.clear_session()
        lr = 0.001  # 0.001 (default)
        opt = Adam(learning_rate=lr)

        model_base = VGG19(include_top=False, weights='imagenet', pooling='max')
        model_base.trainable = False
        for layer in model_base.layers:
            layer.trainable = False

        model = Sequential()
        model.add(model_base)
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dense(128, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dense(output_units, activation=activation_fun))

        model.compile(optimizer=opt, loss=loss_fun, metrics=["accuracy"])

        with open(results_path + 'summary.txt', 'w') as f:
            with redirect_stdout(f):
                model.summary()
        return model

    Y = train_data[['label']]
    skf = StratifiedKFold(n_splits=5, random_state=7, shuffle=True)
    idg = ImageDataGenerator(rescale=(1.0 / 255.0))
    test_data_generator = idg.flow_from_directory(directory=test_data, target_size=set_size,
                                                  class_mode=class_mode, color_mode=mode, batch_size=batch_size, shuffle=False)

    idg_train = ImageDataGenerator(rescale=(1.0/255.0), rotation_range=150, brightness_range=[0.3, 1.5], zoom_range=0.4)


    TEST_ACCURACY = []
    TEST_LOSS = []
    TEST_PRECISION = []
    TEST_RECALL = []
    TEST_F1 = []
    AVR_TIME = []

    count = 0
    for root_dir, cur_dir, files in os.walk(root_data):
        count += len(files)
    print('file count:', count)

    fold_var = 0

    for train_index, val_index in skf.split(np.zeros(count), Y):
        fold_var += 1
        training_data = train_data.iloc[train_index]
        validation_data = train_data.iloc[val_index]

        train_data_generator = idg_train.flow_from_dataframe(training_data,
                                                       x_col="filename", y_col="label", batch_size=batch_size,
                                                       class_mode="binary", target_size=set_size, shuffle=True)
        valid_data_generator = idg.flow_from_dataframe(validation_data,
                                                       x_col="filename", y_col="label", batch_size=batch_size,
                                                       class_mode="binary", target_size=set_size, shuffle=False)

        unique = np.unique(train_data_generator.classes, return_counts=True)
        labels_dict = dict(zip(unique[0], unique[1]))
        weights_for_unbalanced = {0: float(labels_dict[1] / labels_dict[0]), 1: 1.0}
        print("Wagi klas: {}".format(weights_for_unbalanced))

        model = get_model(set_size)

        print("\n--------------------------    Fitting on fold {}    --------------------------\n".format(fold_var))
        start = time.time()
        history = model.fit(train_data_generator, epochs=epochs, callbacks=my_callbacks,
                            validation_data=valid_data_generator, verbose=2, class_weight=weights_for_unbalanced)
        end = time.time()
        elapsed = end - start

        results = model.evaluate(test_data_generator, verbose=2)
        pred = model.predict(test_data_generator)
        pred = np.round(pred).tolist()  # FOR BINARY
        y_test = test_data_generator.labels  # set y_test to the expected output
        recall = recall_score(y_true=y_test, y_pred=pred, average='binary')
        precision = precision_score(y_true=y_test, y_pred=pred, average='binary')
        f1 = f1_score(y_true=y_test, y_pred=pred, average='binary')

        with open(results_path + 'classification_report_' + str(fold_var) + '.txt', 'w') as f:
            with redirect_stdout(f):
                print(classification_report(y_test, pred, digits=4))
                print("\n\n")
                print("Trening trwał: {:.3f} sekund".format(elapsed))
                print("\n")
                print("Test Loss: {:.4f}".format(results[0]))

        results = dict(zip(model.metrics_names, results))
        TEST_ACCURACY.append(results['accuracy'])
        TEST_LOSS.append(results['loss'])
        TEST_PRECISION.append(precision)
        TEST_RECALL.append(recall)
        TEST_F1.append(f1)
        AVR_TIME.append(elapsed)

        acc = history.history["accuracy"]  # report of model
        val_acc = history.history["val_accuracy"]  # history of validation data
        loss = history.history["loss"]  # Training loss
        val_loss = history.history["val_loss"]  # validation loss

        label_a = list(train_data_generator.class_indices.keys())[list(train_data_generator.class_indices.values()).index(0)]
        label_b = list(train_data_generator.class_indices.keys())[list(train_data_generator.class_indices.values()).index(1)]

        print_plots(acc, val_acc, loss, val_loss, y_test, pred, label_a, label_b, results_path, fold_var)

        gc.collect()

    print("Test Accuracy:  {:.4f} (+/- {:.4f})".format(np.mean(TEST_ACCURACY), np.std(TEST_ACCURACY)))
    print("Test Loss:      {:.4f}  (+/- {:.4f})".format(np.mean(TEST_LOSS), np.std(TEST_LOSS)))
    print("Test Precision: {:.4f} (+/- {:.4f})".format(np.mean(TEST_PRECISION), np.std(TEST_PRECISION)))
    print("Test Recall:    {:.4f} (+/- {:.4f})".format(np.mean(TEST_RECALL), np.std(TEST_RECALL)))
    print("Test F1:        {:.4f} (+/- {:.4f})".format(np.mean(TEST_F1), np.std(TEST_F1)))
    print("Test Time:      {:.3f}  (+/- {:.3f})".format(np.mean(AVR_TIME), np.std(AVR_TIME)))

    with open(results_path + 'GENERAL_report.txt', 'w') as f:
        with redirect_stdout(f):
            print("Test Accuracy:  {:.4f} (+/- {:.4f})".format(np.mean(TEST_ACCURACY), np.std(TEST_ACCURACY)))
            print("Test Loss:      {:.4f}  (+/- {:.4f})".format(np.mean(TEST_LOSS), np.std(TEST_LOSS)))
            print("Test Precision: {:.4f} (+/- {:.4f})".format(np.mean(TEST_PRECISION), np.std(TEST_PRECISION)))
            print("Test Recall:    {:.4f} (+/- {:.4f})".format(np.mean(TEST_RECALL), np.std(TEST_RECALL)))
            print("Test F1:        {:.4f} (+/- {:.4f})".format(np.mean(TEST_F1), np.std(TEST_F1)))
            print("Test Time:      {:.3f}  (+/- {:.3f})".format(np.mean(AVR_TIME), np.std(AVR_TIME)))
            print("\n")
            print("tf.keras.callbacks.ReduceLROnPlateau(patience={}, min_lr={})".format(pat, minlr))

    tf.keras.backend.clear_session()

    return




if __name__ == "__main__":

    print('Running on local GPU' if tf.config.list_physical_devices('GPU') else '\t\u2022 GPU device not found. Running on local CPU')

    now = datetime.now()  # current date and time
    date_time = now.strftime("%d-%m_%H-%M")

    '''

    list_of_models_names = ["MODEL_1_2_kfold", "MODEL_2_2_kfold", "MODEL_3_2_kfold",
                            "MODEL_4_1_kfold", "MODEL_4_2_kfold", "MODEL_4_3_kfold", "MODEL_4_4_kfold",
                            "MODEL_5_1_kfold", "MODEL_5_2_kfold", "MODEL_5_3_kfold", "MODEL_5_4_kfold",
                            "MODEL_6_1_kfold",
                            ]

    list_of_functions = [run_model_1_2_kfold, run_model_2_2_kfold, run_model_3_2_kfold,
                         run_model_4_1_kfold, run_model_4_2_kfold, run_model_4_3_kfold, run_model_4_4_kfold,
                         run_model_5_1_kfold, run_model_5_2_kfold, run_model_5_3_kfold, run_model_5_4_kfold,
                         run_model_6_1_kfold]
    '''

    '''
    list_of_models_names = ["MODEL_3_2_kfold_batchnorm", "MODEL_3_2_kfold_dropout", "MODEL_3_2_kfold_dropout2D", "MODEL_3_2_kfold_L2",
                            "MODEL_3_2_kfold_all_regulations", ]

    list_of_functions = [run_model_3_2_kfold_batchnorm, run_model_3_2_kfold_dropout, run_model_3_2_kfold_dropout2D, run_model_3_2_kfold_L2,
                         run_model_3_2_kfold_all_regulations]

    '''

    '''
    list_of_models_names = ["MODEL_3_2_kfold_all_regulations"]

    list_of_functions = [run_model_3_2_kfold_all_regulations]
    '''

    '''
    list_of_models_names = ["MODEL_4_3_kfold_batchnorm", "MODEL_4_3_kfold_dropout", "MODEL_4_3_kfold_dropout2D", "MODEL_4_3_kfold_L2",
                            "MODEL_4_3_kfold_all_regulations" ]

    list_of_functions = [run_model_4_3_kfold_batchnorm, run_model_4_3_kfold_dropout, run_model_4_3_kfold_dropout2D, run_model_4_3_kfold_L2,
                         run_model_4_3_kfold_all_regulations]
    '''

    '''
    list_of_models_names = ["transfer_MobileNetV2", "transfer_ResNet50V2", "transfer_InceptionV3", "transfer_VGG19"]

    list_of_functions = [transfer_MobileNetV2, transfer_ResNet50V2, transfer_InceptionV3, transfer_VGG19]
    '''

    #list_of_models_names = ["MODEL_3_2_kfold_wagi", "MODEL_4_3_kfold_wagi", "MODEL_4_3_kfold_dropout_wagi", "MODEL_4_3_kfold_all_regulations_wagi", "transfer_VGG19_wagi"]

    #list_of_functions = [run_model_3_2_kfold, run_model_4_3_kfold, run_model_4_3_kfold_dropout, run_model_4_3_kfold_all_regulations, transfer_VGG19]

    list_of_models_names = ["MODEL_3_2_kfold_augment_wagi", "MODEL_4_3_kfold_augment_wagi", "MODEL_4_3_kfold_dropout_augment_wagi", "MODEL_4_3_kfold_all_regulations_augment_wagi", "transfer_VGG19_augment_wagi"]

    list_of_functions = [run_model_3_2_kfold, run_model_4_3_kfold, run_model_4_3_kfold_dropout, run_model_4_3_kfold_all_regulations, transfer_VGG19]


    for name, func in zip(list_of_models_names, list_of_functions):
        print("_____________________________________________________________________________________________________________________________")
        print("----------------------------------------------------    {}    -------------------------------------------".format(name))
        print("_____________________________________________________________________________________________________________________________")

        proc = multiprocessing.Process(target=func, args=(date_time, name))
        proc.start()
        proc.join()
        clear_session()






