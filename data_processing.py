from merge_harth_data import read_file, save_file
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mode

from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from keras.models import Sequential, Model
from keras.layers import Dense, GlobalAveragePooling1D, BatchNormalization, MaxPool1D, Reshape, Activation
from keras.layers import Conv1D, LSTM, Conv2D
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns
from scipy.fft import fft

import warnings
warnings.filterwarnings("ignore")

activity_id = {1: "walking", 2: "running", 3: "shuffling", 4: "stairs_ascending", #original label
            5: "stairs_descending", 6: "standing", 7: "sitting", 8: "lying",      #original label
            13: "cycling_sit", 14: "cycling_stand", 130: "cycling_sit_inactive",  #original label
            140: "cycling_stand_inactive"}                                        #original label

def plot_all(df):
    plt.figure(figsize=(15, 5))

    plt.xlabel('Activity Type')
    plt.ylabel('Training examples')
    df['label'].value_counts().plot(kind='bar',
                                    title='Training examples by Activity Types')

    plt.figure(figsize=(15, 5))
    plt.xlabel('User')
    plt.ylabel('Training examples')
    df['user'].value_counts().plot(kind='bar', 
                                    title='Training examples by user')
    plt.show()

def axis_plot(ax, x, y, title):
    ax.plot(x, y, 'r')
    ax.set_title(title)
    ax.xaxis.set_visible(False)
    ax.set_ylim([min(y) - np.std(y), max(y) + np.std(y)])
    ax.set_xlim([min(x), max(x)])
    ax.grid(True)

def split_data(df):
    df_test = df[df['user'] > 14]
    df_train = df[df['user'] <= 14]
    return df_train, df_test

def data_normalization(df):
    for feature in ['back_x', 'thigh_x', 'back_y', 'thigh_y', 'back_z', 'thigh_z']:
        df[feature] = (df[feature]-df[feature].min())/(df[feature].max()-df[feature].min())
    return df

def fix_encoding(df):
    count_users = 0
    df = df.drop(columns=['Unnamed: 9'], inplace=False)
    for user in df['user'].unique():
        df['user'] = df['user'].replace(user, count_users)
        count_users += 1
    save_file('encoded_data', df)
    return df

def plot_signal_simp(df, num_points = 180, var = ['back_x', 'thigh_x', 'back_y', 'thigh_y', 'back_z', 'thigh_z']):
    x = np.arange(num_points)
    for activity in df['label'].unique():
        limit = df[df['label'] == activity][:num_points]
        fig, axs = plt.subplots(nrows=3, ncols=2, sharex=True, figsize=(15, 10))
        axs = axs.flatten()
        for i, feature in enumerate(var):
            axis_plot(axs[i], x, limit[feature], feature) # x = limit['index']
        plt.subplots_adjust(hspace=0.2)
        fig.suptitle(activity_id[activity], fontsize=20)

        plt.subplots_adjust(top=0.9)
        plt.show()

def segments(df, time_steps, step, label_name):
    N_FEATURES = 6
    segments = []
    labels = []
    for i in range(0, len(df) - time_steps, step):
        b_x = df['back_x'].values[i:i+time_steps]
        b_y = df['back_y'].values[i:i+time_steps]
        b_z = df['back_z'].values[i:i+time_steps]
        t_x = df['thigh_x'].values[i:i+time_steps]
        t_y = df['thigh_y'].values[i:i+time_steps]
        t_z = df['thigh_z'].values[i:i+time_steps]
        
        label = mode(df[label_name][i:i+time_steps], keepdims=True)[0][0]
        segments.append([b_x, b_y, b_z, t_x, t_y, t_z])
        labels.append(label)
    
    reshaped_segments = np.asarray(segments, dtype=np.float32).reshape(-1, time_steps, N_FEATURES)
    labels = np.asarray(labels)
    
    return reshaped_segments, labels

def get_activity(df, activity):
    return df[df['label'] == activity]

def correlation(ax, df, activity, func_name = ''):
    df_min = df[['back_x' + func_name, 'thigh_x', 'back_y' + func_name, 
                 'thigh_y', 'back_z' + func_name, 'thigh_z']]  
    corr_matrix = df_min.corr() 
    sns.heatmap(np.round(corr_matrix, 2), annot=True, square=True, ax=ax, cmap="PuOr", vmin=-1, vmax=1)
    ax.set_title('Correlation matrix ' + activity_id[activity])

def correlation_(ax, df, activity, func_name = ''):
    df_min = df[['back_x', 'thigh_x', 'back_y', 'thigh_y', 'back_z', 'thigh_z', func_name + '_back', func_name + '_thigh']]    
    corr_matrix = df_min.corr() 
    sns.heatmap(np.round(corr_matrix, 2), annot=True, square=True, ax=ax, cmap="PuOr", vmin=-1, vmax=1)
    ax.set_title('Correlation matrix ' + activity_id[activity])

def batch_correlation(df, func_name = ''):
    unique_activities = df['label'].unique()
    
    # Define the layout of the subplots (e.g., 3x4 grid)
    rows = 3
    cols = 4

    fig, axes = plt.subplots(rows, cols, figsize=(20, 15))
    axes = axes.flatten()  # Flatten to easily iterate over
    
    for idx, activity in enumerate(unique_activities):
        df_act = get_activity(df, activity)
        correlation_(axes[idx], df_act, activity, func_name)
        #correlation(axes[idx], df_act, activity, func_name)
    
    plt.tight_layout()
    # plt.savefig('archive/results/all_correlation_matrices' + func_name + '.pdf')
    plt.show()

def funtion_generator(df): 
    # funcs = {'_sin': np.sin, '_cos': np.cos, '_tan': np.tan, '_exp': np.exp, '_log': np.log, '_abs': np.abs,
    #          '_sqrt': np.sqrt, '_square': np.square, '_cube': np.cbrt, '_reciprocal': np.reciprocal, 
    #          '_sinh': np.sinh, '_cosh': np.cosh, '_tanh': np.tanh, '_expm1': np.expm1, '_log1p': np.log1p,
    #          '_log10': np.log10, '_log2': np.log2, '_arccos': np.arccos, '_arcsin': np.arcsin, '_arctan': np.arctan}
    funcs = {'_fft': lambda x: fft(x)}

    for func_name, func in funcs.items():
        for feature in ['back_x', 'back_y', 'back_z']:
            # df[f'{feature}{func_name}'] = df[feature].apply(func)
            df[f'{feature}{func_name}'] = func(df[feature].values)
        batch_correlation(df, func_name)

def function_generator_complex(df):
    funcs = {'pitag'}
    for func_name in funcs:
        df = add_function(df, func_name)
        batch_correlation(df, func_name)
    return df

def add_function(df, func_name):
    if func_name == 'pitag':
        func = lambda x, y, z: np.sqrt(x**2 + y**2 + z**2)
    df[f'{func_name}{'_back'}'] = func(df['back_x'], df['back_y'], df['back_z'])
    df[f'{func_name}{'_thigh'}'] = func(df['thigh_x'], df['thigh_y'], df['thigh_z'])
    return df

# preprocessing dataset
# df = read_file('merged_data.csv')
# plot_all(df)
# plot_signal_simp(df)
# fix_encoding(df)

label_encode = LabelEncoder()

df = read_file('encoded_data.csv')
# plot_all(df)
df['activityEncode'] = label_encode.fit_transform(df['label'].values.ravel())
# df = data_normalization(df)
# plot_signal_simp(df)
df_train, df_test = split_data(df)
# batch_correlation(df)
# funtion_generator(df)
df = function_generator_complex(df)
plot_signal_simp(df, 500, ['pitag_back', 'pitag_thigh'])

TIME_PERIOD = 16
STEP_DISTANCE = 8
LABEL = 'activityEncode'
x_train, y_train = segments(df_train, TIME_PERIOD, STEP_DISTANCE, LABEL)
# print('x_train shape:', x_train.shape)
# print('Training samples:', x_train.shape[0])
# print('y_train shape:', y_train.shape)

time_period, sensors = x_train.shape[1], x_train.shape[2]
num_classes = label_encode.classes_.size

input_shape = time_period * sensors
x_train = x_train.reshape(x_train.shape[0], input_shape)
print("Input Shape: ", input_shape)
print("Input Data Shape: ", x_train.shape)

x_train = x_train.astype('float32')
y_train = y_train.astype('float32')

#One hot encoding 
y_train_hot = to_categorical(y_train, num_classes)
print("y_train shape: ", y_train_hot.shape)

#Model definition
model = Sequential()    
model.add(LSTM(32, return_sequences=True, input_shape=(input_shape, 1), activation='relu'))
model.add(LSTM(32,return_sequences=True, activation='relu'))
model.add(Reshape((1, input_shape, 32)))  # Reshape to (480, 32) to match Conv1D expected input
model.add(Conv2D(filters=64, kernel_size=(1, 2), activation='relu', strides=(1, 2)))
# model.add(Conv1D(filters=64, kernel_size=2, activation='relu', strides=2))
model.add(Reshape((int(TIME_PERIOD*sensors/2), 64)))
model.add(MaxPool1D(pool_size=4, padding='same'))
model.add(Conv1D(filters=192, kernel_size=2, activation='relu', strides=1))
model.add(Reshape((int((TIME_PERIOD + STEP_DISTANCE)/2 - 1), 192)))
model.add(GlobalAveragePooling1D())
model.add(BatchNormalization(epsilon=1e-06))
model.add(Dense(6*2))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

history = model.fit(x_train,
                    y_train_hot, 
                    batch_size= 192, 
                    epochs=100
                   )

plt.figure(figsize=(6, 4))
plt.plot(history.history['accuracy'], 'r', label='Accuracy of training data')
plt.plot(history.history['loss'], 'r--', label='Loss of training data')
plt.title('Model Accuracy and Loss')
plt.ylabel('Accuracy and Loss')
plt.xlabel('Training Epoch')
plt.ylim(0)
plt.legend()
plt.show()

y_pred_train = model.predict(x_train)
max_y_pred_train = np.argmax(y_pred_train, axis=1)
print(classification_report(y_train, max_y_pred_train))

#Test on the test dataset
x_test, y_test = segments(df_test,
                         TIME_PERIOD,
                         STEP_DISTANCE,
                         LABEL)

x_test = x_test.reshape(x_test.shape[0], input_shape)
x_test = x_test.astype('float32')
y_test = y_test.astype('float32')
y_test = to_categorical(y_test, num_classes)

score = model.evaluate(x_test, y_test)
print("Accuracy:", score[1])
print("Loss:", score[0])

#Confussion Matrix

predictions = model.predict(x_test)
predictions = np.argmax(predictions, axis=1)
y_test_pred = np.argmax(y_test, axis=1)
cm = confusion_matrix(y_test_pred, predictions)
cm_disp = ConfusionMatrixDisplay(confusion_matrix= cm)
cm_disp.plot()
plt.show()

#Classification Report
print(classification_report(y_test_pred, predictions))