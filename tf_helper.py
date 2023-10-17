import numpy as np
import tensorflow as tf
import datetime
import zipfile
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import os 
import matplotlib.pyplot as plt

# Create TensorBoard callback

def create_tensorboard_callback(dir_name, experiment_name):
  """
  Creates Tensorflow log files in specified location
  Arguments:
  dir_name (str): directory where logs are saved
  experiment_name: folder name of logs
  Returns:
  An Tensorboard callback which can be passed into the model.fit function.
  as a 'callbacks' parameter
  """

  log_dir = dir_name + "/" + experiment_name + "/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
  
  tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
  print(f"Saving TensorBoard log files to: {log_dir}")

  return tensorboard_callback

def unzip(location):
    """
    Shortcut to extract zipfiles
    Arguments:
    location (str): file which is to be extracted
    Returns:
    void
    """

    zip_ref = zipfile.ZipFile(location)
    zip_ref.extractall()
    zip_ref.close()

def walk_through_dataset(directory):
    """
    Looks through dataset and prints amount of data in different folders.
    Arguments:
    directory (str): directory which is to be analyzed
    Retuns:
    void
    """

    for dirpath, dirnames, filenames in os.walk(directory):
        print(f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'.")

def plot_confusion_matrix(model, x_test, y_test):
    """
    Plots a confusion matrix for a multiclass classification matrix.
    Arguments:
    model (tensorflowModel): The model which is to be analyzed
    x_test (tensor): Test data,
    y_test (tensor): Test labels
    Returns:
    Confusion Matrix
    """

    cm = confusion_matrix(y_test, np.argmax(model.predict(x_test),axis=1))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.show()

def plot_loss_curves(history):
  """
  Returns separate loss curves for training and validation metrics.
  """ 
  loss = history.history['loss']
  val_loss = history.history['val_loss']

  accuracy = history.history['accuracy']
  val_accuracy = history.history['val_accuracy']

  epochs = range(len(history.history['loss']))

  # Plot loss
  plt.plot(epochs, loss, label='training_loss')
  plt.plot(epochs, val_loss, label='val_loss')
  plt.title('Loss')
  plt.xlabel('Epochs')
  plt.legend()

  # Plot accuracy
  plt.figure()
  plt.plot(epochs, accuracy, label='training_accuracy')
  plt.plot(epochs, val_accuracy, label='val_accuracy')
  plt.title('Accuracy')
  plt.xlabel('Epochs')
  plt.legend()