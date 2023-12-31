import numpy as np
import tensorflow as tf
import datetime
import zipfile
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import os 
import matplotlib.pyplot as plt
import random
import shutil

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

  log_dir = dir_name + "/" + experiment_name + "/" + datetime.datetime.now().strftime("%Y%m%D-%H%M%S")
  
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



def split_and_move_data(source_dir, target_dir, split_ratio=0.75, subset_fraction=1.0):
    """
    Split and move a fraction of files from subdirectories in the source directory into training and testing sets.

    Parameters:
        source_dir (str): The path to the source directory containing subdirectories with data.
        target_dir (str): The path to the target directory where training and testing data will be organized.
        split_ratio (float, optional): The ratio of files to be placed in the training set. Default is 0.75 (75%).
        subset_fraction (float, optional): The fraction of data to process. Default is 1.0 (process all data).

    Returns:
        None

    Example:
        source_directory = 'path_to_source_directory'
        target_directory = 'path_to_target_directory'
        split_ratio = 0.75  # 75% for training, 25% for testing
        subset_fraction = 0.5  # Process only 50% of the data
        split_and_move_data(source_directory, target_directory, split_ratio, subset_fraction)
    """
    # List all subdirectories in the source directory
    subdirectories = [d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))]

    for subdir in subdirectories:
        subdir_path = os.path.join(source_dir, subdir)
        files = os.listdir(subdir_path)
        
        # Calculate the number of files to move to the training set, considering the subset_fraction
        num_files = len(files)
        num_to_process = int(num_files * subset_fraction)
        num_train = int(num_to_process * split_ratio)
        
        # Randomly shuffle the list of files
        random.shuffle(files)
        
        # Split and move the files
        for i, file in enumerate(files[:num_to_process]):
            source_file = os.path.join(subdir_path, file)
            if i < num_train:
                target_file = os.path.join(target_dir, 'training', subdir, file)
            else:
                target_file = os.path.join(target_dir, 'testing', subdir, file)

            os.makedirs(os.path.dirname(target_file), exist_ok=True)
            shutil.copy(source_file, target_file)

def get_random_examples(train_dataset):
  class_names = train_dataset.class_names

  plt.figure(figsize=(10, 10))
  for images, labels in train_dataset.take(1):
    for i in range(9):
      ax = plt.subplot(3, 3, i + 1)
      plt.imshow(images[i].numpy().astype("uint8"))
      plt.title(class_names[labels[i]])
      plt.axis("off")

# function to compare histories
def plot_fine_tuning_epochs(original_history, new_history, initial_epochs):
  """
  Plots the Fine tuning epochs against the fine tuning epochs
  """
  acc = original_history.history["accuracy"]
  loss = original_history.history["loss"]

  val_acc = original_history.history["val_accuracy"]
  val_loss = original_history.history["val_loss"]

  total_acc = acc + new_history.history["accuracy"]
  total_loss = loss + new_history.history["loss"]
  
  total_val_acc = val_acc + new_history.history["val_accuracy"]
  total_val_loss = val_loss + new_history.history["val_loss"]

  #Plot for accuracy
  plt.figure(figsize=(8,8))
  plt.subplot(2,1,1)
  plt.plot(total_acc, label="Training Accuracy")
  plt.plot(total_val_acc, label="Val Accuracy")
  plt.plot([initial_epochs-1, initial_epochs-1], plt.ylim(), label="Start fine tuning")
  plt.legend(loc="lower right")
  plt.title("Training and Validation Accuracy")
  
  # Plot for loss
  plt.figure(figsize=(8,8))
  plt.subplot(2,1,2)
  plt.plot(total_loss, label="Training Loss")
  plt.plot(total_val_loss, label="Val Loss")
  plt.plot([initial_epochs-1, initial_epochs-1], plt.ylim(), label="Start fine tuning")
  plt.legend(loc="lower right")
  plt.title("Training and Validation Loss")

def unfreeze_layers(base_model , num):
  """
  Unfreezes specified number of layers for fine-tuning
  """
  base_model.trainable = True

  for layer in base_model.layers[:-num]:
    layer.trainable = False
  return base_model

def predict_images(test_data, model):
   # plot 4 random images
  x, y = train_data.next()

  fig, ax = plt.subplots(2, 2)

  # 'fig' is the whole figure, and 'ax' is an array of axes objects
  fig.suptitle("Multiple Examples")

  ax[0, 0].axis(False)
  ax[0, 1].axis(False)
  ax[1, 0].axis(False)
  ax[1, 1].axis(False)

  # You can access individual subplots using indexing
  ax[0, 0].imshow(x[0])
  ax[0, 1].imshow(x[1])
  ax[1, 0].imshow(x[2])
  ax[1, 1].imshow(x[3])

  # make predicitons
  p = [False] * 4
  for i in range(4):
    if model.predict(x[i]) == y[i]:
       p[i] = "g"
    else:
       p[i] = "r"
    p[i] = model.predict(x[i]) == y[i]

  # Customize each subplot as needed
  ax[0, 0].set_title(label_names[np.argmax(y[0])], color=p[0])
  ax[0, 1].set_title(label_names[np.argmax(y[1])], color=p[1])
  ax[1, 0].set_title(label_names[np.argmax(y[2])], color=p[2])
  ax[1, 1].set_title(label_names[np.argmax(y[3])], color=p[3])

  plt.show()