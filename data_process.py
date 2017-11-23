import numpy as np

def process_train_data(frame):
  # convert to np array
  array = np.array(frame)

  # split feature and label
  features, labels = array[:, 1:], array[:, 0]

  # spit data 80:20
  from sklearn.model_selection import train_test_split
  return train_test_split(features, labels, test_size=0.2)
