import re

import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM


def window_transform_series(series, window_size):
    """
    Transform the input series and window-size into a set
    of input/output pairs for use with our RNN model.

    Args:
      series: The input data series.
      window_size: The window size.
    Returns:
      The input/output pairs.
    """
    # container for inputs
    X = []

    # loop over input start indexes
    for i in range(len(series) - window_size):
        # add sliced window to inputs container
        X.append(series[i:i + window_size])

    # outputs are the next values after the window
    y = series[window_size:]

    # reshape each
    X = np.asarray(X)
    X = X.reshape(np.shape(X)[:2])
    y = np.asarray(y)
    y = y.reshape((len(y), 1))

    return X, y


def build_part1_RNN(step_size, window_size):
    """
    Build an RNN to perform regression on our time series input/output data.

    Args:
      step_size: Number of steps per input record.
      window_size: Size of the window on the data.
    Returns:
      The RNN model.
    """
    # Create the model.
    model = Sequential()
    # Add a LSTM layer with 5 hidden units.
    model.add(LSTM(5, input_shape=(window_size, step_size)))
    # Add a fully-connected layer with 1 hidden unit.
    model.add(Dense(1))
    return model


def clean_text(text):
    """
    List all unique characters in the text and remove any non-english ones.

    Args:
      text: The input text.
    Returns:
      The cleaned text.
    """
    # Use a regular expression to replace non-english letters with a space.
    text = re.sub(r'[^a-z.,\-\'\"]+', ' ', text, flags=re.IGNORECASE)
    return text


def window_transform_text(text, window_size, step_size):
    """
    Transform the input text and window-size into a set of
    input/output pairs for use with our RNN model.

    Args:
      text: The input text.
      window_size: The window size.
      step_size: The step size.
    Returns:
      The input/output pairs.
    """
    # container for inputs
    inputs = []

    # loop over input start indexes
    for i in range(0, len(text) - window_size, step_size):
        # add sliced window to inputs container
        inputs.append(text[i:i + window_size])

    # outputs are the next values after the window
    outputs = text[window_size::step_size]

    return inputs, outputs
