import numpy as np

# TensorFlow ≥2.0 is required
import tensorflow as tf
from tensorflow import keras
assert tf.__version__ >= "2.0"



#####
# Give two arrays, calculate the mean squared error for their last element 
# this is used in calculating the loss(error) in the neural network training
#####
def last_time_step_mse(Y_true, Y_pred):
    return keras.metrics.mean_squared_error(Y_true[:, -1], Y_pred[:, -1])


#####
# Given a pandas object and the key name of one of its elements
# return numpy arrays of n_steps elements to suitable for training of a neural net
#####
def get_neural_net_training_sets(pandas_object, key, n_steps):
    # extract the desired values and covert them to a numpy array
    np_data = pandas_object[key].to_numpy()
    
    # truncate the values to the desired length
    np_data = np_data[:n_steps+1]

    # put the array in to the required shape expected by the neual net algoritm
    np_series = np.array([np_data])

    # Cast the values to float32 (required for neural net processing)
    np_series = np_series[..., np.newaxis].astype(np.float32)
    
    # Split these data in to the training set and the labels
    # the labels in this case is simple the last data point in the time series
    np_training_data = np.array(np_series[:,:-1])
    np_labels = np.array(np_series[:,-1])    
    
    return np_training_data, np_labels


#####
# Given a pandas object and the key name of one of its elements
# return two numpy arrays for making forcasts (predictions) in a trained neural network
#####
def get_neural_net_forecast_sets(pandas_object, key, n_steps_training, n_steps_forecast, n_steps_ahead):
    # extract the desired values and covert them to a numpy array
    np_data = pandas_object[key].to_numpy()

    # Get the tailing records in the data set,
    # skipping the leading values up to the 
    # index n_steps_forecast + n_steps_ahead
    np_data = np_data[-(n_steps_forecast + n_steps_ahead):]

    # put the array in to the required shape expected by the neual net algoritm
    forecast_series = np.array([np_data])

    # Cast the values to float32 (required for neural net processing)
    forecast_series = forecast_series[..., np.newaxis].astype(np.float32)
    
    # Split these data in to the training set and the labels
    # the labels in this case is simple the last data point in the time series
    np_training_data = forecast_series[:, :n_steps_forecast]
    np_labels = forecast_series[:, n_steps_forecast:]
    
    return np_training_data, np_labels


#####
# return a keras object containing the neural network model to be used 
#
# Note: In development of the notebook and support libraries, a number of different neural new models were investigated.
#       While only the final model used in the notebook is actually used, the configuration for the other models are 
#       retained in case the are of use in future development
#####
def get_neural_net_model(n_steps_ahead):
    # Set seeds for random number generators so output is consistent across runs
    np.random.seed(42)
    tf.random.set_seed(42)
        
    model = get_time_distributed_1D_convolutional_and_gru_with_dropout_model(n_steps_ahead)
    
    return model



##### Main one employed in the Notebook
# This keras model defines a neural net with several layers:
# Top     : A 1D convolutional to assist in dimensionality reduction of the data
# Top - 1 : A GRU layer (Gated Recurrent Unit) which acts similar to a recurrent neural net but is not as susecptible to 
#          disappearing/expoding gradients
# Top - 2 : A dropout layer to reduce this model's tendency to overfit with limited input data
# Top - 3 : Another GRU layer
# Top - 4 : A time distibuted dense layer set to output an *n_steps_ahead* number of values
#####
def get_time_distributed_1D_convolutional_and_gru_with_dropout_model(n_steps_ahead):

    model = keras.models.Sequential([
        keras.layers.Conv1D(filters=20, kernel_size=4, strides=2, padding="valid",
                            input_shape=[None, 1]),
        keras.layers.GRU(20, return_sequences=True),
        keras.layers.Dropout(rate=0.2),
        keras.layers.GRU(20, return_sequences=True),
        keras.layers.TimeDistributed(keras.layers.Dense(n_steps_ahead))
    ])

    model.compile(loss="mse", optimizer=keras.optimizers.Adam(learning_rate=0.01), metrics=[last_time_step_mse])
    
    return model







#####
# Below are other neural net models that were investigated during development
#####

def get_time_distributed_rnn_with_batch_normalization_model(n_steps_ahead):

    model = keras.models.Sequential([
        keras.layers.SimpleRNN(20, return_sequences=True, input_shape=[None, 1]),
        keras.layers.BatchNormalization(),
        keras.layers.SimpleRNN(20, return_sequences=True),
        keras.layers.BatchNormalization(),
        keras.layers.TimeDistributed(keras.layers.Dense(n_steps_ahead))
    ])    
    
    model.compile(loss="mse", optimizer="adam", metrics=[last_time_step_mse])
        
    return model


def get_time_distributed_rnn_with_mse_loss_and_custom_learning_rate_model(n_steps_ahead):
    # This model makes fairly good predictions
    
    model = keras.models.Sequential([
        keras.layers.SimpleRNN(20, return_sequences=True, input_shape=[None, 1]),
        keras.layers.SimpleRNN(20, return_sequences=True),
        keras.layers.TimeDistributed(keras.layers.Dense(n_steps_ahead))
    ])

    model.compile(loss="mse", optimizer=keras.optimizers.Adam(learning_rate=0.01), metrics=[last_time_step_mse])

    return model

    
def get_rnn_model(n_steps_ahead):

    model = keras.models.Sequential([
        keras.layers.SimpleRNN(20, return_sequences=True, input_shape=[None, 1]),
        keras.layers.SimpleRNN(20),
        keras.layers.Dense(n_steps_ahead)
    ])
    
    model.compile(loss="mse", optimizer="adam")
    
    return model


def get_dense_rnn_with_mse_loss_and_custom_learning_rate_model(n_steps_ahead):

    model = keras.models.Sequential([
        keras.layers.SimpleRNN(20, return_sequences=True, input_shape=[None, 1]),
        keras.layers.SimpleRNN(20),
        keras.layers.Dense(n_steps_ahead)
    ])
    
    model.compile(loss="mse", optimizer=keras.optimizers.Adam(learning_rate=0.01), metrics=[last_time_step_mse])
    
    return model


def get_time_distributed_rnn_with_dropout_and_custom_learning_rate_model(n_steps_ahead):

    model = keras.models.Sequential([
        keras.layers.SimpleRNN(20, return_sequences=True, input_shape=[None, 1]),
        keras.layers.Dropout(rate=0.2),
        keras.layers.SimpleRNN(20, return_sequences=True),
        keras.layers.TimeDistributed(keras.layers.Dense(n_steps_ahead))
    ])

    model.compile(loss="mse", optimizer=keras.optimizers.Adam(learning_rate=0.01), metrics=[last_time_step_mse])
    
    return model


def get_time_distributed_LSTM_with_custom_learning_rate_model(n_steps_ahead):
    model = keras.models.Sequential([
        keras.layers.LSTM(20, return_sequences=True, input_shape=[None, 1]),
        keras.layers.LSTM(20, return_sequences=True),
        keras.layers.TimeDistributed(keras.layers.Dense(n_steps_ahead))
    ])

    model.compile(loss="mse", optimizer=keras.optimizers.Adam(learning_rate=0.01), metrics=[last_time_step_mse])
    
    return model
    

def get_time_distributed_LSTM_mse_loss_model(n_steps_ahead):
    model = keras.models.Sequential([
        keras.layers.LSTM(20, return_sequences=True, input_shape=[None, 1]),
        keras.layers.LSTM(20, return_sequences=True),
        keras.layers.TimeDistributed(keras.layers.Dense(n_steps_ahead))
    ])

    model.compile(loss="mse", optimizer="adam", metrics=[last_time_step_mse])
    
    return model



