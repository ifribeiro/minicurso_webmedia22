import numpy as np
from keras.models import Sequential
from keras.layers import GRU, LSTM, Dense

def get_residuals(fname, X_expected, n_steps=24):
  fsample = np.load(fname)
  _, y_synth = split_sequence(fsample.flatten(), n_steps)
  resid_sample = X_expected - y_synth[:len(X_expected)]
  return resid_sample

# split a univariate sequence into samples
def split_sequence(sequence, n_steps):
	X, y = list(), list()
	for i in range(len(sequence)):
		# find the end of this pattern
		end_ix = i + n_steps
		# check if we are beyond the sequence
		if end_ix > len(sequence)-1:
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
		X.append(seq_x)
		y.append(seq_y)
	return np.array(X), np.array(y)

def make_rnn_model(units, n_layers=3, n_steps=24, n_features=1, output_units=1, net_type='GRU'):
  """
  Make a Recurrent Neural Network model
  Params:
    - units: rnn units size
    - n_layers: number of layers
    - n_steps: timestep of the data
    - n_features: number of features in data
    - output_units: number of output units imn the RNN
    - net_type: net_type to use (GRU or LSTM)
  
  Returns:
    - model: a RNN model (needs to be compiled)
  """

  model = Sequential()
  if net_type == 'GRU':
    for i in range(n_layers):
      if i == n_layers-1:
        model.add(GRU(units,input_shape=(n_steps,n_features), name="GRU_{}".format(i+1)))
      else:
        model.add(GRU(units,input_shape=(n_steps,n_features), return_sequences=True, name="GRU_{}".format(i+1)))
  else:
    for i in range(n_layers):
      if i == n_layers-1:
        model.add(LSTM(units, input_shape=(n_steps,n_features),name="LSTM_{}".format(i+1)))
      else:
        model.add(LSTM(units, input_shape=(n_steps,n_features), return_sequences=True, name="LSTM_{}".format(i+1)))
  model.add(Dense(units=output_units,name='OUT'))
  return model