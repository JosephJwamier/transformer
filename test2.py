import tensorflow as tf
import pandas as pd 
from tensorflow.keras import layers
import tensorflow.keras as kr
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
#from sklearn.preprocessing import MinMaxScaler
#from sklearn.model_selection import train_test_split
from keras.layers import Bidirectional
from keras.layers import Dropout

data = pd.read_csv('EURUSD60.csv')
bigger = data["open"].max()


#here we scale the data between 0 to 1
opendata = data["open"]
opendatas = opendata/bigger
opendatas.shape 


