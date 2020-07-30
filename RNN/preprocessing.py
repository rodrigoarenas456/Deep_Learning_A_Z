import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout


class RNNModel:

    def __init__(self):
        self.scaler = None
        self.lstm = None

    def get_training_data(self):
        self.training_dataset = pd.read_csv('./datasets/Google_Stock_Price_Train.csv')
        training_set = self.training_dataset.iloc[:, 1:2].values
        self.scaler = MinMaxScaler()
        training_set = self.scaler.fit_transform(training_set)
        self.X_train = []
        y_train = []
        for i in range(60, 1258):
            self.X_train.append(training_set[i - 60:i, 0])
            y_train.append(training_set[i, 0])
        self.X_train, y_train = np.array(self.X_train), np.array(y_train)
        self.X_train = np.reshape(self.X_train, newshape=(self.X_train.shape[0], self.X_train.shape[1], 1))
        return self.X_train, y_train

    def get_test_data(self):
        """
        WARNING: In real life scenarios, known data from test set should be passed to make predictions
        it should make 1 step prediction and take this as input to the next one (along the others 59 observations)
        """
        test_dataset = pd.read_csv('./datasets/Google_Stock_Price_Test.csv')
        dataset_total = pd.concat((self.training_dataset['Open'], test_dataset['Open']), axis=0)
        inputs = dataset_total[len(dataset_total) - len(test_dataset) - 60:].values
        inputs = inputs.reshape(-1, 1)
        inputs = self.scaler.transform(inputs)
        self.X_test = []
        for i in range(60, 80):
            self.X_test.append(inputs[i - 60:i, 0])
        self.X_test = np.array(self.X_test)
        self.X_test = np.reshape(self.X_test, (self.X_test.shape[0], self.X_test.shape[1], 1))
        return self.X_test

    def model(self):
        self.lstm = Sequential()
        self.lstm.add(LSTM(units=50, return_sequences=True, input_shape=(self.X_train.shape[1], 1)))
        self.lstm.add(Dropout(0.25))
        self.lstm.add(LSTM(units=50, return_sequences=True))
        self.lstm.add(Dropout(0.25))
        self.lstm.add(LSTM(units=50, return_sequences=True))
        self.lstm.add(Dropout(0.25))
        self.lstm.add(LSTM(units=50))
        self.lstm.add(Dropout(0.25))
        self.lstm.add(Dense(units=1))
        self.lstm.compile(optimizer='adam', loss='mean_squared_error')
        return self.lstm
