import tensorflow as tf
import os
import pandas as pd
import matplotlib.pyplot as plt
from preprocessing import RNNModel

dataset_test = pd.read_csv('./datasets/Google_Stock_Price_Test.csv')
real_stock_price = dataset_test.iloc[:, 1:2].values

rnn = RNNModel()
X_train, y_train = rnn.get_training_data()
if os.path.isfile("LSTM.h5"):
    model = tf.keras.models.load_model("LSTM.h5")
else:
    model = rnn.model()
    model.fit(X_train, y_train, epochs=80, batch_size=32)
    model.save('LSTM.h5')

X_test = rnn.get_test_data()
predicted_stock_price = model.predict(X_test)
predicted_stock_price = rnn.scaler.inverse_transform(predicted_stock_price)
print(predicted_stock_price)

plt.plot(real_stock_price, color='red', label='Real Google Stock Price')
plt.plot(predicted_stock_price, color='blue', label='Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()
