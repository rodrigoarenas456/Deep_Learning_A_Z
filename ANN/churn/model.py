from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


def ann_model():
    model = Sequential()
    model.add(Dense(units=6, activation='relu'))
    model.add(Dense(units=6, activation='relu'))
    model.add(Dense(units=1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics='accuracy')

    return model
