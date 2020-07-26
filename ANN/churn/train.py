import os
import tensorflow as tf
from sklearn.metrics import confusion_matrix, accuracy_score
from preprocessing import process_churn
from model import ann_model

X_train, X_test, y_train, y_test = process_churn()

if os.path.isfile("./saved_models/churn_ann.h5"):
    model = tf.keras.models.load_model("./saved_models/churn_ann.h5")
else:
    model = ann_model()
    model.fit(X_train, y_train, batch_size=32, epochs=100)
    model.save('./saved_models/churn_ann.h5')

print(model.summary())

y_pred = model.predict(X_test)
y_pred = (y_pred > 0.5)

cm = confusion_matrix(y_test, y_pred)
print(cm)
print(accuracy_score(y_test, y_pred))
