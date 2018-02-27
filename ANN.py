from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import Dropout
from keras.layers import Flatten
from keras.optimizers import SGD
from keras.layers.convolutional import MaxPooling2D
from keras.layers.convolutional import Conv2D
import numpy as np
from preprocessing import load_data
from tensorflow.python.client import device_lib
import pandas as pd


X_train, X_val, X_test, y_train, y_val, y_test = load_data()
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_val = X_val.reshape(X_val.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
num_pixels = X_train.shape[1] * X_train.shape[2]
num_classes = y_test.shape[1]
in_shape = (28, 28, 1)


# Model Layers

model = Sequential()
#Input layer
model.add(Conv2D(64, (3, 3), input_shape=in_shape, activation='relu'))

#Hidden Layers
model.add(Conv2D(32, (2, 2), activation='relu'))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(16, activation='relu'))

#Output layer
model.add(Dense(num_classes, activation='softmax'))


# Compile Model
model.compile(optimizer='adam',
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

# Train Model
history = model.fit(X_train, y_train, 
                    validation_data = (X_val, y_val), 
                    epochs=20, 
                    batch_size=200)


# Report Results

print(history.history)


# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Baseline Error: %.2f%%" % (100-scores[1]*100))
print('\n')

# Create Matrix of predictions
y_prediction = model.predict_classes(X_test, batch_size=512)
y_test_matrix = [np.where(r==1)[0][0] for r in y_test]
y_prediction_matrix = [i for i in y_prediction.tolist()]

count = {0: [0, -1], 1: [0, -1], 2: [0, -1], 3: [0, -1], 4: [0, -1], 5: [0, -1], 6: [0, -1], 7: [0, -1], 8: [0, -1], 9: [0, -1]}
correct_preds=0
for i in range(len(y_test)):
  if y_test_matrix[i] == y_prediction_matrix[i]:
    correct_preds+=1
  else:
    count[y_prediction_matrix[i]] = [count[y_prediction_matrix[i]][0]+1, i]

count = sorted(count.items(), key=lambda x: x[1][0], reverse=True)


y_test_matrix = pd.Series(y_test_matrix, name='Actual')
y_prediction_matrix = pd.Series(y_prediction_matrix, name='Predicted')
df_confusion = pd.crosstab(y_test_matrix, y_prediction_matrix, rownames=['Actual'], colnames=['Predicted'], margins=True)
print('\n', df_confusion)

#model.predict(x_train[0])







