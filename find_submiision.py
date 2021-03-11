import pandas as pd
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.regularizers import l2
from keras.optimizers import SGD, Adam
from keras.losses import BinaryCrossentropy
from plot_history import plot_history, loss_frame
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split

data_train = pd.read_csv('./dont-overfit-ii/train.csv')
data_submission = pd.read_csv('./dont-overfit-ii/test.csv')

label = data_train['target'].to_numpy()
feature = data_train.drop(['target', 'id'], axis=1).to_numpy()
x_submission = data_submission.drop('id', axis=1)
id_submission = data_submission['id'].to_numpy()

x_train, x_test, y_train, y_test = train_test_split(scale(feature), label, shuffle=True, test_size=0.1)
rows, cols = x_train.shape

model = Sequential()
model.add(Dense(cols, input_dim=cols, kernel_regularizer=l2(0.01), activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(1, activation='sigmoid', kernel_regularizer=l2(0.01)))
model.compile(optimizer=Adam(lr=0.001), loss='binary_crossentropy', metrics=['accuracy'])
his = model.fit(x_train, y_train, epochs=30, validation_split=0.1)
loss_test, accuracy_test = model.evaluate(x_test, y_test)
print('test accuracy:', accuracy_test)
plot_history(his)
loss_frame(loss_test, his)
