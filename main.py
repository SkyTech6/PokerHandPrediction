import pandas
import warnings
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import regularizers
import matplotlib.pyplot as plot

warnings.filterwarnings('ignore')

poker_train = pandas.read_csv('poker-hand-training-true.data', header=None)
poker_test = pandas.read_csv('poker-hand-testing.data', header=None)
col = ['Suit of card #1', 'Rank of card #1', 'Suit of card #2', 'Rank of card #2', 'Suit of card #3', 'Rank of card #3',
       'Suit of card #4', 'Rank of card #4', 'Suit of card #5', 'Rank of card 5', 'Poker Hand']

poker_train.columns = col
poker_test.columns = col

hand_train = poker_train['Poker Hand']
hand_test = poker_test['Poker Hand']

train = pandas.get_dummies(hand_train)
test = pandas.get_dummies(hand_test)

drop_train = poker_train.drop('Poker Hand', axis=1)
drop_test = poker_test.drop('Poker Hand', axis=1)

print('Training Set: ', drop_train.shape)
print('Testing Set: ', drop_test.shape)

# Build Neural Network with TensorFlow's Keras support
model = keras.Sequential(
    [
        layers.Dense(5 * 16, activation='relu', input_dim=10, name='dense'),
        layers.Dropout(0.5),
        layers.Dense(5 * 16, activation='relu', name='dense-drop'),
        layers.Dropout(0.2),
        # layers.Dense(5 * 8, activation='relu', name='dense-deep'),
        # layers.Dropout(0.2),
        layers.Dense(10, activation='softmax', name='output')

        # layers.Dense(15, activation="relu", input_dim=10),
        # layers.Dense(10, activation="relu"),
        # layers.Dense(10, activation="softmax")
    ]
)

# loss_function = keras.losses.SparseCategoricalCrossentropy()
loss_function = keras.losses.binary_crossentropy
opt = SGD(lr=0.01, momentum=0.9)
adam = keras.optimizers.Adam()
model.compile(loss=loss_function, optimizer=adam, metrics=['accuracy'])

history = model.fit(drop_train, train, epochs=1000, batch_size=256, verbose=1,
                    validation_data=(drop_test, test), shuffle=True)


if __name__ == '__main__':
    score = model.evaluate(drop_test, test, batch_size=256)
    model.summary()
    # plot loss during training
    plot.subplot(211)
    plot.plot(history.history['loss'])
    plot.plot(history.history['val_loss'])
    plot.title('model loss')
    plot.ylabel('loss')
    plot.xlabel('epoc')
    plot.legend(['train', 'test'], loc='upper left')
    plot.show()
    # plot accuracy during training
    plot.subplot(212)
    plot.title('Accuracy')
    plot.plot(history.history['accuracy'])
    plot.plot(history.history['val_accuracy'])
    plot.title('model accuracy')
    plot.ylabel('accuracy')
    plot.legend(['train', 'test'], loc='upper left')
    plot.show()

