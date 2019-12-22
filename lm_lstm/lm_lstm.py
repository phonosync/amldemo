import argparse
import os
import pickle
import string
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from keras import callbacks
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM, Embedding
from keras.preprocessing.text import Tokenizer
from azureml.core import Run


parser = argparse.ArgumentParser()
parser.add_argument('--data-folder', type=str, dest='data_folder', help='data folder mounting point')
parser.add_argument('--fn-sequences', type=str, dest='fn_sequences', help='text sequences file name')
parser.add_argument('--output-folder', type=str, dest='output_folder', help='output folder')
parser.add_argument('--batch-size', type=int, dest='batch_size', default=50,
                    help='mini batch size for training')
parser.add_argument('--n-epochs', type=int, dest='n_epochs', default=200,
                    help='number of training epochs')


# turn a doc into clean tokens
def clean_doc(doc):
    # replace '--' with a space ' '
    doc = doc.replace('--', ' ')
    # split into tokens by white space
    tokens = doc.split()
    # remove punctuation from each token
    table = str.maketrans('', '', string.punctuation)
    tokens = [w.translate(table) for w in tokens]
    # remove remaining tokens that are not alphabetic
    tokens = [word for word in tokens if word.isalpha()]
    # make lower case
    tokens = [word.lower() for word in tokens]
    return tokens


def train(data_folder, fn_sequences, batch_size, n_epochs, output_folder):

    with open(os.path.join(data_folder, fn_sequences), 'r') as f:
        seqs = f.read().split('\n')

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(seqs)
    sequences = tokenizer.texts_to_sequences(seqs)

    vocab_size = len(tokenizer.word_index) + 1
    print(vocab_size)

    from keras.utils import to_categorical
    sequences = np.array(sequences)
    X, y = sequences[:, :-1], sequences[:, -1]
    y = keras.utils.to_categorical(y, num_classes=vocab_size)
    seq_length = X.shape[1]

    # define model
    model = Sequential()
    model.add(Embedding(vocab_size, 50, input_length=seq_length))
    model.add(LSTM(100, return_sequences=True))
    model.add(LSTM(100))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(vocab_size, activation='softmax'))

    print(model.summary())

    model_checkpoint = callbacks.ModelCheckpoint("my_checkpoint.h5", save_best_only=True)
    early_stopping = callbacks.EarlyStopping(patience=50)
    # compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # fit model

    # start an Azure ML run
    run = Run.get_context()

    class LogRunMetrics(callbacks.Callback):
        # callback at the end of every epoch
        def on_epoch_end(self, epoch, log):
            # log a value repeated which creates a list
            run.log('Loss', log['loss'])

    # model.fit(X, y, batch_size=256, epochs=200, callbacks=[early_stopping, model_checkpoint])
    history = model.fit(X, y,
                        batch_size=batch_size,
                        epochs=n_epochs,
                        verbose=2,
                        callbacks=[LogRunMetrics(), early_stopping, model_checkpoint])

    # log a single value

    plt.figure(figsize=(6, 3))
    plt.title('Goethe LM with Keras MLP ({} epochs)'.format(n_epochs), fontsize=14)
    plt.plot(history.history['loss'], 'r--', label='Loss', lw=4, alpha=0.5)
    plt.legend(fontsize=12)
    plt.grid(True)

    # log an image
    run.log_image('Loss', plot=plt)

    # create a ./outputs/model folder in the compute target
    # files saved in the "./outputs" folder are automatically uploaded into run history
    folder_model = os.path.join(output_folder, 'model_goethe')
    os.makedirs(folder_model, exist_ok=True)

    # save the tokenizer
    pickle.dump(tokenizer, open(os.path.join(folder_model, 'tokenizer.pkl', 'wb')))

    # save the model to file
    # model.save('model_goethe_generator.h5')
    # serialize NN architecture to JSON
    model_json = model.to_json()
    # save model JSON
    with open(os.path.join(folder_model, 'model.json'), 'w') as f:
        f.write(model_json)
    # save model weights
    model.save_weights(os.path.join(folder_model, 'model_weights.h5'))
    print("model saved in folder " + folder_model)
    return model


if __name__ == '__main__':
    print("Keras version:", keras.__version__)
    print("Tensorflow version:", tf.__version__)

    if tf.test.is_gpu_available(cuda_only=True, min_cuda_compute_capability=None):
        print('CUDA GPU available')
    else:
        print('no CUDA GPU available')
    if tf.test.is_built_with_cuda():
        print('TF was built with GPU support')
    else:
        print('TF was not built with GPU support')

    args = parser.parse_args()

    train(data_folder=args.data_folder, fn_sequences=args.fn_sequences, batch_size=args.batch_size,
          n_epochs=args.n_epochs, output_folder=args.output_folder)
