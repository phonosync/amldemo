import numpy as np
import argparse
import os
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential, model_from_json
from keras.layers import Dense
from keras.optimizers import RMSprop
from keras.callbacks import Callback
import tensorflow as tf

from azureml.core import Run
from mnist_util import load_data

parser = argparse.ArgumentParser()
parser.add_argument('--data-folder', type=str, dest='data_folder', help='data folder mounting point')
parser.add_argument('--output-folder', type=str, dest='output_folder', help='output folder')
parser.add_argument('--batch-size', type=int, dest='batch_size', default=50,
                    help='mini batch size for training')
parser.add_argument('--first-layer-neurons', type=int, dest='n_hidden_1', default=100,
                    help='# of neurons in the first layer')
parser.add_argument('--second-layer-neurons', type=int, dest='n_hidden_2', default=100,
                    help='# of neurons in the second layer')
parser.add_argument('--learning-rate', type=float, dest='learning_rate', default=0.001, help='learning rate')


def train(data_folder, output_folder, batch_size, n_hidden_1, n_hidden_2, learning_rate):
    # load train and test set into numpy arrays
    # note we scale the pixel intensity values to 0-1 (by dividing it with 255.0) so the model can converge
    # faster.
    X_train = load_data(os.path.join(data_folder, 'train-images.gz'), False) / 255.0
    X_test = load_data(os.path.join(data_folder, 'test-images.gz'), False) / 255.0
    y_train = load_data(os.path.join(data_folder, 'train-labels.gz'), True).reshape(-1)
    y_test = load_data(os.path.join(data_folder, 'test-labels.gz'), True).reshape(-1)
    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape, sep='\n')

    # training_set_size = X_train.shape[0]

    n_inputs = 28 * 28
    n_outputs = 10
    n_epochs = 20
    batch_size = batch_size
    learning_rate = learning_rate

    y_train = np.eye(n_outputs)[y_train.reshape(-1)]
    y_test = np.eye(n_outputs)[y_test.reshape(-1)]
    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape, sep='\n')

    # Build a simple MLP model
    model = Sequential()
    # first hidden layer
    model.add(Dense(n_hidden_1, activation='relu', input_shape=(n_inputs,)))
    # second hidden layer
    model.add(Dense(n_hidden_2, activation='relu'))
    # output layer
    model.add(Dense(n_outputs, activation='softmax'))

    model.summary()

    model.compile(loss='categorical_crossentropy',
                  optimizer=RMSprop(lr=learning_rate),
                  metrics=['accuracy'])

    # start an Azure ML run
    run = Run.get_context()

    class LogRunMetrics(Callback):
        # callback at the end of every epoch
        def on_epoch_end(self, epoch, log):
            # log a value repeated which creates a list
            run.log('Loss', log['loss'])
            run.log('Accuracy', log['acc'])

    history = model.fit(X_train, y_train,
                        batch_size=batch_size,
                        epochs=n_epochs,
                        verbose=2,
                        validation_data=(X_test, y_test),
                        callbacks=[LogRunMetrics()])

    score = model.evaluate(X_test, y_test, verbose=0)

    # log a single value
    run.log("Final test loss", score[0])
    print('Test loss:', score[0])

    run.log('Final test accuracy', score[1])
    print('Test accuracy:', score[1])

    plt.figure(figsize=(6, 3))
    plt.title('MNIST with Keras MLP ({} epochs)'.format(n_epochs), fontsize=14)
    plt.plot(history.history['acc'], 'b-', label='Accuracy', lw=4, alpha=0.5)
    plt.plot(history.history['loss'], 'r--', label='Loss', lw=4, alpha=0.5)
    plt.legend(fontsize=12)
    plt.grid(True)

    # log an image
    run.log_image('Accuracy vs Loss', plot=plt)

    # create a ./outputs/model folder in the compute target
    # files saved in the "./outputs" folder are automatically uploaded into run history
    folder_model = os.path.join(output_folder, 'model')
    os.makedirs(folder_model, exist_ok=True)

    # serialize NN architecture to JSON
    model_json = model.to_json()
    # save model JSON
    with open(os.path.join(folder_model, 'model.json'), 'w') as f:
        f.write(model_json)
    # save model weights
    model.save_weights(os.path.join(folder_model, 'model.h5'))
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

    train(args.data_folder, args.output_folder, args.batch_size, args.n_hidden_1, args.n_hidden_2,
          args.learning_rate)
