import sys
import numpy as np
from cifar import load_CIFAR10
cifar10_dir = './cifar10/cifar-10-batches-py'
X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)
import lasagne
from theano import tensor as T
from lasagne.nonlinearities import *
from memory_profiler import profile
input_X = T.tensor4("X")
target_y = T.vector("target Y integer",dtype='int32')
from net import build_cnn
def layer(net, num_filters):
    net = lasagne.layers.Conv2DLayer(net, num_filters, 3, pad="same")
    net = lasagne.layers.BatchNormLayer(net)
    net = lasagne.layers.Conv2DLayer(net, num_filters, 3, pad="same", nonlinearity=lasagne.nonlinearities.rectify)
    net = lasagne.layers.MaxPool2DLayer(net, 2)
    return net

net = lasagne.layers.InputLayer(shape=(None, 3, 32, 32), input_var=input_X)
net = layer(net, 256)
net = layer(net, 256)
net = layer(net, 1024)
net = layer(net, 1024)
net = layer(net, 1024)
net = lasagne.layers.dense.DenseLayer(net, 2048, nonlinearity=lasagne.nonlinearities.rectify)
net = lasagne.layers.DropoutLayer(net, 0.2)
net = lasagne.layers.dense.DenseLayer(net, 512, nonlinearity=lasagne.nonlinearities.rectify)
net = lasagne.layers.DropoutLayer(net, 0.2)
net = lasagne.layers.dense.DenseLayer(net, 256, nonlinearity=lasagne.nonlinearities.rectify)
net = lasagne.layers.DropoutLayer(net, 0.2)

net = lasagne.layers.DenseLayer(net, num_units = 10, nonlinearity=softmax)


def main():
    net = build_cnn(input_X, 5)
    try:
        with np.load('model.npz') as f:
            param_values = [f['arr_%d' % i] for i in range(len(f.files))]
            lasagne.layers.set_all_param_values(net, param_values)
    except:
        pass

    y_predicted = lasagne.layers.get_output(net)
    all_weights = lasagne.layers.get_all_params(net, trainable=True)
    loss = lasagne.objectives.categorical_crossentropy(y_predicted, target_y).mean() + \
           1e-4*lasagne.regularization.regularize_network_params(net, lasagne.regularization.l2)
    accuracy = lasagne.objectives.categorical_accuracy(y_predicted, target_y).mean()
    learning_rate = theano.shared(lasagne.utils.floatX(1e-1))
    adam = lasagne.updates.momentum(loss, all_weights, learning_rate=learning_rate)
    train_fun = theano.function([input_X,target_y],[loss, accuracy], updates=adam, allow_input_downcast=True)
    accuracy_fun = theano.function([input_X,target_y], accuracy, allow_input_downcast=True)


    from keras.preprocessing.image import ImageDataGenerator
    generator = ImageDataGenerator(
        featurewise_center=True,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True)
    test_generator = ImageDataGenerator(
        featurewise_center=True)
    batch_size = 128
    generator.fit(X_train)
    test_generator.fit(X_train)
    train_flow = generator.flow(X_train, y_train, batch_size=batch_size)
    test_flow = test_generator.flow(X_test, y_test, batch_size=batch_size)

    import time

    num_epochs = 500 #количество проходов по данным

    try:
        for epoch in np.arange(num_epochs):
            # In each epoch, we do a full pass over the training data:
            train_err = 0
            train_acc = 0
            train_batches = 0
            start_time = time.time()
            for batch in train_flow:
                inputs, targets = batch
                train_err_batch, train_acc_batch = train_fun(inputs, targets)
                train_err += train_err_batch
                train_acc += train_acc_batch
                train_batches += 1
                if train_batches % 10 == 0:
                    print(train_batches // 10, end=" ")
                    sys.stdout.flush()
                if train_batches % 100 == 0:
                    print()
                if train_batches * batch_size >= 50000:
                    break
            print()

            print("Epoch {} of {} took {:.3f}s".format(epoch + 1, num_epochs, time.time() - start_time))
            print("  training loss (in-iteration):\t\t{:.6f}".format(train_err / train_batches))
            print("  train accuracy:\t\t{:.2f} %".format(train_acc / train_batches * 100))
            np.savez('checkpoint-{}-{:.2f}.npz'.format(epoch, train_acc / train_batches * 100), *lasagne.layers.get_all_param_values(net))
            if epoch % 5 == 0:
                val_acc = 0
                val_batches = 0
                for batch in test_flow:
                    inputs, targets = batch
                    val_acc += accuracy_fun(inputs, targets)
                    val_batches += 1
                    if val_batches % 10 == 0:
                        print(val_batches // 10, end=" ")
                        sys.stdout.flush()
                    if val_batches % 100 == 0:
                        print()
                    if val_batches * batch_size >= 10000:
                        break
                print()
                print("validation accuracy:\t\t{:.2f} %".format(val_acc / val_batches * 100))
            if epoch == 40 or epoch == 60:
                new_lr = learning_rate.get_value() * 0.1
                print("New LR:"+str(new_lr))
                learning_rate.set_value(lasagne.utils.floatX(new_lr))

    except KeyboardInterrupt:
        pass
    np.savez('model.npz', *lasagne.layers.get_all_param_values(net))
    val_acc = 0
    val_batches = 0
    for batch in test_flow:
        inputs, targets = batch
        val_acc += accuracy_fun(inputs, targets)
        val_batches += 1
        if val_batches % 10 == 0:
            print(val_batches // 10, end=" ")
            sys.stdout.flush()
        if val_batches % 100 == 0:
            print()
        if val_batches * batch_size >= 10000:
            break
    print("  validation accuracy:\t\t{:.2f} %".format(val_acc / val_batches * 100))
main()
