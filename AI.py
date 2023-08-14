import cv2, time, os, sys
import numpy as np
import ipywidgets.widgets as widgets
from .__init__ import Camera
from IPython.display import display
import tensorrt as trt
import atexit
import __main__

import warnings
warnings.filterwarnings(action='ignore')

isnotebook=__main__.isnotebook

IDENTIFIER="_models"

mnist_train_x=None
mnist_train_y=None
mnist_test_x=None
mnist_test_y=None

class Linear_Regression:
    learning_rate=1e-2
    print_every=10
    global_step=0

    _X_data=None
    _Y_data=None
    bias=None
    weight=None

    hypothesis=None

    optimizer=None

    ckpt_name="linear_regression" + IDENTIFIER
    restore=None

    _cb_fit=None

    @property
    def X_data(self):
        return self._X_data

    @X_data.setter
    def X_data(self, arg):
        arr=np.array(arg)
        while len(arr.shape) < 2:
            arr=np.expand_dims(arr, -1)

        self._X_data=arr

    @property
    def Y_data(self):
        return self._Y_data

    @Y_data.setter
    def Y_data(self, arg):
        arr=np.array(arg)
        while len(arr.shape) < 2:
            arr=np.expand_dims(arr, -1)

        self._Y_data=arr

    def on_epoch_end(self, epoch, logs):
        if (epoch + 1) % self.print_every == 0:
                print(self.global_step + 1, "step loss:", logs['loss'])
        self.global_step+=1

    def __init__(self, restore=False, ckpt_name=ckpt_name):
        global tf, layers, Sequential, optimizers, callbacks, losses, Adam, load_model, model_from_json, Model, ctivation, BatchNormalization, Concatenate, Conv2D, Dense, Dropout, GlobalAveragePooling2D, Input, Lambda, MaxPooling2D, add, K
        import tensorflow as tf
        from tensorflow.keras import layers, Sequential, optimizers, callbacks, losses
        from tensorflow.keras.optimizers import Adam
        from tensorflow.keras.models import load_model, model_from_json
        from tensorflow.keras.models import Model
        from tensorflow.keras.layers import Activation, BatchNormalization, Concatenate, Conv2D, Dense, Dropout, GlobalAveragePooling2D, Input, Lambda, MaxPooling2D, add
        from tensorflow.keras import backend as K

        self.optimizer=Adam(lr=self.learning_rate)
        self._cb_fit=callbacks.Callback()

        self.ckpt_name = ckpt_name

        self.restore = restore

        self._cb_fit.on_epoch_end=self.on_epoch_end

        self.hypothesis = Sequential()
        
        self.hypothesis.add(layers.Dense(1, input_shape=[1]))

        self.hypothesis.compile(optimizer=self.optimizer, loss='mse')

        if restore:
            if self.load(self.ckpt_name) is None:
                print("Create a new model.")

    def load(self, path=None):
        if path is None : path=self.ckpt_name

        if os.path.isfile(path+".index"):
            ckpt=path
            self.hypothesis.load_weights(ckpt)
            if ".ckpt-" in ckpt :
                self.global_step=int(ckpt.split(".ckpt-")[-1])
            else:
                print("[Warning] Can't read step log.")
                self.global_step=0

        elif os.path.isfile(path):
            exname = path.split(".")[-1]
            if exname == "index" or "data-" in exname :
                ckpt=path[:len(path)-(len(exname)+1)]
                self.hypothesis.load_weights(ckpt)
                if "model.ckpt-" in ckpt :
                    self.global_step=int(ckpt.split("model.ckpt-")[1])
                else:
                    print("[Warning] Can't read step log.")
                    self.global_step=0
            else:
                print("[Error] Can't find a model in '"+path+"'.")
                return

        elif os.path.isdir(path):
            if os.path.isfile(path+"/checkpoint"):
                ckpt=tf.train.latest_checkpoint(path)
                self.hypothesis.load_weights(ckpt)
                if "model.ckpt-" in ckpt :
                    self.global_step=int(ckpt.split("model.ckpt-")[1])
                else:
                    print("[Warning] Can't read step log.")
                    self.global_step=0
            else:
                print("[Error] Doesn't exist a checkpoint file in '"+path+"'.")
                return

        else:
            print("[Error] Can't find a model in '"+path+"'.")
            return

        self.ckpt_name = path
        return "Loaded successfully."

    def save(self, path=None):
        if path is None : path=self.ckpt_name

        if self.hypothesis is not None: 
            self.hypothesis.save_weights(path+"/model.ckpt-"+str(self.global_step))

    def train(self, times=100, print_every=10):
        self.print_every=print_every

        dtime=time.time()
        
        if self.X_data is not None and self.Y_data is not None :
            X_data = tf.keras.preprocessing.sequence.pad_sequences(self.X_data).astype(np.float32)
            Y_data = tf.keras.preprocessing.sequence.pad_sequences(self.Y_data).astype(np.float32)
            self.hypothesis.fit(X_data, Y_data, epochs=times, verbose=0, callbacks=[self._cb_fit], use_multiprocessing=True)
        elif self.X_data is None :
            print("Please input a data to X_data.")
            return
        elif self.Y_data is None :
            print("Please input a data to Y_data.")
            return

        print("Training is done.\nTime spent:", round(time.time()-dtime,1), "s\nTraining speed:", round(times/(time.time()-dtime),1), "step/s")

        self.save()

    def run(self,inputs=None):
        if inputs is None:
            inputs=self.X_data
            print("run with : ",inputs)

        inputs=np.array(inputs, dtype=np.float32)
        while len(inputs.shape) < 2:
            inputs=np.expand_dims(inputs, -1)

        if self.hypothesis is not None and inputs is not None :
            if isnotebook:
                return self.hypothesis.predict(inputs, use_multiprocessing=True)
            else:
                ret = self.hypothesis.predict(inputs, use_multiprocessing=True)
                print(ret)
                return ret


class Logistic_Regression(Linear_Regression):
    ckpt_name="logistic_regression" + IDENTIFIER
    learning_rate=1e-1
    optimizer=None

    def __init__(self, input_size=1, restore=False, ckpt_name=ckpt_name):
        global tf, layers, Sequential, optimizers, callbacks, losses, Adam, load_model, model_from_json, Model, ctivation, BatchNormalization, Concatenate, Conv2D, Dense, Dropout, GlobalAveragePooling2D, Input, Lambda, MaxPooling2D, add, K
        import tensorflow as tf
        from tensorflow.keras import layers, Sequential, optimizers, callbacks, losses
        from tensorflow.keras.optimizers import Adam
        from tensorflow.keras.models import load_model, model_from_json
        from tensorflow.keras.models import Model
        from tensorflow.keras.layers import Activation, BatchNormalization, Concatenate, Conv2D, Dense, Dropout, GlobalAveragePooling2D, Input, Lambda, MaxPooling2D, add
        from tensorflow.keras import backend as K

        self.optimizer=Adam(lr=self.learning_rate)
        self._cb_fit=callbacks.Callback()

        self.ckpt_name = ckpt_name
        self.restore = restore
        self._cb_fit.on_epoch_end=self.on_epoch_end

        self.hypothesis = Sequential()

        self.hypothesis.add(layers.Dense(1, input_shape=[input_size], activation='sigmoid'))

        self.hypothesis.compile(optimizer=self.optimizer, loss='binary_crossentropy')

        if restore:
            if self.load(self.ckpt_name) is None:
                print("Create a new model.")

    def loss_function(self, y_true, y_pred):
        y_true=tf.cast(y_true,tf.float32)
        y_pred=tf.cast(y_pred,tf.float32)
        return -tf.reduce_mean(y_true * tf.math.log(y_pred) + (1 - y_true) * (tf.math.log(1 - y_pred)),-1)


class Perceptron:
    learning_rate=1e-2
    print_every=10
    global_step=0

    softmax=True

    _X_data=None
    _Y_data=None
    bias=None
    weight=None
    
    model=None
    _mnist_loaded=False

    optimizer=None

    ckpt_name="perceptron" + IDENTIFIER
    restore=None

    _cb_fit=None

    @property
    def X_data(self):
        return self._X_data

    @X_data.setter
    def X_data(self, arg):
        arr=np.array(arg)
        while len(arr.shape) < 2:
            arr=np.expand_dims(arr, -1)

        self._X_data=arr

    @property
    def Y_data(self):
        return self._Y_data

    @Y_data.setter
    def Y_data(self, arg):
        arr=np.array(arg)
        while len(arr.shape) < 2:
            arr=np.expand_dims(arr, -1)

        self._Y_data=arr

    def on_epoch_end(self, epoch, logs):
        if (epoch + 1) % self.print_every == 0:
                print(self.global_step + 1, "step loss:", logs['loss'])
        self.global_step+=1

    def __init__(self, input_size, output_size=1, restore=False, ckpt_name=ckpt_name, softmax=True, activation_function=None):
        global tf, layers, Sequential, optimizers, callbacks, losses, Adam, load_model, model_from_json, Model, ctivation, BatchNormalization, Concatenate, Conv2D, Dense, Dropout, GlobalAveragePooling2D, Input, Lambda, MaxPooling2D, add, K
        import tensorflow as tf
        from tensorflow.keras import layers, Sequential, optimizers, callbacks, losses
        from tensorflow.keras.optimizers import Adam
        from tensorflow.keras.models import load_model, model_from_json
        from tensorflow.keras.models import Model
        from tensorflow.keras.layers import Activation, BatchNormalization, Concatenate, Conv2D, Dense, Dropout, GlobalAveragePooling2D, Input, Lambda, MaxPooling2D, add
        from tensorflow.keras import backend as K

        self.optimizer=Adam(lr=self.learning_rate)
        self._cb_fit=callbacks.Callback()

        self.ckpt_name = ckpt_name

        self.restore=restore
        self.softmax=softmax

        self._cb_fit.on_epoch_end=self.on_epoch_end

        self.model = Sequential()

        if self.softmax :
            if output_size > 1 :
                self.model.add(layers.Dense(output_size,input_shape=[input_size], activation='softmax'))

                self.model.compile(optimizer=self.optimizer, loss='categorical_crossentropy')
            else :
                self.model.add(layers.Dense(output_size,input_shape=[input_size], activation='sigmoid'))
                
                self.model.compile(optimizer=self.optimizer, loss='binary_crossentropy')
        else :
            if activation_function == None:
                self.model.add(layers.Dense(output_size,input_shape=[input_size]))
                
                self.model.compile(optimizer=self.optimizer, loss='mse')
            else:
                self.model.add(layers.Dense(output_size,input_shape=[input_size], activation=activation_function))
                
                self.model.compile(optimizer=self.optimizer, loss='mse')

        if restore:
            if self.load(self.ckpt_name) is None:
                print("Create a new model.")

    def load(self, path=None):
        if path is None : path=self.ckpt_name

        if os.path.isfile(path+".index"):
            ckpt=path
            self.model.load_weights(ckpt)
            if ".ckpt-" in ckpt :
                self.global_step=int(ckpt.split(".ckpt-")[-1])
            else:
                print("[Warning] Can't read step log.")
                self.global_step=0

        elif os.path.isfile(path):
            exname = path.split(".")[-1]
            if exname == "index" or "data-" in exname :
                ckpt=path[:len(path)-(len(exname)+1)]
                self.model.load_weights(ckpt)
                if "model.ckpt-" in ckpt :
                    self.global_step=int(ckpt.split("model.ckpt-")[1])
                else:
                    print("[Warning] Can't read step log.")
                    self.global_step=0
            else:
                print("[Error] Can't find a model in '"+path+"'.")
                return

        elif os.path.isdir(path):
            if os.path.isfile(path+"/checkpoint"):
                ckpt=tf.train.latest_checkpoint(path)
                self.model.load_weights(ckpt)
                if "model.ckpt-" in ckpt :
                    self.global_step=int(ckpt.split("model.ckpt-")[1])
                else:
                    print("[Warning] Can't read step log.")
                    self.global_step=0
            else:
                print("[Error] Doesn't exist a checkpoint file in '"+path+"'.")
                return

        else:
            print("[Error] Can't find a model in '"+path+"'.")
            return

        self.ckpt_name = path
        return "Loaded successfully."

    def save(self, path=None):
        if path is None : path=self.ckpt_name
        
        if self.model is not None: 
            self.model.save_weights(path+"/model.ckpt-"+str(self.global_step))

    def train(self, times=100, print_every=10):
        self.print_every=print_every

        dtime=time.time()
        
        if self.X_data is not None and self.Y_data is not None :
            X_data = tf.keras.preprocessing.sequence.pad_sequences(self.X_data).astype(np.float32)
            Y_data = tf.keras.preprocessing.sequence.pad_sequences(self.Y_data).astype(np.float32)
            self.model.fit(X_data, Y_data, epochs=times, verbose=0, callbacks=[self._cb_fit], use_multiprocessing=True)
        elif self.X_data is None :
            print("Please input a data to X_data.")
            return
        elif self.Y_data is None :
            print("Please input a data to Y_data.")
            return

        print("Training is done.\nTime spent:", round(time.time()-dtime,1), "s\nTraining speed:", round(times/(time.time()-dtime),1), "step/s")

        self.save()

    def run(self,inputs=None):
        if inputs is None:
            inputs=self.X_data
            print("run with : ",inputs)

        inputs=np.array(inputs, dtype=np.float32)
        while len(inputs.shape) < 2:
            inputs=np.expand_dims(inputs, -1)

        if self.model is not None and inputs is not None :
            return self.model.predict(inputs, use_multiprocessing=True)


class ANN(Perceptron):
    layer=None
    ckpt_name="ANN" + IDENTIFIER

    def __init__(self, input_size, hidden_size=10, output_size=1, restore=False, ckpt_name=ckpt_name, softmax=True):
        global tf, layers, Sequential, optimizers, callbacks, losses, Adam, load_model, model_from_json, Model, ctivation, BatchNormalization, Concatenate, Conv2D, Dense, Dropout, GlobalAveragePooling2D, Input, Lambda, MaxPooling2D, add, K
        import tensorflow as tf
        from tensorflow.keras import layers, Sequential, optimizers, callbacks, losses
        from tensorflow.keras.optimizers import Adam
        from tensorflow.keras.models import load_model, model_from_json
        from tensorflow.keras.models import Model
        from tensorflow.keras.layers import Activation, BatchNormalization, Concatenate, Conv2D, Dense, Dropout, GlobalAveragePooling2D, Input, Lambda, MaxPooling2D, add
        from tensorflow.keras import backend as K

        self.optimizer=Adam(lr=self.learning_rate)
        self._cb_fit=callbacks.Callback()

        self.ckpt_name = ckpt_name

        self.restore=restore
        self.softmax=softmax

        self._cb_fit.on_epoch_end=self.on_epoch_end

        self.model = Sequential()

        if self.softmax :
            if output_size > 1 :
                self.model.add(layers.Dense(hidden_size,input_shape=[input_size]))
                self.model.add(layers.Dense(output_size, activation='softmax'))

                self.model.compile(optimizer=self.optimizer, loss='categorical_crossentropy')
            else :
                self.model.add(layers.Dense(hidden_size,input_shape=[input_size]))
                self.model.add(layers.Dense(output_size, activation='sigmoid'))
                
                self.model.compile(optimizer=self.optimizer, loss='binary_crossentropy')
        else :
            self.model.add(layers.Dense(hidden_size,input_shape=[input_size]))
            self.model.add(layers.Dense(output_size))
            
            self.model.compile(optimizer=self.optimizer, loss='mse')

        if restore:
            if self.load(self.ckpt_name) is None:
                print("Create a new model.")


class DNN(ANN):
    learning_rate=1e-2
    level=0
    ckpt_name="DNN" + IDENTIFIER
    optimizer=None

    def __init__(self, input_size, hidden_size=10, output_size=1, layer_level=3, restore=False, ckpt_name=ckpt_name, softmax=True):
        global tf, layers, Sequential, optimizers, callbacks, losses, Adam, load_model, model_from_json, Model, ctivation, BatchNormalization, Concatenate, Conv2D, Dense, Dropout, GlobalAveragePooling2D, InputLayer, Lambda, MaxPooling2D, add, K
        import tensorflow as tf
        from tensorflow.keras import layers, Sequential, optimizers, callbacks, losses
        from tensorflow.keras.optimizers import Adam
        from tensorflow.keras.models import load_model, model_from_json
        from tensorflow.keras.models import Model
        from tensorflow.keras.layers import Activation, BatchNormalization, Concatenate, Conv2D, Dense, Dropout, GlobalAveragePooling2D, InputLayer, Lambda, MaxPooling2D, add
        from tensorflow.keras import backend as K

        self.optimizer=Adam(lr=self.learning_rate)
        self._cb_fit=callbacks.Callback()

        self.ckpt_name = ckpt_name

        self.restore=restore
        self.softmax=softmax

        self._cb_fit.on_epoch_end=self.on_epoch_end

        if layer_level < 1:
            print("Please set a layer level at least 1.")
            del self
        else:
            self.model = Sequential()

            self.model.add(layers.InputLayer(input_shape=(input_size,)))
            
            for _ in range(layer_level):
                self.model.add(layers.Dense(hidden_size))

            if self.softmax :
                if output_size > 1 :
                    self.model.add(layers.Dense(output_size, activation='softmax'))

                    self.model.compile(optimizer=self.optimizer, loss='categorical_crossentropy')
                else :
                    self.model.add(layers.Dense(output_size, activation='sigmoid'))
                    
                    self.model.compile(optimizer=self.optimizer, loss='binary_crossentropy')
            else :
                self.model.add(layers.Dense(output_size))
                
                self.model.compile(optimizer=self.optimizer, loss='mse')

            if restore:
                if self.load(self.ckpt_name) is None:
                    print("Create a new model.")


class CNN(DNN):
    ckpt_name="CNN" + IDENTIFIER
    _input_level=1
    mnist_train_x=None
    mnist_train_y=None
    mnist_test_x=None
    mnist_test_y=None
    _X=None
    _Y=None
    _pre_batch_pos=0
    gray=True

    @property
    def X_data(self):
        return self._X_data

    @X_data.setter
    def X_data(self, arg):
        arr=np.array(arg)
        while len(arr.shape) < 3:
            if len(arr.shape) < 2 :
                arr=np.expand_dims(arr, -1)
            else :
                arr=np.expand_dims(arr, 0)

        self._X_data=arr

    @property
    def Y_data(self):
        return self._Y_data

    @Y_data.setter
    def Y_data(self, arg):
        arr=np.array(arg)
        while len(arr.shape) < 2 :
            arr=np.expand_dims(arr, -1)

        self._Y_data=arr

    def __init__(self, input_size=[28,28], input_level=1, kernel_size=[3,3], kernel_count=32, strides=[1,1], hidden_size=128, output_size=1, conv_level=2, layer_level=1, restore=False, ckpt_name=ckpt_name, softmax=True):
        global tf, layers, Sequential, optimizers, callbacks, losses, Adam, load_model, model_from_json, Model, ctivation, BatchNormalization, Concatenate, Conv2D, Dense, Dropout, GlobalAveragePooling2D, InputLayer, Lambda, MaxPooling2D, add, K
        import tensorflow as tf
        from tensorflow.keras import layers, Sequential, optimizers, callbacks, losses
        from tensorflow.keras.optimizers import Adam
        from tensorflow.keras.models import load_model, model_from_json
        from tensorflow.keras.models import Model
        from tensorflow.keras.layers import Activation, BatchNormalization, Concatenate, Conv2D, Dense, Dropout, GlobalAveragePooling2D, InputLayer, Lambda, MaxPooling2D, add
        from tensorflow.keras import backend as K

        self.optimizer=Adam(lr=self.learning_rate)
        self._cb_fit=callbacks.Callback()

        self.ckpt_name = ckpt_name

        self._ipsize=input_size
        self._opsize=output_size
        self.restore=restore
        self.softmax=softmax

        self._cb_fit.on_epoch_end=self.on_epoch_end

        input_shape=[input_size[0], input_size[1], input_level]

        if input_level == 3 :
            self.gray=False
        
        if layer_level < 1 :
            print("Please set a Fully-connected layer level at least 1.")
            del self
        elif conv_level < 1 :
            print("Please set a Convolutional layer level at least 1.")
            del self
        else:
            self.model = Sequential()

            self.model.add(layers.InputLayer(input_shape=input_shape))
            
            for _ in range(conv_level):
                self.model.add(layers.Conv2D(filters=kernel_count, kernel_size=kernel_size, strides=strides, padding='SAME'))
                self.model.add(layers.MaxPool2D([2,2], [2,2], padding='SAME'))
                self.model.add(layers.Dropout(0.5))

            self.model.add(layers.Flatten())

            for _ in range(layer_level):
                self.model.add(layers.Dense(hidden_size))
                self.model.add(layers.Dropout(0.3))

            if self.softmax :
                if output_size > 1 :
                    self.model.add(layers.Dense(output_size, activation='softmax'))

                    self.model.compile(optimizer=self.optimizer, loss='categorical_crossentropy')
                else :
                    self.model.add(layers.Dense(output_size, activation='sigmoid'))
                    
                    self.model.compile(optimizer=self.optimizer, loss='binary_crossentropy')
            else :
                self.model.add(layers.Dense(output_size))
                
                self.model.compile(optimizer=self.optimizer, loss='mse')

            if restore:
                if self.load(self.ckpt_name) is None:
                    print("Create a new model.")

    def train(self, times=100, batch=500, print_every=10):
        self.print_every=print_every

        dtime=time.time()
        
        if self.X_data is not None and self.Y_data is not None :
            X_data = tf.keras.preprocessing.sequence.pad_sequences(self.X_data).astype(np.float32)
            Y_data = tf.keras.preprocessing.sequence.pad_sequences(self.Y_data).astype(np.float32)
            self.model.fit(X_data, Y_data, epochs=times, batch_size=batch, steps_per_epoch=1, verbose=0, callbacks=[self._cb_fit], use_multiprocessing=True)
        elif self.X_data is None :
            print("Please input a data to X_data.")
            return
        elif self.Y_data is None :
            print("Please input a data to Y_data.")
            return

        print("Training is done.\nTime spent:", round(time.time()-dtime,1), "s\nTraining speed:", round(times/(time.time()-dtime),1), "step/s")

        self.save()

    def run(self,inputs=None,show=True):
        if inputs is None:
            inputs=self.X_data
            print("run with : ",inputs)

        inputs=np.array(inputs, dtype=np.float32)
        while len(inputs.shape) < 3:
            if len(inputs.shape) < 2 :
                inputs=np.expand_dims(inputs, -1)
            else :
                inputs=np.expand_dims(inputs, 0)

        warned_shape=False

        for input in inputs:
            while len(tf.shape(input)) < 3:
                if not warned_shape:
                    print("[Warning] Inputs shape doesn't match. Automatically transformed to 4 Dimensions but may be occur errors or delay.")
                    warned_shape=True
                tf.expand_dims(input,0)

        if len(inputs)<5 and show:
            global plt
            import matplotlib.pyplot as plt
            for input in inputs:
                if self.gray:
                    plt.imshow(input.reshape(self._ipsize[0],self._ipsize[1]), cmap='gray', vmin=0, vmax=255)
                else:
                    input = cv2.cvtColor(input, cv2.COLOR_BGR2RGB)
                    plt.imshow(input.astype('uint8'))
                if isnotebook:
                    plt.show()
                else:
                    plt.savefig('cnn_temp.png', dpi=300)

        inputs=tf.cast(inputs,tf.float32)

        if self.model is not None and inputs is not None :
            return self.model.predict(inputs, use_multiprocessing=True)

    def load_MNIST(self):
        global mnist_train_x, mnist_train_y, mnist_test_x, mnist_test_y
        if mnist_train_x is None and mnist_train_y is None:
            (self.mnist_train_x,self.mnist_train_y), (self.mnist_test_x, self.mnist_test_y) = tf.keras.datasets.mnist.load_data()
            self.mnist_train_x=self.mnist_train_x.reshape(-1, 28, 28, 1)
            self.mnist_train_y=tf.one_hot(self.mnist_train_y, 10)
            self.mnist_test_x=self.mnist_test_x.reshape(-1, 28, 28, 1)
            self.mnist_test_y=tf.one_hot(self.mnist_test_y, 10)
            self.X_data=self.mnist_train_x
            self.Y_data=self.mnist_train_y
        else:
            self.mnist_train_x=mnist_train_x
            self.mnist_train_y=mnist_train_y
            self.mnist_test_x=mnist_test_x
            self.mnist_test_y=mnist_test_y
            self.mnist_train_x=self.mnist_train_x.reshape(-1, 28, 28, 1)
            self.mnist_train_y=tf.one_hot(self.mnist_train_y, 10)
            self.mnist_test_x=self.mnist_test_x.reshape(-1, 28, 28, 1)
            self.mnist_test_y=tf.one_hot(self.mnist_test_y, 10)
            self.X_data=self.mnist_train_x
            self.Y_data=self.mnist_train_y

    def show_img(self, input):
        global plt
        import matplotlib.pyplot as plt
        if self.gray:
            plt.imshow(input.reshape(self._ipsize[0],self._ipsize[1]), cmap='gray', vmin=0, vmax=255)
        else:
            plt.imshow(input.reshape(self._ipsize[0],self._ipsize[1]))
        plt.show()


class RNN(ANN):
    learning_rate=1e-2
    level=0
    ckpt_name="RNN" + IDENTIFIER
    optimizer=None

    # @property
    # def X_data(self):
    #     return self._X_data

    # @X_data.setter
    # def X_data(self, arg):
    #     arr=np.array(arg)
    #     size=arr.shape
    #     arr=arr.reshape((size[0], size[1], 1))
    #     self._X_data=arr

    # @property
    # def Y_data(self):
    #     return self._Y_data

    # @Y_data.setter
    # def Y_data(self, arg):
    #     arr=np.array(arg)
    #     self._Y_data=arr

    def __init__(self, input_size, hidden_size=64, output_size=1, layer_level=1, restore=False, ckpt_name=ckpt_name, softmax=False):
        global tf, layers, Sequential, optimizers, callbacks, losses, Adam, load_model, model_from_json, Model, ctivation, BatchNormalization, Concatenate, Conv2D, Dense, Dropout, GlobalAveragePooling2D, InputLayer, Lambda, MaxPooling2D, add, K
        import tensorflow as tf
        from tensorflow.keras import layers, Sequential, optimizers, callbacks, losses
        from tensorflow.keras.optimizers import Adam
        from tensorflow.keras.models import load_model, model_from_json
        from tensorflow.keras.models import Model
        from tensorflow.keras.layers import Activation, BatchNormalization, Concatenate, Conv2D, Dense, Dropout, GlobalAveragePooling2D, InputLayer, Lambda, MaxPooling2D, add
        from tensorflow.keras import backend as K

        self.optimizer=Adam(lr=self.learning_rate)
        self._cb_fit=callbacks.Callback()

        self.ckpt_name = ckpt_name

        self.restore=restore
        self.softmax=softmax

        self._cb_fit.on_epoch_end=self.on_epoch_end
    
        if layer_level < 1:
            print("Please set a layer level at least 1.")
            del self
        else:
            self.model = Sequential()

            self.model.add(layers.InputLayer(input_shape=(input_size,1)))
            
            self.model.add(layers.Conv1D(filters=32, kernel_size=3, strides=1, padding='same'))
            self.model.add(layers.MaxPooling1D(pool_size=2))
            #self.model.add(layers.Dropout(0.3))

            #self.model.add(layers.Conv1D(filters=16, kernel_size=11, strides=1, padding='valid', activation='relu'))
            #self.model.add(layers.MaxPooling1D(pool_size=3))
            #self.model.add(layers.Dropout(0.3))

            #self.model.add(layers.Conv1D(filters=32, kernel_size=9, strides=1, padding='valid', activation='relu'))
            #self.model.add(layers.MaxPooling1D(pool_size=3))
            #self.model.add(layers.Dropout(0.3))

            for _ in range(layer_level-1):
                #self.model.add(layers.Bidirectional(layers.GRU(hidden_size, return_sequences = True), merge_mode='sum'))
                self.model.add(layers.LSTM(hidden_size, return_sequences = True))

            #self.model.add(layers.Bidirectional(layers.GRU(hidden_size, return_sequences = False), merge_mode='sum'))
            self.model.add(layers.LSTM(hidden_size, return_sequences = False))

            if self.softmax :
                if output_size > 1 :
                    self.model.add(layers.Dense(output_size, activation='softmax'))

                    self.model.compile(optimizer=self.optimizer, loss='categorical_crossentropy')
                else :
                    self.model.add(layers.Dense(output_size, activation='sigmoid'))
                    
                    self.model.compile(optimizer=self.optimizer, loss='binary_crossentropy')
            else :
                self.model.add(layers.Dense(output_size))
                
                self.model.compile(optimizer=self.optimizer, loss='mse')

            if restore:
                if self.load(self.ckpt_name) is None:
                    print("Create a new model.")

    def train(self, times=100, batch=50, print_every=10):
        self.print_every=print_every

        dtime=time.time()
        
        if self.X_data is not None and self.Y_data is not None :
            # print(self.X_data.shape, self.Y_data.shape)
            X_data = tf.keras.preprocessing.sequence.pad_sequences(self.X_data).astype(np.float32)
            Y_data = tf.keras.preprocessing.sequence.pad_sequences(self.Y_data).astype(np.float32)
            self.model.fit(X_data, Y_data, epochs=times, verbose=0, callbacks=[self._cb_fit], use_multiprocessing=True)
        elif self.X_data is None :
            print("Please input a data to X_data.")
            return
        elif self.Y_data is None :
            print("Please input a data to Y_data.")
            return

        print("Training is done.\nTime spent:", round(time.time()-dtime,1), "s\nTraining speed:", round(times/(time.time()-dtime),1), "step/s")

        self.save()

    def run(self,inputs=None):
        if inputs is None:
            inputs=self.X_data
            print("run with : ",inputs)

        inputs=np.array(inputs, dtype=np.float32)
        size=inputs.shape
        inputs=inputs.reshape((size[0], size[1], 1))

        if self.model is not None and inputs is not None :
            return self.model.predict(inputs, use_multiprocessing=True)


class DQN(DNN):
    learning_rate=1e-2
    ckpt_name="DQN" + IDENTIFIER
    rewards=None
    prob=None
    low_limit=1
    low_limit_count=0
    
    def on_train_end(self, logs):
        self.global_step+=1

    def __init__(self, state_size, hidden_size=5, output_size=1, layer_level=1, restore=False, ckpt_name=ckpt_name, softmax=True):
        global tf, layers, Sequential, optimizers, callbacks, losses, Adam, load_model, model_from_json, Model, ctivation, BatchNormalization, Concatenate, Conv2D, Dense, Dropout, GlobalAveragePooling2D, InputLayer, Lambda, MaxPooling2D, add, K
        import tensorflow as tf
        from tensorflow.keras import layers, Sequential, optimizers, callbacks, losses
        from tensorflow.keras.optimizers import Adam
        from tensorflow.keras.models import load_model, model_from_json
        from tensorflow.keras.models import Model
        from tensorflow.keras.layers import Activation, BatchNormalization, Concatenate, Conv2D, Dense, Dropout, GlobalAveragePooling2D, InputLayer, Lambda, MaxPooling2D, add
        from tensorflow.keras import backend as K

        self.optimizer=Adam(lr=self.learning_rate)
        self._cb_fit=callbacks.Callback()

        self.ckpt_name = ckpt_name

        self.restore=restore
        self.softmax=softmax

        self._cb_fit.on_train_end=self.on_train_end

        self.state_size=state_size

        if layer_level < 1:
            print("Please set a layer level at least 1.")
            del self
        else:
            self.model = Sequential()

            self.model.add(layers.InputLayer(input_shape=(state_size,)))
            
            for _ in range(layer_level):
                self.model.add(layers.Dense(hidden_size, kernel_initializer='he_uniform'))

            if self.softmax :
                if output_size > 1 :
                    self.model.add(layers.Dense(output_size, activation='softmax'))

                    self.model.compile(optimizer=self.optimizer, loss=self._catcn_loss)
                else :
                    self.model.add(layers.Dense(output_size, activation='sigmoid'))
                    
                    self.model.compile(optimizer=self.optimizer, loss=self._bincn_loss)
            else :
                self.model.add(layers.Dense(output_size))
                
                self.model.compile(optimizer=self.optimizer, loss=self._mse_loss)

            if restore:
                if self.load(self.ckpt_name) is None:
                    print("Create a new model.")

    def _loss_process(self, loss):
        try : return tf.math.pow(1/tf.math.sqrt(tf.reduce_mean(tf.expand_dims(loss,-1) * self.rewards, -1))*10, 2.)
        except : return loss

    def _bincn_loss(self, y_true, y_pred):
        return self._loss_process(losses.binary_crossentropy(y_true, y_pred))

    def _catcn_loss(self, y_true, y_pred):
        return self._loss_process(losses.categorical_crossentropy(y_true, y_pred))

    def _mse_loss(self, y_true, y_pred):
        return self._loss_process(losses.mse(y_true, y_pred))

    def train(self, states, rewards, actions, times=1, reward_std="time"):
        if reward_std=="time":
            self.rewards=self.process_rewards(rewards)
            states=np.array(states)
            rewards=np.array(rewards)
            actions=tf.cast(actions,tf.float32)

            self.low_limit=max(len(rewards),self.low_limit)#(self.low_limit_count/(self.low_limit_count+1))*self.low_limit+(len(rewards)*1.2)/(self.low_limit_count+1)
            #self.low_limit_count+=1

            hist=self.model.fit(states, actions, epochs=times, verbose=0, callbacks=[self._cb_fit], use_multiprocessing=True)
        
        return tf.reduce_mean(hist.history['loss']).numpy()

    def run(self,inputs=None, boolean=True):
        if inputs is None:
            inputs=self.X_data
            print("run with : ",inputs)
        
        inputs=np.array(inputs, dtype=np.float32)
        while len(inputs.shape) < 2:
            inputs=np.expand_dims(inputs, -1)

        if self.model is not None and inputs is not None :
            if boolean:
                pred=self.model.predict(inputs, use_multiprocessing=True)
                return np.bool(pred>=0.5)
            else:
                return self.model.predict(inputs, use_multiprocessing=True)
                
    def process_rewards(self, r):
        dr = np.zeros_like(r)

        limit=round(self.low_limit*0.7)
        
        tmp=0
        cnt=0
        for i in range(len(r)-limit,len(r)):
            if i>=0:
                tmp+=r[i]
                cnt+=1
        
        dr[-1]=tmp/cnt*limit

        for i in reversed(range(len(r)-limit,len(r)-1)):
            if i>=0:
                dr[i]=dr[i+1]-r[i+1]
            
        for i in reversed(range(len(r)-limit,len(r))):
            if i>=0:
                dr[i]=1/dr[i]
            
        for i in reversed(range(0,len(r)-limit)):
            if i>=0:
                dr[i]=dr[i+1]+r[i+1]

        #dr[-1] = r[-1]
        #for t in reversed(range(0, len(r)-1)):
        #    dr[t] = dr[t+1] + r[t]
        
        return dr#np.power(dr,2)


class FaceNet:
    BASE_PATH=os.path.dirname(os.path.abspath(__file__))
    threshold=0.35
    registed_face=[]
    face_cascade=None

    def __init__(self, path=None):
        global tf, layers, Sequential, optimizers, callbacks, losses, Adam, load_model, model_from_json, Model, ctivation, BatchNormalization, Concatenate, Conv2D, Dense, Dropout, GlobalAveragePooling2D, Input, Lambda, MaxPooling2D, add, K
        import tensorflow as tf
        from tensorflow.keras import layers, Sequential, optimizers, callbacks, losses
        from tensorflow.keras.optimizers import Adam
        from tensorflow.keras.models import load_model, model_from_json
        from tensorflow.keras.models import Model
        from tensorflow.keras.layers import Activation, BatchNormalization, Concatenate, Conv2D, Dense, Dropout, GlobalAveragePooling2D, Input, Lambda, MaxPooling2D, add
        from tensorflow.keras import backend as K

        global cv2
        import cv2

        if path is not None:
            self.BASE_PATH=path

        haar_face= '/usr/local/share/opencv4/haarcascades/haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(haar_face)

        ver=sys.version_info
        self.model_path=self.BASE_PATH+os.path.sep+"model"+os.path.sep+"FaceNet"+os.path.sep+"models"+os.path.sep+str(ver.major)+"."+str(ver.minor)

        if os.path.exists(self.model_path+os.path.sep+"FaceNet.json"):
            try:
                print(1)
                model_file=open(self.model_path+os.path.sep+"FaceNet.json","r")
                model_json=model_file.read()
                model_file.close()
                self.model=model_from_json(model_json)
            except:
                
                self.model=self.create_model()
        else:
            self.model=self.create_model()
        
        self.model.load_weights(self.BASE_PATH+os.path.sep+"model"+os.path.sep+"FaceNet"+os.path.sep+"weights"+os.path.sep+"FaceNet.h5")

    def regist(self, label, data, cropped=True):
        from copy import deepcopy
        img=deepcopy(data)
        face=img

        if not cropped:
            gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            det=self.face_cascade.detectMultiScale(gray, scaleFactor=1.3 ,minNeighbors=1,minSize=(100,100))

            for (x1,y1,w,h) in det:
                th=20
            
                x2=x1+w
                y2=y1+h
                
                x1-=th
                y1-=th
                x2+=th*2
                y2+=th*2
                
                if x1<0: x1=0
                if y1<0: y1=0
                if x2>len(img[0]): x2=len(img[0])
                if y2>len(img): y2=len(img)

                face=img[y1:y2,x1:x2]

        face=cv2.resize(face,(160,160))

        pred=self.model.predict(np.array([face]))

        self.registed_face.append({"name":label, "value":pred[0]})

        return face

    def run(self, inputs, cropped=True):
        inputs=tf.keras.preprocessing.sequence.pad_sequences(inputs).astype(np.float32)
        while len(inputs.shape) < 4 :
            inputs=np.expand_dims(inputs, 0)

        preds=self.model.predict(inputs)

        faceid=[]
        for pred in preds:
            dist=100
            i=0
            faceid.append({"name":"Unknown","value":None,"feature":pred})
            for face in self.registed_face:
                x=self._norm(pred)
                y=self._norm(face["value"])
                d=self._EuD(x,y)
                if dist>d :
                    dist=d
                    if self.threshold>d:
                        faceid[i]["name"]=face["name"]
                        faceid[i]["value"]=d
            i+=1
        
        return faceid


    def _norm(self, x):
        return x / np.sqrt(np.sum(np.multiply(x, x)))

    def _EuD(self, source_representation, test_representation):
        euclidean_distance = source_representation - test_representation
        euclidean_distance = np.sum(np.multiply(euclidean_distance, euclidean_distance))
        euclidean_distance = np.sqrt(euclidean_distance)
        return euclidean_distance

    def _scaling(self, x, scale):
        return x * scale

    def create_model(self):
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

        inputs = Input(shape=(160, 160, 3))
        x = Conv2D(32, 3, strides=2, padding='valid', use_bias=False, name= 'Conv2d_1a_3x3') (inputs)
        x = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Conv2d_1a_3x3_BatchNorm')(x)
        x = Activation('relu', name='Conv2d_1a_3x3_Activation')(x)
        x = Conv2D(32, 3, strides=1, padding='valid', use_bias=False, name= 'Conv2d_2a_3x3') (x)
        x = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Conv2d_2a_3x3_BatchNorm')(x)
        x = Activation('relu', name='Conv2d_2a_3x3_Activation')(x)
        x = Conv2D(64, 3, strides=1, padding='same', use_bias=False, name= 'Conv2d_2b_3x3') (x)
        x = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Conv2d_2b_3x3_BatchNorm')(x)
        x = Activation('relu', name='Conv2d_2b_3x3_Activation')(x)
        x = MaxPooling2D(3, strides=2, name='MaxPool_3a_3x3')(x)
        x = Conv2D(80, 1, strides=1, padding='valid', use_bias=False, name= 'Conv2d_3b_1x1') (x)
        x = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Conv2d_3b_1x1_BatchNorm')(x)
        x = Activation('relu', name='Conv2d_3b_1x1_Activation')(x)
        x = Conv2D(192, 3, strides=1, padding='valid', use_bias=False, name= 'Conv2d_4a_3x3') (x)
        x = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Conv2d_4a_3x3_BatchNorm')(x)
        x = Activation('relu', name='Conv2d_4a_3x3_Activation')(x)
        x = Conv2D(256, 3, strides=2, padding='valid', use_bias=False, name= 'Conv2d_4b_3x3') (x)
        x = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Conv2d_4b_3x3_BatchNorm')(x)
        x = Activation('relu', name='Conv2d_4b_3x3_Activation')(x)
        
        branch_0 = Conv2D(32, 1, strides=1, padding='same', use_bias=False, name= 'Block35_1_Branch_0_Conv2d_1x1') (x)
        branch_0 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block35_1_Branch_0_Conv2d_1x1_BatchNorm')(branch_0)
        branch_0 = Activation('relu', name='Block35_1_Branch_0_Conv2d_1x1_Activation')(branch_0)
        branch_1 = Conv2D(32, 1, strides=1, padding='same', use_bias=False, name= 'Block35_1_Branch_1_Conv2d_0a_1x1') (x)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block35_1_Branch_1_Conv2d_0a_1x1_BatchNorm')(branch_1)
        branch_1 = Activation('relu', name='Block35_1_Branch_1_Conv2d_0a_1x1_Activation')(branch_1)
        branch_1 = Conv2D(32, 3, strides=1, padding='same', use_bias=False, name= 'Block35_1_Branch_1_Conv2d_0b_3x3') (branch_1)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block35_1_Branch_1_Conv2d_0b_3x3_BatchNorm')(branch_1)
        branch_1 = Activation('relu', name='Block35_1_Branch_1_Conv2d_0b_3x3_Activation')(branch_1)
        branch_2 = Conv2D(32, 1, strides=1, padding='same', use_bias=False, name= 'Block35_1_Branch_2_Conv2d_0a_1x1') (x)
        branch_2 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block35_1_Branch_2_Conv2d_0a_1x1_BatchNorm')(branch_2)
        branch_2 = Activation('relu', name='Block35_1_Branch_2_Conv2d_0a_1x1_Activation')(branch_2)
        branch_2 = Conv2D(32, 3, strides=1, padding='same', use_bias=False, name= 'Block35_1_Branch_2_Conv2d_0b_3x3') (branch_2)
        branch_2 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block35_1_Branch_2_Conv2d_0b_3x3_BatchNorm')(branch_2)
        branch_2 = Activation('relu', name='Block35_1_Branch_2_Conv2d_0b_3x3_Activation')(branch_2)
        branch_2 = Conv2D(32, 3, strides=1, padding='same', use_bias=False, name= 'Block35_1_Branch_2_Conv2d_0c_3x3') (branch_2)
        branch_2 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block35_1_Branch_2_Conv2d_0c_3x3_BatchNorm')(branch_2)
        branch_2 = Activation('relu', name='Block35_1_Branch_2_Conv2d_0c_3x3_Activation')(branch_2)
        branches = [branch_0, branch_1, branch_2]
        mixed = Concatenate(axis=3, name='Block35_1_Concatenate')(branches)
        up = Conv2D(256, 1, strides=1, padding='same', use_bias=True, name= 'Block35_1_Conv2d_1x1') (mixed)
        up = Lambda(self._scaling, output_shape=K.int_shape(up)[1:], arguments={'scale': 0.17})(up)
        x = add([x, up])
        x = Activation('relu', name='Block35_1_Activation')(x)
        
        branch_0 = Conv2D(32, 1, strides=1, padding='same', use_bias=False, name= 'Block35_2_Branch_0_Conv2d_1x1') (x)
        branch_0 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block35_2_Branch_0_Conv2d_1x1_BatchNorm')(branch_0)
        branch_0 = Activation('relu', name='Block35_2_Branch_0_Conv2d_1x1_Activation')(branch_0)
        branch_1 = Conv2D(32, 1, strides=1, padding='same', use_bias=False, name= 'Block35_2_Branch_1_Conv2d_0a_1x1') (x)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block35_2_Branch_1_Conv2d_0a_1x1_BatchNorm')(branch_1)
        branch_1 = Activation('relu', name='Block35_2_Branch_1_Conv2d_0a_1x1_Activation')(branch_1)
        branch_1 = Conv2D(32, 3, strides=1, padding='same', use_bias=False, name= 'Block35_2_Branch_1_Conv2d_0b_3x3') (branch_1)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block35_2_Branch_1_Conv2d_0b_3x3_BatchNorm')(branch_1)
        branch_1 = Activation('relu', name='Block35_2_Branch_1_Conv2d_0b_3x3_Activation')(branch_1)
        branch_2 = Conv2D(32, 1, strides=1, padding='same', use_bias=False, name= 'Block35_2_Branch_2_Conv2d_0a_1x1') (x)
        branch_2 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block35_2_Branch_2_Conv2d_0a_1x1_BatchNorm')(branch_2)
        branch_2 = Activation('relu', name='Block35_2_Branch_2_Conv2d_0a_1x1_Activation')(branch_2)
        branch_2 = Conv2D(32, 3, strides=1, padding='same', use_bias=False, name= 'Block35_2_Branch_2_Conv2d_0b_3x3') (branch_2)
        branch_2 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block35_2_Branch_2_Conv2d_0b_3x3_BatchNorm')(branch_2)
        branch_2 = Activation('relu', name='Block35_2_Branch_2_Conv2d_0b_3x3_Activation')(branch_2)
        branch_2 = Conv2D(32, 3, strides=1, padding='same', use_bias=False, name= 'Block35_2_Branch_2_Conv2d_0c_3x3') (branch_2)
        branch_2 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block35_2_Branch_2_Conv2d_0c_3x3_BatchNorm')(branch_2)
        branch_2 = Activation('relu', name='Block35_2_Branch_2_Conv2d_0c_3x3_Activation')(branch_2)
        branches = [branch_0, branch_1, branch_2]
        mixed = Concatenate(axis=3, name='Block35_2_Concatenate')(branches)
        up = Conv2D(256, 1, strides=1, padding='same', use_bias=True, name= 'Block35_2_Conv2d_1x1') (mixed)
        up = Lambda(self._scaling, output_shape=K.int_shape(up)[1:], arguments={'scale': 0.17})(up)
        x = add([x, up])
        x = Activation('relu', name='Block35_2_Activation')(x)
        
        branch_0 = Conv2D(32, 1, strides=1, padding='same', use_bias=False, name= 'Block35_3_Branch_0_Conv2d_1x1') (x)
        branch_0 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block35_3_Branch_0_Conv2d_1x1_BatchNorm')(branch_0)
        branch_0 = Activation('relu', name='Block35_3_Branch_0_Conv2d_1x1_Activation')(branch_0)
        branch_1 = Conv2D(32, 1, strides=1, padding='same', use_bias=False, name= 'Block35_3_Branch_1_Conv2d_0a_1x1') (x)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block35_3_Branch_1_Conv2d_0a_1x1_BatchNorm')(branch_1)
        branch_1 = Activation('relu', name='Block35_3_Branch_1_Conv2d_0a_1x1_Activation')(branch_1)
        branch_1 = Conv2D(32, 3, strides=1, padding='same', use_bias=False, name= 'Block35_3_Branch_1_Conv2d_0b_3x3') (branch_1)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block35_3_Branch_1_Conv2d_0b_3x3_BatchNorm')(branch_1)
        branch_1 = Activation('relu', name='Block35_3_Branch_1_Conv2d_0b_3x3_Activation')(branch_1)
        branch_2 = Conv2D(32, 1, strides=1, padding='same', use_bias=False, name= 'Block35_3_Branch_2_Conv2d_0a_1x1') (x)
        branch_2 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block35_3_Branch_2_Conv2d_0a_1x1_BatchNorm')(branch_2)
        branch_2 = Activation('relu', name='Block35_3_Branch_2_Conv2d_0a_1x1_Activation')(branch_2)
        branch_2 = Conv2D(32, 3, strides=1, padding='same', use_bias=False, name= 'Block35_3_Branch_2_Conv2d_0b_3x3') (branch_2)
        branch_2 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block35_3_Branch_2_Conv2d_0b_3x3_BatchNorm')(branch_2)
        branch_2 = Activation('relu', name='Block35_3_Branch_2_Conv2d_0b_3x3_Activation')(branch_2)
        branch_2 = Conv2D(32, 3, strides=1, padding='same', use_bias=False, name= 'Block35_3_Branch_2_Conv2d_0c_3x3') (branch_2)
        branch_2 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block35_3_Branch_2_Conv2d_0c_3x3_BatchNorm')(branch_2)
        branch_2 = Activation('relu', name='Block35_3_Branch_2_Conv2d_0c_3x3_Activation')(branch_2)
        branches = [branch_0, branch_1, branch_2]
        mixed = Concatenate(axis=3, name='Block35_3_Concatenate')(branches)
        up = Conv2D(256, 1, strides=1, padding='same', use_bias=True, name= 'Block35_3_Conv2d_1x1') (mixed)
        up = Lambda(self._scaling, output_shape=K.int_shape(up)[1:], arguments={'scale': 0.17})(up)
        x = add([x, up])
        x = Activation('relu', name='Block35_3_Activation')(x)
        
        branch_0 = Conv2D(32, 1, strides=1, padding='same', use_bias=False, name= 'Block35_4_Branch_0_Conv2d_1x1') (x)
        branch_0 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block35_4_Branch_0_Conv2d_1x1_BatchNorm')(branch_0)
        branch_0 = Activation('relu', name='Block35_4_Branch_0_Conv2d_1x1_Activation')(branch_0)
        branch_1 = Conv2D(32, 1, strides=1, padding='same', use_bias=False, name= 'Block35_4_Branch_1_Conv2d_0a_1x1') (x)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block35_4_Branch_1_Conv2d_0a_1x1_BatchNorm')(branch_1)
        branch_1 = Activation('relu', name='Block35_4_Branch_1_Conv2d_0a_1x1_Activation')(branch_1)
        branch_1 = Conv2D(32, 3, strides=1, padding='same', use_bias=False, name= 'Block35_4_Branch_1_Conv2d_0b_3x3') (branch_1)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block35_4_Branch_1_Conv2d_0b_3x3_BatchNorm')(branch_1)
        branch_1 = Activation('relu', name='Block35_4_Branch_1_Conv2d_0b_3x3_Activation')(branch_1)
        branch_2 = Conv2D(32, 1, strides=1, padding='same', use_bias=False, name= 'Block35_4_Branch_2_Conv2d_0a_1x1') (x)
        branch_2 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block35_4_Branch_2_Conv2d_0a_1x1_BatchNorm')(branch_2)
        branch_2 = Activation('relu', name='Block35_4_Branch_2_Conv2d_0a_1x1_Activation')(branch_2)
        branch_2 = Conv2D(32, 3, strides=1, padding='same', use_bias=False, name= 'Block35_4_Branch_2_Conv2d_0b_3x3') (branch_2)
        branch_2 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block35_4_Branch_2_Conv2d_0b_3x3_BatchNorm')(branch_2)
        branch_2 = Activation('relu', name='Block35_4_Branch_2_Conv2d_0b_3x3_Activation')(branch_2)
        branch_2 = Conv2D(32, 3, strides=1, padding='same', use_bias=False, name= 'Block35_4_Branch_2_Conv2d_0c_3x3') (branch_2)
        branch_2 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block35_4_Branch_2_Conv2d_0c_3x3_BatchNorm')(branch_2)
        branch_2 = Activation('relu', name='Block35_4_Branch_2_Conv2d_0c_3x3_Activation')(branch_2)
        branches = [branch_0, branch_1, branch_2]
        mixed = Concatenate(axis=3, name='Block35_4_Concatenate')(branches)
        up = Conv2D(256, 1, strides=1, padding='same', use_bias=True, name= 'Block35_4_Conv2d_1x1') (mixed)
        up = Lambda(self._scaling, output_shape=K.int_shape(up)[1:], arguments={'scale': 0.17})(up)
        x = add([x, up])
        x = Activation('relu', name='Block35_4_Activation')(x)
        
        branch_0 = Conv2D(32, 1, strides=1, padding='same', use_bias=False, name= 'Block35_5_Branch_0_Conv2d_1x1') (x)
        branch_0 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block35_5_Branch_0_Conv2d_1x1_BatchNorm')(branch_0)
        branch_0 = Activation('relu', name='Block35_5_Branch_0_Conv2d_1x1_Activation')(branch_0)
        branch_1 = Conv2D(32, 1, strides=1, padding='same', use_bias=False, name= 'Block35_5_Branch_1_Conv2d_0a_1x1') (x)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block35_5_Branch_1_Conv2d_0a_1x1_BatchNorm')(branch_1)
        branch_1 = Activation('relu', name='Block35_5_Branch_1_Conv2d_0a_1x1_Activation')(branch_1)
        branch_1 = Conv2D(32, 3, strides=1, padding='same', use_bias=False, name= 'Block35_5_Branch_1_Conv2d_0b_3x3') (branch_1)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block35_5_Branch_1_Conv2d_0b_3x3_BatchNorm')(branch_1)
        branch_1 = Activation('relu', name='Block35_5_Branch_1_Conv2d_0b_3x3_Activation')(branch_1)
        branch_2 = Conv2D(32, 1, strides=1, padding='same', use_bias=False, name= 'Block35_5_Branch_2_Conv2d_0a_1x1') (x)
        branch_2 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block35_5_Branch_2_Conv2d_0a_1x1_BatchNorm')(branch_2)
        branch_2 = Activation('relu', name='Block35_5_Branch_2_Conv2d_0a_1x1_Activation')(branch_2)
        branch_2 = Conv2D(32, 3, strides=1, padding='same', use_bias=False, name= 'Block35_5_Branch_2_Conv2d_0b_3x3') (branch_2)
        branch_2 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block35_5_Branch_2_Conv2d_0b_3x3_BatchNorm')(branch_2)
        branch_2 = Activation('relu', name='Block35_5_Branch_2_Conv2d_0b_3x3_Activation')(branch_2)
        branch_2 = Conv2D(32, 3, strides=1, padding='same', use_bias=False, name= 'Block35_5_Branch_2_Conv2d_0c_3x3') (branch_2)
        branch_2 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block35_5_Branch_2_Conv2d_0c_3x3_BatchNorm')(branch_2)
        branch_2 = Activation('relu', name='Block35_5_Branch_2_Conv2d_0c_3x3_Activation')(branch_2)
        branches = [branch_0, branch_1, branch_2]
        mixed = Concatenate(axis=3, name='Block35_5_Concatenate')(branches)
        up = Conv2D(256, 1, strides=1, padding='same', use_bias=True, name= 'Block35_5_Conv2d_1x1') (mixed)
        up = Lambda(self._scaling, output_shape=K.int_shape(up)[1:], arguments={'scale': 0.17})(up)
        x = add([x, up])
        x = Activation('relu', name='Block35_5_Activation')(x)

        branch_0 = Conv2D(384, 3, strides=2, padding='valid', use_bias=False, name= 'Mixed_6a_Branch_0_Conv2d_1a_3x3') (x)
        branch_0 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Mixed_6a_Branch_0_Conv2d_1a_3x3_BatchNorm')(branch_0)
        branch_0 = Activation('relu', name='Mixed_6a_Branch_0_Conv2d_1a_3x3_Activation')(branch_0)
        branch_1 = Conv2D(192, 1, strides=1, padding='same', use_bias=False, name= 'Mixed_6a_Branch_1_Conv2d_0a_1x1') (x)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Mixed_6a_Branch_1_Conv2d_0a_1x1_BatchNorm')(branch_1)
        branch_1 = Activation('relu', name='Mixed_6a_Branch_1_Conv2d_0a_1x1_Activation')(branch_1)
        branch_1 = Conv2D(192, 3, strides=1, padding='same', use_bias=False, name= 'Mixed_6a_Branch_1_Conv2d_0b_3x3') (branch_1)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Mixed_6a_Branch_1_Conv2d_0b_3x3_BatchNorm')(branch_1)
        branch_1 = Activation('relu', name='Mixed_6a_Branch_1_Conv2d_0b_3x3_Activation')(branch_1)
        branch_1 = Conv2D(256, 3, strides=2, padding='valid', use_bias=False, name= 'Mixed_6a_Branch_1_Conv2d_1a_3x3') (branch_1)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Mixed_6a_Branch_1_Conv2d_1a_3x3_BatchNorm')(branch_1)
        branch_1 = Activation('relu', name='Mixed_6a_Branch_1_Conv2d_1a_3x3_Activation')(branch_1)
        branch_pool = MaxPooling2D(3, strides=2, padding='valid', name='Mixed_6a_Branch_2_MaxPool_1a_3x3')(x)
        branches = [branch_0, branch_1, branch_pool]
        x = Concatenate(axis=3, name='Mixed_6a')(branches)

        branch_0 = Conv2D(128, 1, strides=1, padding='same', use_bias=False, name= 'Block17_1_Branch_0_Conv2d_1x1') (x)
        branch_0 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block17_1_Branch_0_Conv2d_1x1_BatchNorm')(branch_0)
        branch_0 = Activation('relu', name='Block17_1_Branch_0_Conv2d_1x1_Activation')(branch_0)
        branch_1 = Conv2D(128, 1, strides=1, padding='same', use_bias=False, name= 'Block17_1_Branch_1_Conv2d_0a_1x1') (x)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block17_1_Branch_1_Conv2d_0a_1x1_BatchNorm')(branch_1)
        branch_1 = Activation('relu', name='Block17_1_Branch_1_Conv2d_0a_1x1_Activation')(branch_1)
        branch_1 = Conv2D(128, [1, 7], strides=1, padding='same', use_bias=False, name= 'Block17_1_Branch_1_Conv2d_0b_1x7') (branch_1)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block17_1_Branch_1_Conv2d_0b_1x7_BatchNorm')(branch_1)
        branch_1 = Activation('relu', name='Block17_1_Branch_1_Conv2d_0b_1x7_Activation')(branch_1)
        branch_1 = Conv2D(128, [7, 1], strides=1, padding='same', use_bias=False, name= 'Block17_1_Branch_1_Conv2d_0c_7x1') (branch_1)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block17_1_Branch_1_Conv2d_0c_7x1_BatchNorm')(branch_1)
        branch_1 = Activation('relu', name='Block17_1_Branch_1_Conv2d_0c_7x1_Activation')(branch_1)
        branches = [branch_0, branch_1]
        mixed = Concatenate(axis=3, name='Block17_1_Concatenate')(branches)
        up = Conv2D(896, 1, strides=1, padding='same', use_bias=True, name= 'Block17_1_Conv2d_1x1') (mixed)
        up = Lambda(self._scaling, output_shape=K.int_shape(up)[1:], arguments={'scale': 0.1})(up)
        x = add([x, up])
        x = Activation('relu', name='Block17_1_Activation')(x)
        
        branch_0 = Conv2D(128, 1, strides=1, padding='same', use_bias=False, name= 'Block17_2_Branch_0_Conv2d_1x1') (x)
        branch_0 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block17_2_Branch_0_Conv2d_1x1_BatchNorm')(branch_0)
        branch_0 = Activation('relu', name='Block17_2_Branch_0_Conv2d_1x1_Activation')(branch_0)
        branch_1 = Conv2D(128, 1, strides=1, padding='same', use_bias=False, name= 'Block17_2_Branch_2_Conv2d_0a_1x1') (x)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block17_2_Branch_2_Conv2d_0a_1x1_BatchNorm')(branch_1)
        branch_1 = Activation('relu', name='Block17_2_Branch_2_Conv2d_0a_1x1_Activation')(branch_1)
        branch_1 = Conv2D(128, [1, 7], strides=1, padding='same', use_bias=False, name= 'Block17_2_Branch_2_Conv2d_0b_1x7') (branch_1)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block17_2_Branch_2_Conv2d_0b_1x7_BatchNorm')(branch_1)
        branch_1 = Activation('relu', name='Block17_2_Branch_2_Conv2d_0b_1x7_Activation')(branch_1)
        branch_1 = Conv2D(128, [7, 1], strides=1, padding='same', use_bias=False, name= 'Block17_2_Branch_2_Conv2d_0c_7x1') (branch_1)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block17_2_Branch_2_Conv2d_0c_7x1_BatchNorm')(branch_1)
        branch_1 = Activation('relu', name='Block17_2_Branch_2_Conv2d_0c_7x1_Activation')(branch_1)
        branches = [branch_0, branch_1]
        mixed = Concatenate(axis=3, name='Block17_2_Concatenate')(branches)
        up = Conv2D(896, 1, strides=1, padding='same', use_bias=True, name= 'Block17_2_Conv2d_1x1') (mixed)
        up = Lambda(self._scaling, output_shape=K.int_shape(up)[1:], arguments={'scale': 0.1})(up)
        x = add([x, up])
        x = Activation('relu', name='Block17_2_Activation')(x)
        
        branch_0 = Conv2D(128, 1, strides=1, padding='same', use_bias=False, name= 'Block17_3_Branch_0_Conv2d_1x1') (x)
        branch_0 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block17_3_Branch_0_Conv2d_1x1_BatchNorm')(branch_0)
        branch_0 = Activation('relu', name='Block17_3_Branch_0_Conv2d_1x1_Activation')(branch_0)
        branch_1 = Conv2D(128, 1, strides=1, padding='same', use_bias=False, name= 'Block17_3_Branch_3_Conv2d_0a_1x1') (x)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block17_3_Branch_3_Conv2d_0a_1x1_BatchNorm')(branch_1)
        branch_1 = Activation('relu', name='Block17_3_Branch_3_Conv2d_0a_1x1_Activation')(branch_1)
        branch_1 = Conv2D(128, [1, 7], strides=1, padding='same', use_bias=False, name= 'Block17_3_Branch_3_Conv2d_0b_1x7') (branch_1)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block17_3_Branch_3_Conv2d_0b_1x7_BatchNorm')(branch_1)
        branch_1 = Activation('relu', name='Block17_3_Branch_3_Conv2d_0b_1x7_Activation')(branch_1)
        branch_1 = Conv2D(128, [7, 1], strides=1, padding='same', use_bias=False, name= 'Block17_3_Branch_3_Conv2d_0c_7x1') (branch_1)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block17_3_Branch_3_Conv2d_0c_7x1_BatchNorm')(branch_1)
        branch_1 = Activation('relu', name='Block17_3_Branch_3_Conv2d_0c_7x1_Activation')(branch_1)
        branches = [branch_0, branch_1]
        mixed = Concatenate(axis=3, name='Block17_3_Concatenate')(branches)
        up = Conv2D(896, 1, strides=1, padding='same', use_bias=True, name= 'Block17_3_Conv2d_1x1') (mixed)
        up = Lambda(self._scaling, output_shape=K.int_shape(up)[1:], arguments={'scale': 0.1})(up)
        x = add([x, up])
        x = Activation('relu', name='Block17_3_Activation')(x)
        
        branch_0 = Conv2D(128, 1, strides=1, padding='same', use_bias=False, name= 'Block17_4_Branch_0_Conv2d_1x1') (x)
        branch_0 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block17_4_Branch_0_Conv2d_1x1_BatchNorm')(branch_0)
        branch_0 = Activation('relu', name='Block17_4_Branch_0_Conv2d_1x1_Activation')(branch_0)
        branch_1 = Conv2D(128, 1, strides=1, padding='same', use_bias=False, name= 'Block17_4_Branch_4_Conv2d_0a_1x1') (x)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block17_4_Branch_4_Conv2d_0a_1x1_BatchNorm')(branch_1)
        branch_1 = Activation('relu', name='Block17_4_Branch_4_Conv2d_0a_1x1_Activation')(branch_1)
        branch_1 = Conv2D(128, [1, 7], strides=1, padding='same', use_bias=False, name= 'Block17_4_Branch_4_Conv2d_0b_1x7') (branch_1)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block17_4_Branch_4_Conv2d_0b_1x7_BatchNorm')(branch_1)
        branch_1 = Activation('relu', name='Block17_4_Branch_4_Conv2d_0b_1x7_Activation')(branch_1)
        branch_1 = Conv2D(128, [7, 1], strides=1, padding='same', use_bias=False, name= 'Block17_4_Branch_4_Conv2d_0c_7x1') (branch_1)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block17_4_Branch_4_Conv2d_0c_7x1_BatchNorm')(branch_1)
        branch_1 = Activation('relu', name='Block17_4_Branch_4_Conv2d_0c_7x1_Activation')(branch_1)
        branches = [branch_0, branch_1]
        mixed = Concatenate(axis=3, name='Block17_4_Concatenate')(branches)
        up = Conv2D(896, 1, strides=1, padding='same', use_bias=True, name= 'Block17_4_Conv2d_1x1') (mixed)
        up = Lambda(self._scaling, output_shape=K.int_shape(up)[1:], arguments={'scale': 0.1})(up)
        x = add([x, up])
        x = Activation('relu', name='Block17_4_Activation')(x)
        
        branch_0 = Conv2D(128, 1, strides=1, padding='same', use_bias=False, name= 'Block17_5_Branch_0_Conv2d_1x1') (x)
        branch_0 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block17_5_Branch_0_Conv2d_1x1_BatchNorm')(branch_0)
        branch_0 = Activation('relu', name='Block17_5_Branch_0_Conv2d_1x1_Activation')(branch_0)
        branch_1 = Conv2D(128, 1, strides=1, padding='same', use_bias=False, name= 'Block17_5_Branch_5_Conv2d_0a_1x1') (x)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block17_5_Branch_5_Conv2d_0a_1x1_BatchNorm')(branch_1)
        branch_1 = Activation('relu', name='Block17_5_Branch_5_Conv2d_0a_1x1_Activation')(branch_1)
        branch_1 = Conv2D(128, [1, 7], strides=1, padding='same', use_bias=False, name= 'Block17_5_Branch_5_Conv2d_0b_1x7') (branch_1)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block17_5_Branch_5_Conv2d_0b_1x7_BatchNorm')(branch_1)
        branch_1 = Activation('relu', name='Block17_5_Branch_5_Conv2d_0b_1x7_Activation')(branch_1)
        branch_1 = Conv2D(128, [7, 1], strides=1, padding='same', use_bias=False, name= 'Block17_5_Branch_5_Conv2d_0c_7x1') (branch_1)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block17_5_Branch_5_Conv2d_0c_7x1_BatchNorm')(branch_1)
        branch_1 = Activation('relu', name='Block17_5_Branch_5_Conv2d_0c_7x1_Activation')(branch_1)
        branches = [branch_0, branch_1]
        mixed = Concatenate(axis=3, name='Block17_5_Concatenate')(branches)
        up = Conv2D(896, 1, strides=1, padding='same', use_bias=True, name= 'Block17_5_Conv2d_1x1') (mixed)
        up = Lambda(self._scaling, output_shape=K.int_shape(up)[1:], arguments={'scale': 0.1})(up)
        x = add([x, up])
        x = Activation('relu', name='Block17_5_Activation')(x)
        
        branch_0 = Conv2D(128, 1, strides=1, padding='same', use_bias=False, name= 'Block17_6_Branch_0_Conv2d_1x1') (x)
        branch_0 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block17_6_Branch_0_Conv2d_1x1_BatchNorm')(branch_0)
        branch_0 = Activation('relu', name='Block17_6_Branch_0_Conv2d_1x1_Activation')(branch_0)
        branch_1 = Conv2D(128, 1, strides=1, padding='same', use_bias=False, name= 'Block17_6_Branch_6_Conv2d_0a_1x1') (x)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block17_6_Branch_6_Conv2d_0a_1x1_BatchNorm')(branch_1)
        branch_1 = Activation('relu', name='Block17_6_Branch_6_Conv2d_0a_1x1_Activation')(branch_1)
        branch_1 = Conv2D(128, [1, 7], strides=1, padding='same', use_bias=False, name= 'Block17_6_Branch_6_Conv2d_0b_1x7') (branch_1)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block17_6_Branch_6_Conv2d_0b_1x7_BatchNorm')(branch_1)
        branch_1 = Activation('relu', name='Block17_6_Branch_6_Conv2d_0b_1x7_Activation')(branch_1)
        branch_1 = Conv2D(128, [7, 1], strides=1, padding='same', use_bias=False, name= 'Block17_6_Branch_6_Conv2d_0c_7x1') (branch_1)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block17_6_Branch_6_Conv2d_0c_7x1_BatchNorm')(branch_1)
        branch_1 = Activation('relu', name='Block17_6_Branch_6_Conv2d_0c_7x1_Activation')(branch_1)
        branches = [branch_0, branch_1]
        mixed = Concatenate(axis=3, name='Block17_6_Concatenate')(branches)
        up = Conv2D(896, 1, strides=1, padding='same', use_bias=True, name= 'Block17_6_Conv2d_1x1') (mixed)
        up = Lambda(self._scaling, output_shape=K.int_shape(up)[1:], arguments={'scale': 0.1})(up)
        x = add([x, up])
        x = Activation('relu', name='Block17_6_Activation')(x)    
        
        branch_0 = Conv2D(128, 1, strides=1, padding='same', use_bias=False, name= 'Block17_7_Branch_0_Conv2d_1x1') (x)
        branch_0 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block17_7_Branch_0_Conv2d_1x1_BatchNorm')(branch_0)
        branch_0 = Activation('relu', name='Block17_7_Branch_0_Conv2d_1x1_Activation')(branch_0)
        branch_1 = Conv2D(128, 1, strides=1, padding='same', use_bias=False, name= 'Block17_7_Branch_7_Conv2d_0a_1x1') (x)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block17_7_Branch_7_Conv2d_0a_1x1_BatchNorm')(branch_1)
        branch_1 = Activation('relu', name='Block17_7_Branch_7_Conv2d_0a_1x1_Activation')(branch_1)
        branch_1 = Conv2D(128, [1, 7], strides=1, padding='same', use_bias=False, name= 'Block17_7_Branch_7_Conv2d_0b_1x7') (branch_1)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block17_7_Branch_7_Conv2d_0b_1x7_BatchNorm')(branch_1)
        branch_1 = Activation('relu', name='Block17_7_Branch_7_Conv2d_0b_1x7_Activation')(branch_1)
        branch_1 = Conv2D(128, [7, 1], strides=1, padding='same', use_bias=False, name= 'Block17_7_Branch_7_Conv2d_0c_7x1') (branch_1)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block17_7_Branch_7_Conv2d_0c_7x1_BatchNorm')(branch_1)
        branch_1 = Activation('relu', name='Block17_7_Branch_7_Conv2d_0c_7x1_Activation')(branch_1)
        branches = [branch_0, branch_1]
        mixed = Concatenate(axis=3, name='Block17_7_Concatenate')(branches)
        up = Conv2D(896, 1, strides=1, padding='same', use_bias=True, name= 'Block17_7_Conv2d_1x1') (mixed)
        up = Lambda(self._scaling, output_shape=K.int_shape(up)[1:], arguments={'scale': 0.1})(up)
        x = add([x, up])
        x = Activation('relu', name='Block17_7_Activation')(x)
        
        branch_0 = Conv2D(128, 1, strides=1, padding='same', use_bias=False, name= 'Block17_8_Branch_0_Conv2d_1x1') (x)
        branch_0 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block17_8_Branch_0_Conv2d_1x1_BatchNorm')(branch_0)
        branch_0 = Activation('relu', name='Block17_8_Branch_0_Conv2d_1x1_Activation')(branch_0)
        branch_1 = Conv2D(128, 1, strides=1, padding='same', use_bias=False, name= 'Block17_8_Branch_8_Conv2d_0a_1x1') (x)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block17_8_Branch_8_Conv2d_0a_1x1_BatchNorm')(branch_1)
        branch_1 = Activation('relu', name='Block17_8_Branch_8_Conv2d_0a_1x1_Activation')(branch_1)
        branch_1 = Conv2D(128, [1, 7], strides=1, padding='same', use_bias=False, name= 'Block17_8_Branch_8_Conv2d_0b_1x7') (branch_1)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block17_8_Branch_8_Conv2d_0b_1x7_BatchNorm')(branch_1)
        branch_1 = Activation('relu', name='Block17_8_Branch_8_Conv2d_0b_1x7_Activation')(branch_1)
        branch_1 = Conv2D(128, [7, 1], strides=1, padding='same', use_bias=False, name= 'Block17_8_Branch_8_Conv2d_0c_7x1') (branch_1)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block17_8_Branch_8_Conv2d_0c_7x1_BatchNorm')(branch_1)
        branch_1 = Activation('relu', name='Block17_8_Branch_8_Conv2d_0c_7x1_Activation')(branch_1)
        branches = [branch_0, branch_1]
        mixed = Concatenate(axis=3, name='Block17_8_Concatenate')(branches)
        up = Conv2D(896, 1, strides=1, padding='same', use_bias=True, name= 'Block17_8_Conv2d_1x1') (mixed)
        up = Lambda(self._scaling, output_shape=K.int_shape(up)[1:], arguments={'scale': 0.1})(up)
        x = add([x, up])
        x = Activation('relu', name='Block17_8_Activation')(x)
        
        branch_0 = Conv2D(128, 1, strides=1, padding='same', use_bias=False, name= 'Block17_9_Branch_0_Conv2d_1x1') (x)
        branch_0 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block17_9_Branch_0_Conv2d_1x1_BatchNorm')(branch_0)
        branch_0 = Activation('relu', name='Block17_9_Branch_0_Conv2d_1x1_Activation')(branch_0)
        branch_1 = Conv2D(128, 1, strides=1, padding='same', use_bias=False, name= 'Block17_9_Branch_9_Conv2d_0a_1x1') (x)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block17_9_Branch_9_Conv2d_0a_1x1_BatchNorm')(branch_1)
        branch_1 = Activation('relu', name='Block17_9_Branch_9_Conv2d_0a_1x1_Activation')(branch_1)
        branch_1 = Conv2D(128, [1, 7], strides=1, padding='same', use_bias=False, name= 'Block17_9_Branch_9_Conv2d_0b_1x7') (branch_1)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block17_9_Branch_9_Conv2d_0b_1x7_BatchNorm')(branch_1)
        branch_1 = Activation('relu', name='Block17_9_Branch_9_Conv2d_0b_1x7_Activation')(branch_1)
        branch_1 = Conv2D(128, [7, 1], strides=1, padding='same', use_bias=False, name= 'Block17_9_Branch_9_Conv2d_0c_7x1') (branch_1)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block17_9_Branch_9_Conv2d_0c_7x1_BatchNorm')(branch_1)
        branch_1 = Activation('relu', name='Block17_9_Branch_9_Conv2d_0c_7x1_Activation')(branch_1)
        branches = [branch_0, branch_1]
        mixed = Concatenate(axis=3, name='Block17_9_Concatenate')(branches)
        up = Conv2D(896, 1, strides=1, padding='same', use_bias=True, name= 'Block17_9_Conv2d_1x1') (mixed)
        up = Lambda(self._scaling, output_shape=K.int_shape(up)[1:], arguments={'scale': 0.1})(up)
        x = add([x, up])
        x = Activation('relu', name='Block17_9_Activation')(x)
        
        branch_0 = Conv2D(128, 1, strides=1, padding='same', use_bias=False, name= 'Block17_10_Branch_0_Conv2d_1x1') (x)
        branch_0 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block17_10_Branch_0_Conv2d_1x1_BatchNorm')(branch_0)
        branch_0 = Activation('relu', name='Block17_10_Branch_0_Conv2d_1x1_Activation')(branch_0)
        branch_1 = Conv2D(128, 1, strides=1, padding='same', use_bias=False, name= 'Block17_10_Branch_10_Conv2d_0a_1x1') (x)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block17_10_Branch_10_Conv2d_0a_1x1_BatchNorm')(branch_1)
        branch_1 = Activation('relu', name='Block17_10_Branch_10_Conv2d_0a_1x1_Activation')(branch_1)
        branch_1 = Conv2D(128, [1, 7], strides=1, padding='same', use_bias=False, name= 'Block17_10_Branch_10_Conv2d_0b_1x7') (branch_1)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block17_10_Branch_10_Conv2d_0b_1x7_BatchNorm')(branch_1)
        branch_1 = Activation('relu', name='Block17_10_Branch_10_Conv2d_0b_1x7_Activation')(branch_1)
        branch_1 = Conv2D(128, [7, 1], strides=1, padding='same', use_bias=False, name= 'Block17_10_Branch_10_Conv2d_0c_7x1') (branch_1)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block17_10_Branch_10_Conv2d_0c_7x1_BatchNorm')(branch_1)
        branch_1 = Activation('relu', name='Block17_10_Branch_10_Conv2d_0c_7x1_Activation')(branch_1)
        branches = [branch_0, branch_1]
        mixed = Concatenate(axis=3, name='Block17_10_Concatenate')(branches)
        up = Conv2D(896, 1, strides=1, padding='same', use_bias=True, name= 'Block17_10_Conv2d_1x1') (mixed)
        up = Lambda(self._scaling, output_shape=K.int_shape(up)[1:], arguments={'scale': 0.1})(up)
        x = add([x, up])
        x = Activation('relu', name='Block17_10_Activation')(x)

        branch_0 = Conv2D(256, 1, strides=1, padding='same', use_bias=False, name= 'Mixed_7a_Branch_0_Conv2d_0a_1x1') (x)
        branch_0 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Mixed_7a_Branch_0_Conv2d_0a_1x1_BatchNorm')(branch_0)
        branch_0 = Activation('relu', name='Mixed_7a_Branch_0_Conv2d_0a_1x1_Activation')(branch_0)
        branch_0 = Conv2D(384, 3, strides=2, padding='valid', use_bias=False, name= 'Mixed_7a_Branch_0_Conv2d_1a_3x3') (branch_0)
        branch_0 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Mixed_7a_Branch_0_Conv2d_1a_3x3_BatchNorm')(branch_0)
        branch_0 = Activation('relu', name='Mixed_7a_Branch_0_Conv2d_1a_3x3_Activation')(branch_0)
        branch_1 = Conv2D(256, 1, strides=1, padding='same', use_bias=False, name= 'Mixed_7a_Branch_1_Conv2d_0a_1x1') (x)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Mixed_7a_Branch_1_Conv2d_0a_1x1_BatchNorm')(branch_1)
        branch_1 = Activation('relu', name='Mixed_7a_Branch_1_Conv2d_0a_1x1_Activation')(branch_1)
        branch_1 = Conv2D(256, 3, strides=2, padding='valid', use_bias=False, name= 'Mixed_7a_Branch_1_Conv2d_1a_3x3') (branch_1)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Mixed_7a_Branch_1_Conv2d_1a_3x3_BatchNorm')(branch_1)
        branch_1 = Activation('relu', name='Mixed_7a_Branch_1_Conv2d_1a_3x3_Activation')(branch_1)
        branch_2 = Conv2D(256, 1, strides=1, padding='same', use_bias=False, name= 'Mixed_7a_Branch_2_Conv2d_0a_1x1') (x)
        branch_2 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Mixed_7a_Branch_2_Conv2d_0a_1x1_BatchNorm')(branch_2)
        branch_2 = Activation('relu', name='Mixed_7a_Branch_2_Conv2d_0a_1x1_Activation')(branch_2)
        branch_2 = Conv2D(256, 3, strides=1, padding='same', use_bias=False, name= 'Mixed_7a_Branch_2_Conv2d_0b_3x3') (branch_2)
        branch_2 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Mixed_7a_Branch_2_Conv2d_0b_3x3_BatchNorm')(branch_2)
        branch_2 = Activation('relu', name='Mixed_7a_Branch_2_Conv2d_0b_3x3_Activation')(branch_2)
        branch_2 = Conv2D(256, 3, strides=2, padding='valid', use_bias=False, name= 'Mixed_7a_Branch_2_Conv2d_1a_3x3') (branch_2)
        branch_2 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Mixed_7a_Branch_2_Conv2d_1a_3x3_BatchNorm')(branch_2)
        branch_2 = Activation('relu', name='Mixed_7a_Branch_2_Conv2d_1a_3x3_Activation')(branch_2)
        branch_pool = MaxPooling2D(3, strides=2, padding='valid', name='Mixed_7a_Branch_3_MaxPool_1a_3x3')(x)
        branches = [branch_0, branch_1, branch_2, branch_pool]
        x = Concatenate(axis=3, name='Mixed_7a')(branches)
        
        branch_0 = Conv2D(192, 1, strides=1, padding='same', use_bias=False, name= 'Block8_1_Branch_0_Conv2d_1x1') (x)
        branch_0 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block8_1_Branch_0_Conv2d_1x1_BatchNorm')(branch_0)
        branch_0 = Activation('relu', name='Block8_1_Branch_0_Conv2d_1x1_Activation')(branch_0)
        branch_1 = Conv2D(192, 1, strides=1, padding='same', use_bias=False, name= 'Block8_1_Branch_1_Conv2d_0a_1x1') (x)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block8_1_Branch_1_Conv2d_0a_1x1_BatchNorm')(branch_1)
        branch_1 = Activation('relu', name='Block8_1_Branch_1_Conv2d_0a_1x1_Activation')(branch_1)
        branch_1 = Conv2D(192, [1, 3], strides=1, padding='same', use_bias=False, name= 'Block8_1_Branch_1_Conv2d_0b_1x3') (branch_1)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block8_1_Branch_1_Conv2d_0b_1x3_BatchNorm')(branch_1)
        branch_1 = Activation('relu', name='Block8_1_Branch_1_Conv2d_0b_1x3_Activation')(branch_1)
        branch_1 = Conv2D(192, [3, 1], strides=1, padding='same', use_bias=False, name= 'Block8_1_Branch_1_Conv2d_0c_3x1') (branch_1)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block8_1_Branch_1_Conv2d_0c_3x1_BatchNorm')(branch_1)
        branch_1 = Activation('relu', name='Block8_1_Branch_1_Conv2d_0c_3x1_Activation')(branch_1)
        branches = [branch_0, branch_1]
        mixed = Concatenate(axis=3, name='Block8_1_Concatenate')(branches)
        up = Conv2D(1792, 1, strides=1, padding='same', use_bias=True, name= 'Block8_1_Conv2d_1x1') (mixed)
        up = Lambda(self._scaling, output_shape=K.int_shape(up)[1:], arguments={'scale': 0.2})(up)
        x = add([x, up])
        x = Activation('relu', name='Block8_1_Activation')(x)
        
        branch_0 = Conv2D(192, 1, strides=1, padding='same', use_bias=False, name= 'Block8_2_Branch_0_Conv2d_1x1') (x)
        branch_0 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block8_2_Branch_0_Conv2d_1x1_BatchNorm')(branch_0)
        branch_0 = Activation('relu', name='Block8_2_Branch_0_Conv2d_1x1_Activation')(branch_0)
        branch_1 = Conv2D(192, 1, strides=1, padding='same', use_bias=False, name= 'Block8_2_Branch_2_Conv2d_0a_1x1') (x)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block8_2_Branch_2_Conv2d_0a_1x1_BatchNorm')(branch_1)
        branch_1 = Activation('relu', name='Block8_2_Branch_2_Conv2d_0a_1x1_Activation')(branch_1)
        branch_1 = Conv2D(192, [1, 3], strides=1, padding='same', use_bias=False, name= 'Block8_2_Branch_2_Conv2d_0b_1x3') (branch_1)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block8_2_Branch_2_Conv2d_0b_1x3_BatchNorm')(branch_1)
        branch_1 = Activation('relu', name='Block8_2_Branch_2_Conv2d_0b_1x3_Activation')(branch_1)
        branch_1 = Conv2D(192, [3, 1], strides=1, padding='same', use_bias=False, name= 'Block8_2_Branch_2_Conv2d_0c_3x1') (branch_1)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block8_2_Branch_2_Conv2d_0c_3x1_BatchNorm')(branch_1)
        branch_1 = Activation('relu', name='Block8_2_Branch_2_Conv2d_0c_3x1_Activation')(branch_1)
        branches = [branch_0, branch_1]
        mixed = Concatenate(axis=3, name='Block8_2_Concatenate')(branches)
        up = Conv2D(1792, 1, strides=1, padding='same', use_bias=True, name= 'Block8_2_Conv2d_1x1') (mixed)
        up = Lambda(self._scaling, output_shape=K.int_shape(up)[1:], arguments={'scale': 0.2})(up)
        x = add([x, up])
        x = Activation('relu', name='Block8_2_Activation')(x)
        
        branch_0 = Conv2D(192, 1, strides=1, padding='same', use_bias=False, name= 'Block8_3_Branch_0_Conv2d_1x1') (x)
        branch_0 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block8_3_Branch_0_Conv2d_1x1_BatchNorm')(branch_0)
        branch_0 = Activation('relu', name='Block8_3_Branch_0_Conv2d_1x1_Activation')(branch_0)
        branch_1 = Conv2D(192, 1, strides=1, padding='same', use_bias=False, name= 'Block8_3_Branch_3_Conv2d_0a_1x1') (x)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block8_3_Branch_3_Conv2d_0a_1x1_BatchNorm')(branch_1)
        branch_1 = Activation('relu', name='Block8_3_Branch_3_Conv2d_0a_1x1_Activation')(branch_1)
        branch_1 = Conv2D(192, [1, 3], strides=1, padding='same', use_bias=False, name= 'Block8_3_Branch_3_Conv2d_0b_1x3') (branch_1)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block8_3_Branch_3_Conv2d_0b_1x3_BatchNorm')(branch_1)
        branch_1 = Activation('relu', name='Block8_3_Branch_3_Conv2d_0b_1x3_Activation')(branch_1)
        branch_1 = Conv2D(192, [3, 1], strides=1, padding='same', use_bias=False, name= 'Block8_3_Branch_3_Conv2d_0c_3x1') (branch_1)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block8_3_Branch_3_Conv2d_0c_3x1_BatchNorm')(branch_1)
        branch_1 = Activation('relu', name='Block8_3_Branch_3_Conv2d_0c_3x1_Activation')(branch_1)
        branches = [branch_0, branch_1]
        mixed = Concatenate(axis=3, name='Block8_3_Concatenate')(branches)
        up = Conv2D(1792, 1, strides=1, padding='same', use_bias=True, name= 'Block8_3_Conv2d_1x1') (mixed)
        up = Lambda(self._scaling, output_shape=K.int_shape(up)[1:], arguments={'scale': 0.2})(up)
        x = add([x, up])
        x = Activation('relu', name='Block8_3_Activation')(x)
        
        branch_0 = Conv2D(192, 1, strides=1, padding='same', use_bias=False, name= 'Block8_4_Branch_0_Conv2d_1x1') (x)
        branch_0 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block8_4_Branch_0_Conv2d_1x1_BatchNorm')(branch_0)
        branch_0 = Activation('relu', name='Block8_4_Branch_0_Conv2d_1x1_Activation')(branch_0)
        branch_1 = Conv2D(192, 1, strides=1, padding='same', use_bias=False, name= 'Block8_4_Branch_4_Conv2d_0a_1x1') (x)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block8_4_Branch_4_Conv2d_0a_1x1_BatchNorm')(branch_1)
        branch_1 = Activation('relu', name='Block8_4_Branch_4_Conv2d_0a_1x1_Activation')(branch_1)
        branch_1 = Conv2D(192, [1, 3], strides=1, padding='same', use_bias=False, name= 'Block8_4_Branch_4_Conv2d_0b_1x3') (branch_1)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block8_4_Branch_4_Conv2d_0b_1x3_BatchNorm')(branch_1)
        branch_1 = Activation('relu', name='Block8_4_Branch_4_Conv2d_0b_1x3_Activation')(branch_1)
        branch_1 = Conv2D(192, [3, 1], strides=1, padding='same', use_bias=False, name= 'Block8_4_Branch_4_Conv2d_0c_3x1') (branch_1)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block8_4_Branch_4_Conv2d_0c_3x1_BatchNorm')(branch_1)
        branch_1 = Activation('relu', name='Block8_4_Branch_4_Conv2d_0c_3x1_Activation')(branch_1)
        branches = [branch_0, branch_1]
        mixed = Concatenate(axis=3, name='Block8_4_Concatenate')(branches)
        up = Conv2D(1792, 1, strides=1, padding='same', use_bias=True, name= 'Block8_4_Conv2d_1x1') (mixed)
        up = Lambda(self._scaling, output_shape=K.int_shape(up)[1:], arguments={'scale': 0.2})(up)
        x = add([x, up])
        x = Activation('relu', name='Block8_4_Activation')(x)
        
        branch_0 = Conv2D(192, 1, strides=1, padding='same', use_bias=False, name= 'Block8_5_Branch_0_Conv2d_1x1') (x)
        branch_0 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block8_5_Branch_0_Conv2d_1x1_BatchNorm')(branch_0)
        branch_0 = Activation('relu', name='Block8_5_Branch_0_Conv2d_1x1_Activation')(branch_0)
        branch_1 = Conv2D(192, 1, strides=1, padding='same', use_bias=False, name= 'Block8_5_Branch_5_Conv2d_0a_1x1') (x)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block8_5_Branch_5_Conv2d_0a_1x1_BatchNorm')(branch_1)
        branch_1 = Activation('relu', name='Block8_5_Branch_5_Conv2d_0a_1x1_Activation')(branch_1)
        branch_1 = Conv2D(192, [1, 3], strides=1, padding='same', use_bias=False, name= 'Block8_5_Branch_5_Conv2d_0b_1x3') (branch_1)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block8_5_Branch_5_Conv2d_0b_1x3_BatchNorm')(branch_1)
        branch_1 = Activation('relu', name='Block8_5_Branch_5_Conv2d_0b_1x3_Activation')(branch_1)
        branch_1 = Conv2D(192, [3, 1], strides=1, padding='same', use_bias=False, name= 'Block8_5_Branch_5_Conv2d_0c_3x1') (branch_1)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block8_5_Branch_5_Conv2d_0c_3x1_BatchNorm')(branch_1)
        branch_1 = Activation('relu', name='Block8_5_Branch_5_Conv2d_0c_3x1_Activation')(branch_1)
        branches = [branch_0, branch_1]
        mixed = Concatenate(axis=3, name='Block8_5_Concatenate')(branches)
        up = Conv2D(1792, 1, strides=1, padding='same', use_bias=True, name= 'Block8_5_Conv2d_1x1') (mixed)
        up = Lambda(self._scaling, output_shape=K.int_shape(up)[1:], arguments={'scale': 0.2})(up)
        x = add([x, up])
        x = Activation('relu', name='Block8_5_Activation')(x)
        
        branch_0 = Conv2D(192, 1, strides=1, padding='same', use_bias=False, name= 'Block8_6_Branch_0_Conv2d_1x1') (x)
        branch_0 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block8_6_Branch_0_Conv2d_1x1_BatchNorm')(branch_0)
        branch_0 = Activation('relu', name='Block8_6_Branch_0_Conv2d_1x1_Activation')(branch_0)
        branch_1 = Conv2D(192, 1, strides=1, padding='same', use_bias=False, name= 'Block8_6_Branch_1_Conv2d_0a_1x1') (x)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block8_6_Branch_1_Conv2d_0a_1x1_BatchNorm')(branch_1)
        branch_1 = Activation('relu', name='Block8_6_Branch_1_Conv2d_0a_1x1_Activation')(branch_1)
        branch_1 = Conv2D(192, [1, 3], strides=1, padding='same', use_bias=False, name= 'Block8_6_Branch_1_Conv2d_0b_1x3') (branch_1)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block8_6_Branch_1_Conv2d_0b_1x3_BatchNorm')(branch_1)
        branch_1 = Activation('relu', name='Block8_6_Branch_1_Conv2d_0b_1x3_Activation')(branch_1)
        branch_1 = Conv2D(192, [3, 1], strides=1, padding='same', use_bias=False, name= 'Block8_6_Branch_1_Conv2d_0c_3x1') (branch_1)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block8_6_Branch_1_Conv2d_0c_3x1_BatchNorm')(branch_1)
        branch_1 = Activation('relu', name='Block8_6_Branch_1_Conv2d_0c_3x1_Activation')(branch_1)
        branches = [branch_0, branch_1]
        mixed = Concatenate(axis=3, name='Block8_6_Concatenate')(branches)
        up = Conv2D(1792, 1, strides=1, padding='same', use_bias=True, name= 'Block8_6_Conv2d_1x1') (mixed)
        up = Lambda(self._scaling, output_shape=K.int_shape(up)[1:], arguments={'scale': 1})(up)
        x = add([x, up])
        
        x = GlobalAveragePooling2D(name='AvgPool')(x)
        x = Dropout(1.0 - 0.8, name='Dropout')(x)
        x = Dense(128, use_bias=False, name='Bottleneck')(x)
        x = BatchNormalization(momentum=0.995, epsilon=0.001, scale=False, name='Bottleneck_BatchNorm')(x)

        model = Model(inputs, x, name='inception_resnet_v1')

        model_json=model.to_json()
        with open(self.model_path+os.path.sep+"FaceNet.json", "w") as json_file : 
            json_file.write(model_json)

        return model



def onehot(array, classes):
    arr=np.array(array)
    return np.squeeze(np.eye(classes)[arr.reshape(-1)])



#---------------------------------------------  Ondevice AI ---------------------------------------------

def import_tensorflow():
    global tf
    import tensorflow as tf
    import tensorflow.keras as __module
    
    for k in [m for m in dir(__module) if not "__" in m]:
        globals()[k]=__module.__dict__[k]

    import tensorflow.keras.models as __module
    
    for k in [m for m in dir(__module) if not "__" in m]:
        globals()[k]=__module.__dict__[k]

    import tensorflow.keras.losses as __module
    
    for k in [m for m in dir(__module) if not "__" in m]:
        globals()[k]=__module.__dict__[k]

    import tensorflow.keras.layers as __module
    
    for k in [m for m in dir(__module) if not "__" in m]:
        globals()[k]=__module.__dict__[k]

    import tensorflow.keras.optimizers as __module
    
    for k in [m for m in dir(__module) if not "__" in m]:
        globals()[k]=__module.__dict__[k]

    import tensorflow.keras.utils as __module
    
    for k in [m for m in dir(__module) if not "__" in m]:
        globals()[k]=__module.__dict__[k]

    gpus=tf.config.experimental.list_physical_devices('GPU')

    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

def bgr8_to_jpeg(value):
    return bytes(cv2.imencode('.jpg', value)[1])


class _cam_based_class:
    _camera=None

    def __init__(self, camera=None):
        if hasattr(camera,'code') and camera.code==Camera.code:
            self._camera=camera
        else:
            print('This Camera class is not available.')
            del self
    @property
    def camera(self):
        return self._camera
    
    @camera.setter
    def camera(self,camera):
        if type(camera)==Camera:
            self._camera=camera
        else:
            print('Not available class.')

# class DataLoader(Sequence):
#     def __init__(self, x_set, y_set, batch_size):
#         self.x, self.y = x_set, y_set
#         self.batch_size = batch_size

#     def __len__(self):
#         return math.ceil(len(self.x) / self.batch_size)

#     def __getitem__(self, idx):
#         batch_x = self.x[idx * self.batch_size:(idx + 1) *
#         self.batch_size]
#         batch_y = self.y[idx * self.batch_size:(idx + 1) *
#         self.batch_size]

#         return np.array(batch_x), np.array(batch_y)


class Collision_Avoid(_cam_based_class):
    MODEL_PATH = 'collision_avoid_model.h5'
    dataset_path = 'collision_dataset'
    datasets=None
    train_dataset=None
    test_dataset=None
    train_loader=None
    test_loader=None
    device=None
    model=None
    ready2show=False
    slider=None
    STAT_DEFINED=0
    STAT_READY=1
    _stat=STAT_DEFINED
    # BATCH_SIZE=8

    def __init__(self,camera):
        super().__init__(camera)
        import_tensorflow()
        self.default_path=os.path.abspath(__file__+"/../model/Collision_Avoid/")

    def load_datasets(self, path=dataset_path):
        if isnotebook:
            datasets_noti_widget = widgets.Label(value="Loading datasets...")
            model_noti_widget = widgets.Label(value="Creating a new model...")

            display(datasets_noti_widget)

        if not os.path.exists(path):
            print(path," doesn't exist.")
            return

        width=self.camera.width
        height=self.camera.height

        X=[]
        Y=[]

        blocked_data_list=os.listdir(path+"/blocked")

        for f in blocked_data_list:
            if ".jpg" in f.lower():
                img=cv2.imread(path+"/blocked/"+f)

                if img is not None:
                    # height, width, _ = img.shape
                    if width!=300 and height!=300: img=cv2.resize(img,(300,300))

                    X.append(img)
                    Y.append([1])

        free_data_list=os.listdir(path+"/free")

        for f in free_data_list:
            if ".jpg" in f:
                img=cv2.imread(path+"/free/"+f)

                if img is not None:
                    # height, width, _ = img.shape
                    if width!=300 and height!=300: img=cv2.resize(img,(300,300))

                    X.append(img)
                    Y.append([0])

        self.X=np.array(X)
        self.Y=np.array(Y)

        if len(X)<=0:
            if isnotebook:
                datasets_noti_widget.value="Can't access datasets. Check file permission or existence."
            return
        else:
            if isnotebook:
                datasets_noti_widget.value="Loaded "+str(len(X))+" datasets."
            else:
                pass

        if self.model is None:
            if isnotebook:
                display(model_noti_widget)

            self.model=load_model(self.default_path+"/Collision_Avoid.h5", compile=False)

            adam=Adam(learning_rate=1e-3)
            
            self.model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])

            if isnotebook:
                model_noti_widget.value="Model creation completed."
            
        self._stat=self.STAT_READY

    def train(self, times=5, autosave=True):
        if self._stat < self.STAT_READY :
            print("Please load datasets as load_datasets() method.")
            return

        if isnotebook:
            totally_progress_widget = widgets.FloatProgress(min=0.0, max=100.0, description='Total')
            progress_widget = widgets.FloatProgress(min=0.0, max=100.0, description='This step')
            total_percentage_widget = widgets.Label(value="0%")
            current_percentage_widget = widgets.Label(value="0%")
            remaining_time_widget = widgets.Label(value="0")
            loss_widget = widgets.Label(value="0")

            row1=widgets.HBox([totally_progress_widget, total_percentage_widget])
            row2=widgets.HBox([progress_widget, current_percentage_widget])
            row3=widgets.HBox([widgets.Label(value="Remaining"), remaining_time_widget, widgets.Label(value="sec")])
            row4=widgets.HBox([widgets.Label(value="Loss : "), loss_widget])

            display(widgets.VBox([row1, row2, row3, row4]))

        class Train_Progress(callbacks.Callback):
            def __init__(self, batch_size, data_size, train_size):
                self.batch_size=batch_size
                self.data_size=data_size
                self.train_size=train_size

                self.last_epoch_log=time.time()
                self.epoch=0

                super().__init__()

            def on_train_begin(self, logs=None):
                self.train_timelog=time.time()

            def on_epoch_begin(self, epoch, logs=None):
                self.epoch_timelog=time.time()

            def on_epoch_end(self, epoch, logs=None):
                self.epoch=epoch+1
                self.last_epoch_log=time.time()

                print(str(epoch+1),' epoch loss : '+str(round(logs['loss'], 11))+", \taccuracy : "+str(round(logs['accuracy']*100, 1))+"%")

            def on_train_batch_begin(self, batch, logs=None):
                self.batch_timelog=time.time()

            def on_train_batch_end(self, batch, logs=None):
                batch+=1
                #Epoch ETA

                spent_time_batches=time.time()-self.epoch_timelog

                remain_data=self.data_size-(batch*self.batch_size)
                time_per_sample=spent_time_batches/(batch*self.batch_size)

                epoch_eta=max(remain_data*time_per_sample, 0)
                epoch_progress_rate=min(round((float(batch*self.batch_size)/self.data_size)*100, 1), 100.0)

                #Train ETA

                if self.epoch>0:
                    time_per_epoch=float(self.last_epoch_log-self.train_timelog)/self.epoch
                else:
                    time_per_epoch=self.data_size*time_per_sample

                remained_time_epochs=(self.train_size-self.epoch-1)*time_per_epoch

                train_eta=max(remained_time_epochs+epoch_eta, 0)
                train_progress_rate=min(round((1-train_eta/(self.train_size*time_per_epoch))*100, 1), 100.0)

                if isnotebook:
                    current_percentage_widget.value=str(epoch_progress_rate)+"%"
                    progress_widget.value=epoch_progress_rate

                    total_percentage_widget.value=str(train_progress_rate)+"%"
                    totally_progress_widget.value=train_progress_rate

                    #spent_time+=time.time()-time_check
                    remaining_time_widget.value=str(round(train_eta, 3))

                    loss_widget.value=str(round(logs['loss'], 7))

        self.model.fit(self.X, self.Y, batch_size=2, epochs=times, callbacks=[Train_Progress(16, len(self.Y), times)], verbose=0)

        if autosave:
            self.model.save(self.MODEL_PATH)

    def load_model(self,path=MODEL_PATH):
        if not os.path.exists(path):
            print(path," doesn't exist.")
            return

        self.model=load_model(path)

    def save_model(self,path=MODEL_PATH):
        if self.model is not None:
            self.model.save(path)
            print("Save completed.")
        else:
            print("The model can't be saved cause it's None.")

    def show(self):
        if self.slider is None:
            self.slider = widgets.FloatSlider(description='blocked', min=0.0, max=1.0, orientation='vertical')
        display(widgets.HBox([self.camera.image, self.slider]))

    def run(self, value=None, callback=None):  
        if self.model is None :
            print("Please load datasets as load_datasets() method or load a trained model as load_model() method.")
            return

        x = self.camera.value if value is None else value
        y = self.model(np.array([x]).astype(np.float32)).numpy()[0][0]
        
        if self.slider is not None:
            self.slider.value = y
        
        try:
            if callback is not None:
                callback(y)
        except Exception as e:
            print("Error : Can't callback this method.")
            print(e)

        self.value = x

        return y


class Object_Follow(_cam_based_class):
    label_list=['person', 'light_stop', 'light_wait', 'light_go', 'light_left', 'sign_cross', 'sign_50', 'sign_30', 'sign_left', 'sign_stop', 'sign_park', 'deer', 'bottom_l', 'bottom_r', 'sign_back', 'car_back']
    #label_list=['person', 'bicycle', 'car', 'motorbike', 'aeroplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'sofa', 'pottedplant', 'bed', 'diningtable', 'toilet', 'tvmonitor', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

    def __init__(self, camera, classes_path=None, label_list=None):
        import cv2, os, ctypes
        import tensorrt as trt
        import numpy as np
        import pycuda.driver as cuda
        import pycuda.autoinit
        global cv2, os, ctypes, trt, np, cuda

        self.default_path=os.path.abspath(__file__+"/../model/yolov4-tiny/")

        if label_list != None:
            self.label_list = label_list
            
        super().__init__(camera)
        import_tensorflow()
        if isnotebook:
            self.probWidget = widgets.Image(format='jpeg', width=camera.width, height=camera.height)
        else:
            self.probWidget = None
        self.cls_dict = {i: n for i, n in enumerate(self.label_list)}

    def __allocate_buffers(self, engine):
        inputs = []
        outputs = []
        bindings = []
        output_idx = 0
        stream = cuda.Stream()
        
        class HostDeviceMem(object):
            def __init__(self, host_mem, device_mem):
                self.host = host_mem
                self.device = device_mem

            def __repr__(self):
                return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)
    
        for binding in engine:
            binding_dims = engine.get_binding_shape(binding)
            if len(binding_dims) == 4:
                size = trt.volume(binding_dims)
            elif len(binding_dims) == 3:
                size = trt.volume(binding_dims) * engine.max_batch_size
            else:
                raise ValueError('bad dims of binding %s: %s' % (binding, str(binding_dims)))
            dtype = trt.nptype(engine.get_binding_dtype(binding))
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            bindings.append(int(device_mem))
            if engine.binding_is_input(binding):
                inputs.append(HostDeviceMem(host_mem, device_mem))
            else:
                assert size % 7 == 0
                outputs.append(HostDeviceMem(host_mem, device_mem))
                output_idx += 1
        assert len(inputs) == 1
        assert len(outputs) == 1
        return inputs, outputs, bindings, stream
    
    def __preprocess_yolo(self, img, input_shape):
        img = cv2.resize(img, (input_shape[1], input_shape[0]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.transpose((2, 0, 1)).astype(np.float32)
        img /= 255.0
        return img
    
    def __postprocess_yolo(self, trt_outputs, img_w, img_h, conf_th, nms_threshold, input_shape):
        detections = []
        for o in trt_outputs:
            dets = o.reshape((-1, 7))
            dets = dets[dets[:, 4] * dets[:, 6] >= conf_th]
            detections.append(dets)
        detections = np.concatenate(detections, axis=0)

        if len(detections) == 0:
            boxes = np.zeros((0, 4), dtype=np.int)
            scores = np.zeros((0,), dtype=np.float32)
            classes = np.zeros((0,), dtype=np.float32)
        else:
            old_h, old_w = img_h, img_w
            detections[:, 0:4] *= np.array([old_w, old_h, old_w, old_h], dtype=np.float32)

            nms_detections = np.zeros((0, 7), dtype=detections.dtype)
            for class_id in set(detections[:, 5]):
                idxs = np.where(detections[:, 5] == class_id)
                cls_detections = detections[idxs]                
                
                x_coord = cls_detections[:, 0]
                y_coord = cls_detections[:, 1]
                width = cls_detections[:, 2]
                height = cls_detections[:, 3]
                box_confidences = cls_detections[:, 4] * cls_detections[:, 6]

                areas = width * height
                ordered = box_confidences.argsort()[::-1]

                keep = list()
                while ordered.size > 0:
                    i = ordered[0]
                    keep.append(i)
                    xx1 = np.maximum(x_coord[i], x_coord[ordered[1:]])
                    yy1 = np.maximum(y_coord[i], y_coord[ordered[1:]])
                    xx2 = np.minimum(x_coord[i] + width[i], x_coord[ordered[1:]] + width[ordered[1:]])
                    yy2 = np.minimum(y_coord[i] + height[i], y_coord[ordered[1:]] + height[ordered[1:]])

                    width1 = np.maximum(0.0, xx2 - xx1 + 1)
                    height1 = np.maximum(0.0, yy2 - yy1 + 1)
                    intersection = width1 * height1
                    union = (areas[i] + areas[ordered[1:]] - intersection)
                    iou = intersection / union
                    indexes = np.where(iou <= nms_threshold)[0]
                    ordered = ordered[indexes + 1]

                keep = np.array(keep)                    
                nms_detections = np.concatenate([nms_detections, cls_detections[keep]], axis=0)

            xx = nms_detections[:, 0].reshape(-1, 1)
            yy = nms_detections[:, 1].reshape(-1, 1)
            ww = nms_detections[:, 2].reshape(-1, 1)
            hh = nms_detections[:, 3].reshape(-1, 1)
            boxes = np.concatenate([xx, yy, xx+ww, yy+hh], axis=1) + 0.5
            boxes = boxes.astype(np.int)
            scores = nms_detections[:, 4] * nms_detections[:, 6]
            classes = nms_detections[:, 5]
        return boxes, scores, classes
    
    def __inference_fn(self, context, bindings, inputs, outputs, stream):
        [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
        context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
        [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
        stream.synchronize()
        return [out.host for out in outputs]
    
    def __draw_bboxes(self, img, boxes, clss, confs, idx, mx):
        for i, (bb, cl, cf) in enumerate(zip(boxes, clss, confs)):
            cl = int(cl)
            x_min, y_min, x_max, y_max = bb[0], bb[1], bb[2], bb[3]
            if cl == idx and i == mx:
                box_color = (0,255,0)
                text_color = (0, 0, 0)
            else:
                box_color = (255,0,0)
                text_color = (255, 255, 255)
            cv2.rectangle(img, (x_min, y_min), (x_max, y_max), box_color, 2)
            txt_loc = (max(x_min+2, 0), max(y_min+2, 0))
            cls_name = self.cls_dict.get(cl, 'CLS{}'.format(cl))
            txt = '{} {:.2f}'.format(cls_name, cf)
            img_h, img_w, _ = img.shape
            if txt_loc[0] >= img_w or txt_loc[1] >= img_h:
                return img
            margin = 3
            size = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            w = size[0][0] + margin * 2
            h = size[0][1] + margin * 2            
            cv2.rectangle(img, (x_min-1, y_min-1-h), (x_min+w, y_min), box_color, -1)
            cv2.putText(img, txt, (x_min+margin, y_min-margin-2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, lineType=cv2.LINE_AA)
        return img
    
    def load_model(self, path=None):    
        if path is None:
            self.model_path=self.default_path+'/yolov4-tiny.trt'
        else:
            self.model_path=path
        self.layer_path=self.default_path+'/libyolo_layer.so'

        try:
            ctypes.cdll.LoadLibrary(self.layer_path)
        except OSError as e:
            raise SystemExit('ERROR: failed to load layer file.') from e

        self.trt_logger = trt.Logger(trt.Logger.INFO)
        with open(self.model_path, 'rb') as f, trt.Runtime(self.trt_logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        if not self.engine:
            raise ValueError('failed to deserialize the engine.')

        binding = self.engine[0]
        binding_dims = self.engine.get_binding_shape(binding)
        try:
            self.input_shape = tuple(binding_dims[2:])
        except:            
            raise ValueError('bad dims of binding %s: %s' % (binding, str(binding_dims)))

        try:
            self.context = self.engine.create_execution_context()
            self.inputs, self.outputs, self.bindings, self.stream = self.__allocate_buffers(self.engine)
        except Exception as e:
            raise RuntimeError('fail to allocate CUDA resources') from e
    
    def show(self):
        display(self.probWidget)

    def detect(self, image=None, index=None, threshold=0.5, show=True, callback=None):
        if image is None:
            image=self.camera.value
        
        width = image.shape[1]
        height = image.shape[0]

        if type(index)==str:
            try:
                index=self.label_list.index(index)
            except ValueError:
                print("index is not available.")
                return

        img_resized = self.__preprocess_yolo(image, self.input_shape)
        self.inputs[0].host = np.ascontiguousarray(img_resized)

        self.trt_outputs = self.__inference_fn(self.context, self.bindings, self.inputs, self.outputs, self.stream)
        boxes, scores, classes = self.__postprocess_yolo(self.trt_outputs, width, height, threshold, nms_threshold=0.5, input_shape=self.input_shape)
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, width-1)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, height-1)

        detections = []
        raw_value = []
        for bb, cl in zip(boxes, classes):
            detections.append({'label' : int(cl), 
                            'bbox' : [round(bb[0]/(width/2)-1,2), round(bb[1]/(height/2)-1,2), round(bb[2]/(width/2)-1,2), round(bb[3]/(height/2)-1,2)], 
                            'x' : round(((bb[0]+bb[2])/2)/(width/2)-1, 5), 
                            'y' : round(((bb[1]+bb[3])/2)/(height/2)-1, 5), 
                            'size_rate': round((bb[2]-bb[0])*(bb[3]-bb[1])/(width * height), 5)})
            raw_value.append([int(cl), bb.tolist()])

        max_size = None
        if index is None:
            result = []
            for det in detections:
                result.append({'label': det['label'], 'bbox': det['bbox']})
        else:
            matching_detections = [d for d in detections if d['label'] == index]

            sizes = [det['size_rate'] for det in matching_detections if det['label']==index]
            if len(sizes) > 0:
                max_size = np.array(sizes).argmax()
                result = matching_detections[max_size]
                result.pop('label')
                result.pop('bbox')
            else:
                result = None

        if show:
            image = self.__draw_bboxes(image, boxes, classes, scores, index, max_size)
            if isnotebook:
                self.probWidget.value=bgr8_to_jpeg(image)
            self.value=image            
        self.raw_value = raw_value
        
        return result


class Track_Follow(_cam_based_class):
    MODEL_PATH = 'Track_Follow.h5'
    dataset_path = 'track_dataset'
    BATCH_SIZE = 2
    model=None
    device=None
    datasets=None
    optimizer=None
    prob=None
    probWidget=None
    STAT_DEFINED=0
    STAT_READY=1
    _stat=STAT_DEFINED

    def __init__(self, camera):
        super().__init__(camera)
        import_tensorflow()
        self.default_path=os.path.abspath(__file__+"/../model/Track_Follow/")
        if isnotebook:
            self.probWidget = widgets.Image(format='jpeg', width=camera.width, height=camera.height)
            
    def load_datasets(self, path=dataset_path):
        if isnotebook:
            datasets_noti_widget = widgets.Label(value="Loading datasets...")
            model_noti_widget = widgets.Label(value="Creating a new model...")

            display(datasets_noti_widget)

        if not os.path.exists(path):
            print(path," doesn't exist.")
            return

        file_list=os.listdir(path)

        X=[]
        Y=[]

        for f in file_list:
            if ".jpg" in f.lower():
                img=cv2.imread(path+"/"+f)
                height, width, _ = img.shape

                if img is not None:
                    if width!=300 and height!=300: img=cv2.resize(img,(300,300))

                    s=f.split("_")

                    a=(int(s[0])/width-0.5)*2
                    b=(int(s[1])/height-0.5)*2
                    
                    X.append(img)
                    Y.append([a,b])

        self.X=np.array(X)
        self.Y=np.array(Y)

        if len(X)<=0:
            if isnotebook:
                datasets_noti_widget.value="Can't access datasets. Check file permission or existence."
            else:
                print("Can't access datasets. Check file permission or existence.")
            return
        else:
            if isnotebook:
                datasets_noti_widget.value="Loaded "+str(len(X))+" datasets."
            else:
                print("Loaded "+str(len(X))+" datasets.")

        if self.model is None:
            if isnotebook:
                display(model_noti_widget)

            self.model=load_model(self.default_path+"/Track_Follow.h5", compile=False)
            
            adam=Adam(learning_rate=1e-6)
            
            self.model.compile(optimizer=adam, loss='MAE')

            if isnotebook:
                model_noti_widget.value="Model creation completed."
            else:
                print("Model creation completed.")
            
        self._stat=self.STAT_READY

    def train(self, times=5, autosave=True):
        if self._stat < self.STAT_READY :
            print("Please load datasets as load_datasets() method.")
            return

        if isnotebook:
            totally_progress_widget = widgets.FloatProgress(min=0.0, max=100.0, description='Total')
            progress_widget = widgets.FloatProgress(min=0.0, max=100.0, description='This step')
            total_percentage_widget = widgets.Label(value="0%")
            current_percentage_widget = widgets.Label(value="0%")
            remaining_time_widget = widgets.Label(value="0")
            loss_widget = widgets.Label(value="0")

            row1=widgets.HBox([totally_progress_widget, total_percentage_widget])
            row2=widgets.HBox([progress_widget, current_percentage_widget])
            row3=widgets.HBox([widgets.Label(value="Remaining"), remaining_time_widget, widgets.Label(value="sec")])
            row4=widgets.HBox([widgets.Label(value="Loss : "), loss_widget])

            display(widgets.VBox([row1, row2, row3, row4]))

        class Train_Progress(callbacks.Callback):
            timelog=0

            def __init__(self, batch_size, data_size, train_size):
                self.batch_size=batch_size
                self.data_size=data_size
                self.train_size=train_size

                self.last_epoch_log=time.time()
                self.epoch=0

                super().__init__()

            def on_train_begin(self, logs=None):
                self.train_timelog=time.time()

            def on_epoch_begin(self, epoch, logs=None):
                self.epoch_timelog=time.time()

            def on_epoch_end(self, epoch, logs=None):
                self.epoch=epoch+1
                self.last_epoch_log=time.time()

                print(str(epoch+1),' epoch loss : '+str(logs['loss']))

            def on_train_batch_begin(self, batch, logs=None):
                self.batch_timelog=time.time()

            def on_train_batch_end(self, batch, logs=None):
                batch+=1
                #Epoch ETA

                spent_time_batches=time.time()-self.epoch_timelog

                remain_data=self.data_size-(batch*self.batch_size)
                time_per_sample=spent_time_batches/(batch*self.batch_size)

                epoch_eta=max(remain_data*time_per_sample, 0)
                epoch_progress_rate=min(round((float(batch*self.batch_size)/self.data_size)*100 ,1), 100.0)


                #Train ETA

                if self.epoch>0:
                    time_per_epoch=float(self.last_epoch_log-self.train_timelog)/self.epoch
                else:
                    time_per_epoch=self.data_size*time_per_sample

                remained_time_epochs=(self.train_size-self.epoch-1)*time_per_epoch

                train_eta=max(remained_time_epochs+epoch_eta, 0)
                train_progress_rate=min(round((1-train_eta/(self.train_size*time_per_epoch))*100, 1), 100.0)

                if isnotebook:
                    current_percentage_widget.value=str(epoch_progress_rate)+"%"
                    progress_widget.value=epoch_progress_rate

                    total_percentage_widget.value=str(train_progress_rate)+"%"
                    totally_progress_widget.value=train_progress_rate

                    # spent_time+=time.time()-time_check
                    remaining_time_widget.value=str(round(train_eta, 3))

                    loss_widget.value=str(round(logs['loss'], 7))

        self.model.fit(self.X, self.Y, batch_size=self.BATCH_SIZE, epochs=times, callbacks=[Train_Progress(self.BATCH_SIZE, len(self.Y), times)], verbose=0)

        if autosave:
            self.model.save(self.MODEL_PATH)

    def load_model(self,path=MODEL_PATH):
        if not os.path.exists(path):
            print(path," doesn't exist.")
            return

        self.model=load_model(path)

    def save_model(self,path=MODEL_PATH):
        if path[-3:] != '.h5':
            path = path + '.h5'
            
        if self.model is not None:
            self.model.save(path)
            print("Save completed.")
        else:
            print("The model can't be saved cause it's None.")

    def show(self):
        display(self.probWidget)

    def run(self, value=None, callback=None):
        if self.model is None :
            print("Please load datasets as load_datasets() method or load a trained model as load_model() method.")
            return
        
        img = self.camera.value if value is None else value
        x, y = self.model(np.array([img]).astype(np.float32)).numpy()[0]

        cX = int(self.camera.width * (x / 2.0 + 0.5))
        cY = int(self.camera.height * (y / 2.0 + 0.5))

        if isnotebook:
            self.value = cv2.circle(img, (cX, cY), 6, (255, 255, 255), 2)
            self.probWidget.value = bgr8_to_jpeg(self.value)

        result={"x":x,"y":y}

        if callback is not None:
            callback(result)

        return result
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import *
from tensorflow.keras.models import *
from tensorflow.keras.losses import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.utils import *
class Track_Follow_TF(_cam_based_class):
    MODEL_PATH = 'Track_Follow.h5'
    dataset_path = 'track_dataset'
    BATCH_SIZE = 8
    model=None
    device=None
    datasets=None
    optimizer=None
    prob=None
    probWidget=None
    STAT_DEFINED=0
    STAT_READY=1
    _stat=STAT_DEFINED

    def __init__(self,camera):
        super().__init__(camera)
        import_tensorflow()
        self.default_path=os.path.abspath(__file__+"/../model/Track_Follow/")
        if isnotebook:
            self.probWidget = widgets.Image(format='jpeg', width=camera.width, height=camera.height)

    def _load_layers(self):
        class IvtBotn_ResBlock(tf.keras.layers.Layer):
            def __init__(self, out_channel, kernel_size=3, stride=1, activation=None, padding='valid'):
                super().__init__()

                self.out_channel=out_channel
                self.kernel_size=kernel_size
                self.stride=stride
                self.activation=activation
                self.padding=padding

                self.conv0=Conv2D(self.out_channel, 1, strides=1, activation=self.activation, padding=self.padding)
                self.norm1=BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)
                self.conv1=Conv2D(self.out_channel, self.kernel_size, strides=self.stride, activation=self.activation, padding=self.padding)
                self.botn=Conv2D(self.out_channel*4, 1, strides=1, activation=self.activation, padding='same')
                self.conv2=Conv2D(self.out_channel, self.kernel_size, strides=self.stride, activation=self.activation, padding=self.padding)
                self.norm2=BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)

            def call(self, x):
                org = self.conv0(x)
                org = self.norm1(org)
                out = self.conv1(org)
                out = self.botn(out)
                out = self.conv2(out)
                out = self.norm2(out)

                out = Add()([out, org])

                return out

            def get_config(self):
                config = super().get_config().copy()
                config.update({
                    'out_channel': self.out_channel,
                    'kernel_size': self.kernel_size,
                    'stride': self.stride,
                    'activation': self.activation,
                    'padding': self.padding,
                })
                return config

#         inputs=Input((150,300,3))
#         norm=BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(inputs)
#         block1=IvtBotn_ResBlock(16, 5, activation='relu', padding='same')(norm)
#         maxpool1=MaxPool2D(2)(block1)
#         block2=IvtBotn_ResBlock(16, 5, activation='relu', padding='same')(maxpool1)
#         maxpool2=MaxPool2D(2)(block2)
#         block3=IvtBotn_ResBlock(16, 5, activation='relu', padding='same')(maxpool2)
#         maxpool3=MaxPool2D(2)(block3)
#         block4=IvtBotn_ResBlock(16, 5, activation='relu', padding='same')(maxpool3)
#         maxpool4=MaxPool2D(2)(block4)
#         flat=Flatten()(maxpool4)

#         dense1=Dense(512, activation="relu")(flat)
#         dense2=Dense(256, activation="relu")(dense1)
#         dense3=Dense(128, activation="relu")(dense2)
#         dense4=Dense(64, activation="relu")(dense3)
#         outputs=Dense(2, activation="sigmoid")(dense4)

#         self.model=Model(inputs=inputs, outputs=outputs)

        '''input1 = keras.layers.Input(shape=(300, 300, 3,))
        conv1 = keras.layers.Conv2D(filters=64, kernel_size=(7, 7), strides=(2, 2), padding="same", activation="relu")(input1)
        norm1 = keras.layers.BatchNormalization(axis=3)(conv1)
        actv1 = keras.layers.Activation("relu")(norm1)
        pool1 = keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(actv1)
        conv2 = keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(2, 2), padding="same", activation="relu")(pool1)
        norm2 = keras.layers.BatchNormalization(axis=3)(conv2)
        actv2 = keras.layers.Activation("relu")(norm2)
        conv3 = keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding="same", activation="relu")(actv2)
        drop1 = keras.layers.Dropout(rate=0.3)(conv3)
        norm3 = keras.layers.BatchNormalization(axis=3)(drop1)
        actv2 = keras.layers.Activation("relu")(norm3)
        add1 = keras.layers.Add()([actv2, conv2])
        conv4 = keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(2, 2), padding="same", activation="relu")(add1)
        norm4 = keras.layers.BatchNormalization(axis=3)(conv4)
        actv3 = keras.layers.Activation("relu")(norm4)
        conv5 = keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding="same", activation="relu")(actv3)
        drop2 = keras.layers.Dropout(rate=0.3)(conv5)
        norm5 = keras.layers.BatchNormalization(axis=3)(drop2)
        actv4 = keras.layers.Activation("relu")(norm5)
        add2 = keras.layers.Add()([actv4, conv4])
        conv6 = keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(2, 2), padding="same", activation="relu")(add2)
        norm6 = keras.layers.BatchNormalization(axis=3)(conv6)
        actv5 = keras.layers.Activation("relu")(norm6)
        conv7 = keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding="same", activation="relu")(actv5)
        drop3 = keras.layers.Dropout(rate=0.3)(conv7)
        norm7 = keras.layers.BatchNormalization(axis=3)(drop3)
        actv6 = keras.layers.Activation("relu")(norm7)
        add3 = keras.layers.Add()([actv6, conv6])
        pool2 = keras.layers.AveragePooling2D(pool_size=(3, 3), strides=(2, 2))(add3)
        flat1 = keras.layers.Flatten(data_format="channels_last")(pool2)
        dense1 = keras.layers.Dense(2, activation="tanh")(flat1)
        # dense1 = keras.layers.Dense(1, activation="tanh")(flat1)
        self.model = keras.models.Model(inputs=input1, outputs=dense1)'''
###############################################################################################################################
        input1 = keras.layers.Input(shape=(150, 300, 3,))
        conv1 = keras.layers.Conv2D(filters=16, kernel_size=(3, 3), strides=(2, 2), padding="same", activation="swish")(input1)
        norm1 = keras.layers.BatchNormalization()(conv1)
        pool1 = keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(norm1)
        conv2 = keras.layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(2, 2), padding="same", activation="swish")(pool1)
        norm2 = keras.layers.BatchNormalization()(conv2)
        conv3 = keras.layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding="same", activation="swish")(norm2)
        norm3 = keras.layers.BatchNormalization()(conv3)
        add1 = keras.layers.Add()([norm2, norm3])
        conv4 = keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(2, 2), padding="same", activation="swish")(add1)
        norm4 = keras.layers.BatchNormalization()(conv4)
        conv5 = keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding="same", activation="swish")(norm4)
        norm5 = keras.layers.BatchNormalization()(conv5)
        add2 = keras.layers.Add()([norm4, norm5])
        conv6 = keras.layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(2, 2), padding="same", activation="swish")(add2)
        norm6 = keras.layers.BatchNormalization()(conv6)
        conv7 = keras.layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding="same", activation="swish")(norm6)
        norm7 = keras.layers.BatchNormalization()(conv7)
        add3 = keras.layers.Add()([norm6, norm7])
        conv8 = keras.layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(2, 2), padding="same", activation="swish")(add3)
        norm7 = keras.layers.BatchNormalization()(conv8)
        conv9 = keras.layers.Conv2D(filters=512, kernel_size=(3, 3), strides=(2, 2), padding="same", activation="swish")(norm7)
        norm8 = keras.layers.BatchNormalization()(conv9)
        flat1 = keras.layers.Flatten()(norm8)
        dense1 = keras.layers.Dense(128, activation="swish")(flat1)
        norm9 = keras.layers.BatchNormalization()(dense1)
        dense2 = keras.layers.Dense(64, activation="swish")(norm9)
        norm10 = keras.layers.BatchNormalization()(dense2)
        dense3 = keras.layers.Dense(64, activation="swish")(norm10)
        norm11 = keras.layers.BatchNormalization()(dense3)
        dense4 = keras.layers.Dense(2, activation="tanh")(norm11)
        self.model = keras.models.Model(inputs=input1, outputs=dense4)

    def load_datasets(self, path=dataset_path):
        if isnotebook:
            datasets_noti_widget = widgets.Label(value="Loading datasets...")
            model_noti_widget = widgets.Label(value="Creating a new model...")

            display(datasets_noti_widget)

        if not os.path.exists(path):
            print(path," doesn't exist.")
            return

        # width=self.camera.width
        # height=self.camera.height

        file_list=os.listdir(path)

        X=[]
        Y=[]

        for f in file_list:
            if ".jpg" in f.lower():
                img=cv2.imread(path+"/"+f)
                height, width, _ = img.shape

                if img is not None:
                    if width!=300 and height!=300: img=cv2.resize(img,(300,300))

                    s=f.split("_")

                    a=(int(s[0])/width-0.5)*2
                    b=(int(s[1])/height-0.5)*2

                    X.append(img)
                    Y.append([a,b])

            self.X=np.array(X)
            self.Y=np.array(Y)

        if len(X)<=0:
            if isnotebook:
                datasets_noti_widget.value="Can't access datasets. Check file permission or existence."
            else:
                print("Can't access datasets. Check file permission or existence.")
            return
        else:
            if isnotebook:
                datasets_noti_widget.value="Loaded "+str(len(X))+" datasets."
            else:
                print("Loaded "+str(len(X))+" datasets.")

            if self.model is None:
                if isnotebook:
                    display(model_noti_widget)

                self._load_layers()

                decay=schedules.ExponentialDecay(1.0e-04, decay_steps=800, decay_rate=0.96, staircase=True)
                adam=Adam(learning_rate=decay, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False)

                self.model.compile(optimizer=adam, loss='MAE')

            if isnotebook:
                model_noti_widget.value="Model creation completed."
            else:
                print("Model creation completed.")

            self._stat=self.STAT_READY

    def train(self, times=5, autosave=True):
        if self._stat < self.STAT_READY :
            print("Please load datasets as load_datasets() method.")
            return

        if isnotebook:
            totally_progress_widget = widgets.FloatProgress(min=0.0, max=100.0, description='Total')
            progress_widget = widgets.FloatProgress(min=0.0, max=100.0, description='This step')
            total_percentage_widget = widgets.Label(value="0%")
            current_percentage_widget = widgets.Label(value="0%")
            remaining_time_widget = widgets.Label(value="0")
            loss_widget = widgets.Label(value="0")

            row1=widgets.HBox([totally_progress_widget, total_percentage_widget])
            row2=widgets.HBox([progress_widget, current_percentage_widget])
            row3=widgets.HBox([widgets.Label(value="Remaining"), remaining_time_widget, widgets.Label(value="sec")])
            row4=widgets.HBox([widgets.Label(value="Loss : "), loss_widget])

            display(widgets.VBox([row1, row2, row3, row4]))

            class Train_Progress(callbacks.Callback):
                timelog=0

                def __init__(self, batch_size, data_size, train_size):
                    self.batch_size=batch_size
                    self.data_size=data_size
                    self.train_size=train_size

                    self.last_epoch_log=time.time()
                    self.epoch=0

                    super().__init__()

                def on_train_begin(self, logs=None):
                    self.train_timelog=time.time()

                def on_epoch_begin(self, epoch, logs=None):
                    self.epoch_timelog=time.time()

                def on_epoch_end(self, epoch, logs=None):
                    self.epoch=epoch+1
                    self.last_epoch_log=time.time()

                    print(str(epoch+1),' epoch loss : '+str(logs['loss']))

                def on_train_batch_begin(self, batch, logs=None):
                    self.batch_timelog=time.time()

                def on_train_batch_end(self, batch, logs=None):
                    batch+=1
                    #Epoch ETA

                    spent_time_batches=time.time()-self.epoch_timelog

                    remain_data=self.data_size-(batch*self.batch_size)
                    time_per_sample=spent_time_batches/(batch*self.batch_size)

                    epoch_eta=max(remain_data*time_per_sample, 0)
                    epoch_progress_rate=min(round((float(batch*self.batch_size)/self.data_size)*100 ,1), 100.0)


                    #Train ETA

                    if self.epoch>0:
                        time_per_epoch=float(self.last_epoch_log-self.train_timelog)/self.epoch
                    else:
                        time_per_epoch=self.data_size*time_per_sample

                    remained_time_epochs=(self.train_size-self.epoch-1)*time_per_epoch

                    train_eta=max(remained_time_epochs+epoch_eta, 0)
                    train_progress_rate=min(round((1-train_eta/(self.train_size*time_per_epoch))*100, 1), 100.0)

                    if isnotebook:
                        current_percentage_widget.value=str(epoch_progress_rate)+"%"
                        progress_widget.value=epoch_progress_rate

                        total_percentage_widget.value=str(train_progress_rate)+"%"
                        totally_progress_widget.value=train_progress_rate

                        remaining_time_widget.value=str(round(train_eta, 3))

                        loss_widget.value=str(round(logs['loss'], 7))

            self.model.fit(self.X, self.Y, batch_size=2, epochs=times, callbacks=[Train_Progress(16, len(self.Y), times)], verbose=0)

            if autosave:
                self.model.save(self.MODEL_PATH)

    def load_model(self,path=MODEL_PATH):
        if not os.path.exists(path):
            print(path," doesn't exist.")
            return

        if self.model is None:
            self._load_layers()
            decay=schedules.ExponentialDecay(1.0e-04, decay_steps=800, decay_rate=0.96, staircase=True)
            adam=Adam(learning_rate=decay, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False)
            self.model.compile(optimizer=adam, loss='MAE')

        self.model.load_weights(path)

    def save_model(self,path=MODEL_PATH):
        if path[-3:] != '.h5':
            path = path + '.h5'
        if self.model is not None:
            self.model.save_weights(path)
            print("Save completed.")
        else:
            print("The model can't be saved cause it's None.")

    def show(self):
        display(self.probWidget)

    def run(self, value=None, callback=None):
        if self.model is None :
            print("Please load datasets as load_datasets() method or load a trained model as load_model() method.")
            return

        img = self.camera.value if value is None else value
        x, y = self.model(np.array([img]).astype(np.float32)).numpy()[0]
        # x = self.model(np.array([img]).astype(np.float32)).numpy()[0][0]
        # cX = int(self.camera.width * x)
        # cY = int(self.camera.height * y)
        height, width, _ = img.shape
        #cX = int(width * (x / 2.0 + 0.5))
        #cY = int(height * (y / 2.0 + 0.5))
        cX = int(width * (x))
        cY = int(height * (y))

        self.value = cv2.circle(img, (cX, cY), 6, (255, 255, 255), 2)
        # self.value = cv2.circle(img, (cX, 50), 6, (255, 255, 255), 2)
        if isnotebook:
            self.probWidget.value = bgr8_to_jpeg(self.value)

            result={"x":x,"y":y,"cx":cX,"cy":cY}
            # result={"x":x,"cx":cX}

            if callback is not None:
                callback(result)

            return result
        
    def run_TF(self, value=None, callback=None):
        if self.model is None :
            print("Please load datasets as load_datasets() method or load a trained model as load_model() method.")
            return
        img = self.camera.value if value is None else value
        crop_img = img[130:280,:300]
        x, y = self.model(np.array([crop_img]).astype(np.float32)).numpy()[0]
        height, width,_ = crop_img.shape
        cX = int(width * (x))
        cY = int(height * (y))

        self.value = cv2.circle(img, (cY, 150), 6, (255, 0, 0), 2)
        if isnotebook:
            self.probWidget.value = bgr8_to_jpeg(self.value)

            result={"x":x,"y":y,"cx":cX,"cy":cY}

            if callback is not None:
                callback(result)

            return result
#---------------------------------------------  Object Recognition TRT Model for OF ---------------------------------------------


import ctypes
import subprocess
import tensorrt as trt

TRT_INPUT_NAME = 'input'
TRT_OUTPUT_NAME = 'nms'
LABEL_IDX = 1
CONFIDENCE_IDX = 2
X0_IDX = 3
Y0_IDX = 4
X1_IDX = 5
Y1_IDX = 6

def parse_boxes(outputs):
    bboxes = outputs[0]
            
    all_detections = []
    for i in range(bboxes.shape[0]):

        detections = []
        for j in range(bboxes.shape[2]):

            bbox = bboxes[i][0][j]
            label = bbox[LABEL_IDX]

            if label < 0: 
                break

            detections.append(dict(
                label=int(label),
                confidence=float(bbox[CONFIDENCE_IDX]),
                bbox=[
                    float(bbox[X0_IDX]),
                    float(bbox[Y0_IDX]),
                    float(bbox[X1_IDX]),
                    float(bbox[Y1_IDX])
                ]
            ))

        all_detections.append(detections)

    return all_detections

def load_plugins():
    library_path = os.path.join(os.path.dirname(__file__), 'libssd_tensorrt.so')
    ctypes.CDLL(library_path)


class TRTModel(object):
    
    def __init__(self, engine_path, input_names=None, output_names=None, final_shapes=None):
        
        # load engine
        self.logger = trt.Logger()
        self.runtime = trt.Runtime(self.logger)
        with open(engine_path, 'rb') as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        
        if input_names is None:
            self.input_names = self._trt_input_names()
        else:
            self.input_names = input_names
            
        if output_names is None:
            self.output_names = self._trt_output_names()
        else:
            self.output_names = output_names
            
        self.final_shapes = final_shapes
        
        # destroy at exit
        atexit.register(self.destroy)
    
    def _input_binding_indices(self):
        return [i for i in range(self.engine.num_bindings) if self.engine.binding_is_input(i)]
    
    def _output_binding_indices(self):
        return [i for i in range(self.engine.num_bindings) if not self.engine.binding_is_input(i)]
    
    def _trt_input_names(self):
        return [self.engine.get_binding_name(i) for i in self._input_binding_indices()]
    
    def _trt_output_names(self):
        return [self.engine.get_binding_name(i) for i in self._output_binding_indices()]
    
    def torch_dtype_from_trt(dtype):
        if dtype == trt.int8:
            return torch.int8
        elif dtype == trt.int32:
            return torch.int32
        elif dtype == trt.float16:
            return torch.float16
        elif dtype == trt.float32:
            return torch.float32
        else:
            raise TypeError('%s is not supported by torch' % dtype)

    def torch_device_from_trt(device):
        if device == trt.TensorLocation.DEVICE:
            return torch.device('cuda')
        elif device == trt.TensorLocation.HOST:
            return torch.device('cpu')
        else:
            return TypeError('%s is not supported by torch' % device)

    def create_output_buffers(self, batch_size):
        outputs = [None] * len(self.output_names)
        for i, output_name in enumerate(self.output_names):
            idx = self.engine.get_binding_index(output_name)
            dtype = torch_dtype_from_trt(self.engine.get_binding_dtype(idx))
            if self.final_shapes is not None:
                shape = (batch_size, ) + self.final_shapes[i]
            else:
                shape = (batch_size, ) + tuple(self.engine.get_binding_shape(idx))
            device = torch_device_from_trt(self.engine.get_location(idx))
            output = torch.empty(size=shape, dtype=dtype, device=device)
            outputs[i] = output
        return outputs
    
    def execute(self, *inputs):
        batch_size = inputs[0].shape[0]
        
        bindings = [None] * (len(self.input_names) + len(self.output_names))
        
        # map input bindings
        inputs_torch = [None] * len(self.input_names)
        for i, name in enumerate(self.input_names):
            idx = self.engine.get_binding_index(name)
            
            # convert to appropriate format
            inputs_torch[i] = torch.from_numpy(inputs[i])
            inputs_torch[i] = inputs_torch[i].to(torch_device_from_trt(self.engine.get_location(idx)))
            inputs_torch[i] = inputs_torch[i].type(torch_dtype_from_trt(self.engine.get_binding_dtype(idx)))
            
            bindings[idx] = int(inputs_torch[i].data_ptr())
            
        output_buffers = self.create_output_buffers(batch_size)
        
        # map output bindings
        for i, name in enumerate(self.output_names):
            idx = self.engine.get_binding_index(name)
            bindings[idx] = int(output_buffers[i].data_ptr())
        
        self.context.execute(batch_size, bindings)
        
        outputs = [buffer.cpu().numpy() for buffer in output_buffers]

        return outputs
    
    def __call__(self, *inputs):
        return self.execute(*inputs)

    def destroy(self):
        self.runtime.destroy()
        self.logger.destroy()
        self.engine.destroy()
        self.context.destroy()

mean = 255.0 * np.array([0.5, 0.5, 0.5])
stdev = 255.0 * np.array([0.5, 0.5, 0.5])

def bgr8_to_ssd_input(camera_value):
    x = camera_value
    x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
    x = x.transpose((2, 0, 1)).astype(np.float32)
    x -= mean[:, None, None]
    x /= stdev[:, None, None]
    return x[None, ...]


class ObjectDetector(object):    
    def __init__(self, engine_path, preprocess_fn=bgr8_to_ssd_input):
        logger = trt.Logger()
        trt.init_libnvinfer_plugins(logger, '')
        load_plugins()
        self.trt_model = TRTModel(engine_path, input_names=[TRT_INPUT_NAME], output_names=[TRT_OUTPUT_NAME, TRT_OUTPUT_NAME + '_1'])
        self.preprocess_fn = preprocess_fn
        
    def execute(self, *inputs):
        trt_outputs = self.trt_model(self.preprocess_fn(*inputs))
        return parse_boxes(trt_outputs)
    
    def __call__(self, *inputs):
        return self.execute(*inputs)


class Pose_Estimation():
    def __init__(self, camera=None):
        import cv2, os, json, torch, torchvision, torch2trt, time
        import trt_pose.plugins as plugins
        import PIL.Image as Image
        import ipywidgets.widgets as widgets
        from IPython.display import display
        global cv2, os, json, torch, torchvision, torch2trt, time, plugins, Image, widgets, display

        print("Model load started.")
        log = time.time()
        
        self.cam=camera
        self.model_path=os.path.abspath(__file__+"/../model/trtpose/")
        if isnotebook:
            self.probWidget=widgets.Image(format="jpeg", width=self.cam.width, height=self.cam.height)
        
        json_path=self.model_path + '/human_pose.json'
        with open(json_path, 'r') as f:
            human_pose = json.load(f)
            
        skeleton = human_pose['skeleton']
        K = len(skeleton)
        self.topology = torch.zeros((K, 4)).int()
        for k in range(K):
            self.topology[k][0] = 2 * k
            self.topology[k][1] = 2 * k + 1
            self.topology[k][2] = skeleton[k][0] - 1
            self.topology[k][3] = skeleton[k][1] - 1
        
        self.WIDTH = 224
        self.HEIGHT = 224
        
        OPTIMIZED_MODEL = self.model_path + '/resnet18_baseline_att_224x224_A_epoch_249_trt.pth'
        
        self.model_trt = torch2trt.TRTModule()
        self.model_trt.load_state_dict(torch.load(OPTIMIZED_MODEL))
        
        self.mean = torch.Tensor([0.485, 0.456, 0.406]).cuda()
        self.std = torch.Tensor([0.229, 0.224, 0.225]).cuda()
        
        print(f"Model load completed. ({round(time.time()-log, 2)}s)")

    def __image(self, value):
        if value is None:
            if self.cam is None:
                raise ValueError("No images or cameras were found on this object.")
            else:
                value=self.cam.value
        else:
            if type(value) is str:
                value=cv2.imread(value)
                if value is None:
                    raise ValueError("This path is not available.")
            elif type(value) in (list, tuple, np.ndarray):
                value=np.array(value)
            else:
                raise ValueError("This data is not available.")  

        if value.size>0:
            value = cv2.resize(value, dsize=(self.WIDTH, self.HEIGHT), interpolation=cv2.INTER_AREA)
            return value
        else:
            raise ValueError("Image is empty")

    def __preprocess(self, value):
        try:
            device = torch.device('cuda')
            image = cv2.cvtColor(value, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
            image = torchvision.transforms.functional.to_tensor(image).to(device)
            image.sub_(self.mean[:, None, None]).div_(self.std[:, None, None])
            return image[None, ...]
        except:
            raise ValueError("Image is empty")

    def __bgr8_to_jpeg(self, value):
        return bytes(cv2.imencode('.jpg', value)[1])
    
    def __ParseObjects(self, topology, cmap, paf, cmap_threshold=0.1, link_threshold=0.1, cmap_window=5, line_integral_samples=7, max_num_parts=100, max_num_objects=100):
        peak_counts, peaks = plugins.find_peaks(cmap, cmap_threshold, cmap_window, max_num_parts)
        normalized_peaks = plugins.refine_peaks(peak_counts, peaks, cmap, cmap_window)
        score_graph = plugins.paf_score_graph(paf, topology, peak_counts, normalized_peaks, line_integral_samples)
        connections = plugins.assignment(score_graph, topology, peak_counts, link_threshold)
        object_counts, objects = plugins.connect_parts(connections, topology, peak_counts, max_num_objects)
        
        return object_counts, objects, normalized_peaks

    def __DrawObjects(self, topology, parseobjects, image):
        height = self.HEIGHT
        width = self.WIDTH
        object_counts, objects, normalized_peaks = parseobjects
        
        K = self.topology.shape[0]
        count = int(object_counts[0])
        K = self.topology.shape[0]
        for i in range(count):
            color = (0, 255, 0)
            obj = objects[0][i]
            C = obj.shape[0]
            for j in range(C):
                k = int(obj[j])
                if k >= 0:
                    peak = normalized_peaks[0][j][k]
                    x = round(float(peak[1]) * width)
                    y = round(float(peak[0]) * height)
                    cv2.circle(image, (x, y), 3, color, 2)                   

            for k in range(K):
                c_a = self.topology[k][2]
                c_b = self.topology[k][3]
                if obj[c_a] >= 0 and obj[c_b] >= 0:
                    peak0 = normalized_peaks[0][c_a][obj[c_a]]
                    peak1 = normalized_peaks[0][c_b][obj[c_b]]
                    x0 = round(float(peak0[1]) * width)
                    y0 = round(float(peak0[0]) * height)
                    x1 = round(float(peak1[1]) * width)
                    y1 = round(float(peak1[0]) * height)
                    cv2.line(image, (x0, y0), (x1, y1), color, 2)

        return image

    def detect(self, value=None):
        image = self.__image(value)
        data = self.__preprocess(image)
        cmap, paf = self.model_trt(data)
        cmap, paf = cmap.detach().cpu(), paf.detach().cpu()
        parse_objects = self.__ParseObjects(self.topology, cmap, paf)
        draw_object = self.__DrawObjects(self.topology, parse_objects, image)
        if isnotebook:
            self.probWidget.value = self.__bgr8_to_jpeg(draw_object[:, ::-1, :])
        else:
            self.valud = draw_object[:, ::-1, :]

        ret = []
        if parse_objects:
            ret = []
            for i in range(parse_objects[0]):
                tmp = []
                for j in range(18):
                    tmp.append([round(float(parse_objects[2][0][j][i][1]),3), round(float(parse_objects[2][0][j][i][0]),3)])
                ret.append(tmp)
            return ret

    def show(self):
        display(self.probWidget)
