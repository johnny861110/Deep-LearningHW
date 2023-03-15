import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense
from tensorflow.keras import Model
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Flatten
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.initializers import RandomNormal, GlorotNormal
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import LearningRateScheduler
import numpy as np
from kerastuner.tuners import BayesianOptimization
print("TensorFlow version:",tf.__version__)

from tensorflow.python.client import device_lib


#construct model stucture
class MyModel(Model):
    def __init__(self, activation_fn, hidden_node,
                  init_weight, optimizer_type,
                    regularization_coefficients, num_class):
        super(MyModel,self).__init__()
        self.activation_fn = activation_fn
        self.hidden_node = hidden_node
        self.init_weight = init_weight
        self.optimizer_type = optimizer_type
        self.regularization_coefficients = regularization_coefficients
        self.num_class = num_class
        
        # Define input Layer
        
        
        
        # Define the layers
        self.flatten_layer = Flatten()
        self.hidden_layer = Dense(units = self.hidden_node,
                                  activation = self.activation_fn,
                                  kernel_initializer = self.init_weight,
                                  kernel_regularizer = l2(self.regularization_coefficients))
        
        # Define output layer
        self.output_layer = Dense(units = self.num_class,activation = 'softmax')
    
    def call(self,inputs):
        
        out = self.flatten_layer(inputs)
        out = self.hidden_layer(out)
        out = self.output_layer(out)
        return out

def build_model(hp):
    activation_fn = hp.Choice('activation_fn', values=['tanh', 'relu'])
    hidden_node = hp.Choice('hidden_node', values=[5, 8, 11])
    init_weight = hp.Choice('init_weight', values=['random_normal', 'glorot_uniform', 'he_uniform'])
    optimizer_type = hp.Choice('optimizer_type', values=['SGD', 'Adam','Momenton'])
    regularization_coefficients = hp.Choice('regularization_coefficients', values=[0.001, 0.0001])
    learning_rate = hp.Float('learning_rate', min_value=0.0001, max_value=0.1, step=0.099)
    decay_schedule = hp.Choice('decay_schedule', values=['none', 'cosine'])
    loss_function = hp.Choice('loss_function', values=['categorical_crossentropy', 'mean_squared_error'])

    # define the optimizer
    if optimizer_type == 'Momenton':
        optimizer = keras.optimizers.SGD(learning_rate = learning_rate, momenton=0.9)
    elif optimizer_type == 'Adam':
        optimizer = keras.optimizers.Adam(learning_rate = learning_rate)
    elif optimizer_type == 'SGD':
        optimizer = keras.optimizers.SGD(learning_rate = learning_rate)
    
    #define the lr decay schedule
    if decay_schedule == 'none':
        learning_rate_decay = 0.0
    elif decay_schedule == 'cosine':
        learning_rate_decay = keras.experimental.CosineDecay(initial_learning_rate = learning_rate, decay_steps = 10000)
    
    # build the model
    model = MyModel(activation_fn,hidden_node,init_weight,optimizer,regularization_coefficients,num_class=10)
    model.compile(optimizer=optimizer,loss=loss_function,metrics = ['accuracy'])
    return model
