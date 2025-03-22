try:
    import cupy as np
    is_cupy_available = True
except:
    import numpy as np
    is_cupy_available = False

from transformer.activations import ReLU
from transformer.layers.base.dense import Dense
from transformer.layers.base.dropout import Dropout

class PositionwiseFeedforward():
    def __init__(self, d_model = 512, d_ff = 2048, dropout = 0.1, data_type = np.float32):
        self.fc_1 = Dense(inputs_num = d_model, units_num = d_ff, data_type = data_type)
        self.activation = ReLU()
        self.fc_2 = Dense(inputs_num = d_ff, units_num = d_model, data_type = data_type)
        self.dropout = Dropout(dropout, data_type)

    def forward(self, X, training = True):
        
        X = self.fc_1.forward(X)
        X = self.activation.forward(X)
        X = self.dropout.forward(X, training)
        X = self.fc_2.forward(X)
    
        return X

    def backward(self, error):
        error = self.fc_2.backward(error)
        error = self.dropout.backward(error)
        error = self.activation.backward(error)
        error = self.fc_1.backward(error)
        return error

    def set_optimizer(self, optimizer):
        self.fc_1.set_optimizer(optimizer)
        self.fc_2.set_optimizer(optimizer)

    def update_weights(self, layer_num):
        layer_num = self.fc_1.update_weights(layer_num)
        layer_num = self.fc_2.update_weights(layer_num)

        return layer_num