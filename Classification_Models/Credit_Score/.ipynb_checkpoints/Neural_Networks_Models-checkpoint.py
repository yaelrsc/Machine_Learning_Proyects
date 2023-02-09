from tensorflow.keras import (layers,Model,Input)
from tensorflow.keras.backend import clear_session
from tensorflow.keras.models import clone_model

def single_layer_perceptron(input_shape,n_outputs,optimizer,loss,metrics=None,**kwargs):
    
    clear_session()
    
    inputs = Input(shape=input_shape)
    outputs = layers.Dense(n_outputs,**kwargs)(inputs)
    
    model = Model(inputs=inputs,outputs=outputs)
    
    model.compile(optimizer,loss,metrics)
    
    return model

def multi_layer_perceptron(input_shape,n_outputs,n_neurons,n_hidden_layers,act,out_act,
                           optimizer,loss,metrics=None,dropout=False,batch_normalization=False,
                           dropout_rate=None,momentum_bn=None,epsilon_bn=None,rescaling=True,
                           data_argumentation=None,**kwargs):
    
    clear_session()
    
    inputs = Input(shape=input_shape)
    
    if data_argumentation != None:
        
        for da in data_argumentation:
            
            inputs = da(inputs)
        
    
    if rescaling:
        
        x = layers.Rescaling(1.0 / 255)(inputs)
    
    for i in range(n_hidden_layers):
        
        if i == 0:
            
            x = layers.Dense(n_neurons,activation=act,**kwargs)(inputs)
            
        else:
            
            x = layers.Dense(n_neurons,activation=act,**kwargs)(x)
        
        if dropout:
            
            x = layers.Dropout(dropout_rate)(x)
            
        if batch_normalization:
            
            x = layers.BatchNormalization(momentum=momentum_bn,epsilon=epsilon_bn)(x)
            
        
        
    outputs = layers.Dense(n_outputs,activation=out_act,**kwargs)(x)
    
    model = Model(inputs=inputs,outputs=outputs)
    
    model.compile(optimizer,loss,metrics)
    
    return model

