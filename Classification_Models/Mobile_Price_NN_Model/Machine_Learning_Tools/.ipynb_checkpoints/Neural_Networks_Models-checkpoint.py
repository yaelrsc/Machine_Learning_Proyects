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
                           dropout_rate=None,momentum_bn=None,epsilon_bn=None,**kwargs):
    
    clear_session()
    
    inputs = Input(shape=input_shape)
    
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

def convolutinal_neural_network(input_shape,n_outputs,n_conv_maxpol_layers=2,n_filters=50,kernel_size=(2,2),strides=(1,1),
                                padding='valid',pool_size=(2,2),pool_strides=None,n_neurons=100,n_hidden_layers=2,act='relu',
                                out_act='softmax',optimizer=None,loss=None,metrics=None,dropout=False,batch_normalization=False,
                                dropout_rate=None,momentum_bn=None,epsilon_bn=None,pool_kind='MAX',**kwargs):
    
    clear_session()
    
    inputs = Input(shape=input_shape)
    
    for k in range(n_conv_maxpol_layers):
        
        if k == 0:
            
            x = layers.Conv2D(n_filters,kernel_size,strides,padding,**kwargs)(inputs)
            
        else:
            
            x = layers.Conv2D(n_filters,kernel_size,strides,padding,**kwargs)(x)
        
        if pool_kind == 'MAX':
            
            x = layers.MaxPooling2D(pool_size,pool_strides,padding)(x)
        
        elif pool_kind == 'AVG':
            
            x = layers.AveragePooling2D(pool_size,pool_strides,padding)(x)
            
    
        x = layers.Activation(act)(x)
        
        if dropout:
            
            x = layers.Dropout(dropout_rate)(x)
            
        if batch_normalization:
            
            x = layers.BatchNormalization(momentum=momentum_bn,epsilon=epsilon_bn)(x)
        
    x = layers.Flatten()(x)
        
        
    for i in range(n_hidden_layers):
        
        x = layers.Dense(n_neurons,activation=act,**kwargs)(x)
        
        if dropout:
            
            x = layers.Dropout(dropout_rate)(x)
            
        if batch_normalization:
            
            x = layers.BatchNormalization(momentum=momentum_bn,epsilon=epsilon_bn)(x)
                 
        
    outputs = layers.Dense(n_outputs,activation=out_act,**kwargs)(x)
    
    model = Model(inputs=inputs,outputs=outputs)
    
    model.compile(optimizer,loss,metrics)
    
    return model

def cross_val_metrics(model,X,y,n_folds=10,seed=None):
    
    kfold = KFold(n_splits=n_folds,shuffle=True,random_state=seed)
    cv_train_metrics = []
    cv_test_metrics = []
        
    for train_index, test_index in kfold.split(self.X.values):
        
        X_train, X_test = X[train_index,:], X[test_index,:]
        y_train, y_test = y[train_index,:], y[test_index,:]
        
        c_model = clone_model(model)

        c_model.fit(X_train,y_train,verbose=0)

        cv_train_metrics.append(c_model.evaluate(X_train,y_train))
        cv_test_metrics.append(c_model.evaluate(X_test,y_test))
    
    return cv_train_metrics, cv_test_metrics
    