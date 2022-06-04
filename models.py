import tensorflow as tf
import tensorflow_addons as tfa
from typing import Union, Iterable
from tensorflow_addons.layers.adaptive_pooling import AdaptiveAveragePooling3D

##########################  Layer Class ######################################    
class AEDmodule(tf.keras.layers.Layer):
    def __init__(self,
                 units=32,
                 activation=None,
                 initializer="glorot_unitform",
                 regularizer=None,
                 name=None,
                 **kwargs):
        super(AEDmodule, self).__init__(name=name, **kwargs)
        self.units       = units
        self.activation  = tf.keras.activations.get(activation)
        self.initializer = tf.keras.initializers.get(initializer)
        self.regularizer = tf.keras.regularizers.get(regularizer)
        
        self.pool        = tf.keras.layers.GlobalAveragePooling1D(name=f"{self.name}_gp")
    
    def build(self, input_shape):
        self.input_features = input_shape[-1]
        self.w = self.add_weight(
            shape=(self.units, self.units),
            initializer=self.initializer,
            regularizer=self.regularizer,
            trainable=True,
            name=f"{self.name}_weight"
        )
        
        self.u = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer=self.initializer,
            regularizer=self.regularizer,
            trainable=True,
            name=f"{self.name}_power"
        )
        
        self.b = self.add_weight(
            shape=(self.units,),
            initializer="zeros",
            trainable=True,
            name=f"{self.name}_bias"
        )
        
        self.built = True
    
    def call(self, inputs):
        x = tf.math.exp(-tf.matmul(inputs, self.u))
        x = tf.matmul(x, self.w) + self.b
        x = self.activation(x)
        x = tf.expand_dims(x, -1)
        x = self.pool(x)
        x = tf.tile(x, [1,self.input_features])
        
        return x
    
    def compute_output_shape(self, input_shape):
        output_shape = list(input_shape)
        output_shape[-1] = 1
        return tuple(output_shape)
        
    def get_config(self):
        config = {
            'units': self.units,
            'activation':tf.keras.activations.serialize(self.activation),
            'initializer': tf.keras.initializers.serialize(self.initializer),
            'regularizer': tf.keras.regularizers.serialize(self.regularizer),
        }
        base_config = super(AEDmodule, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
      
class StackDPDense(tf.keras.layers.Layer):
    def __init__(self,
                 units=32,
                 activation=None,
                 att_activation=None,
                 num_poly=3,
                 initializer="glorot_uniform",
                 regularizer=None,
                 name=None,
                 **kwargs):
        super(StackDPDense, self).__init__(name=name, **kwargs)
        self.units       = units
        self.num_poly    = num_poly
        self.activation  = tf.keras.activations.get(activation)
        self.att_activation = tf.keras.activations.get(att_activation)
        self.initializer = tf.keras.initializers.get(initializer)
        self.regularizer = tf.keras.regularizers.get(regularizer)
    
    def build(self, input_shape):
        self.first_block = tf.keras.layers.Dense(self.units,
                                               kernel_initializer=self.initializer,
                                               kernel_regularizer=self.regularizer,
                                               name=f"{self.name}_first_block")
        self.second_block = tf.keras.layers.Dense(self.units,
                                                  kernel_initializer=self.initializer,
                                                  kernel_regularizer=self.regularizer,
                                                  name=f"{self.name}_second_block")
        
        self.reshape_1 = tf.keras.layers.Reshape((self.units, 1), name=f"{self.name}_reshape_1")
        self.reshape_2 = tf.keras.layers.Reshape((self.units, 1), name=f"{self.name}_reshape_2")
        self.concat_1  = tf.keras.layers.Concatenate(axis=-1, name=f"{self.name}_concat_1")
        self.concat_2  = tf.keras.layers.Concatenate(axis=-1, name=f"{self.name}_concat_2")
        
        self.att_activate_1 = tf.keras.layers.Activation(self.att_activation, name=f"{self.name}_att_activate_1")
        self.att_activate_2 = tf.keras.layers.Activation(self.att_activation, name=f"{self.name}_att_activate_2")
        
        self.w1 = self.add_weight(
            shape=(3, 1),
            initializer=self.initializer,
            regularizer=self.regularizer,
            trainable=True,
            name=f"{self.name}_weight_1"
        )
        self.b1 = self.add_weight(
            shape=(1,),
            initializer="zeros",
            trainable=True,
            name=f"{self.name}_bias_1"
        )
        self.w2 = self.add_weight(
            shape=(3, 1),
            initializer=self.initializer,
            regularizer=self.regularizer,
            trainable=True,
            name=f"{self.name}_weight_2"
        )
        self.b2 = self.add_weight(
            shape=(1,),
            initializer="zeros",
            trainable=True,
            name=f"{self.name}_bias_2"
        )
                
        self.built = True
    
    def call(self, inputs):
        # first DP
        x_1st    = self.first_block(inputs)
        x_1st_re = self.reshape_1(x_1st)
        x_2nd    = x_1st_re * x_1st_re
        x_3rd    = x_2nd * x_1st_re
        x_poly   = self.concat_1([x_1st_re, x_2nd, x_3rd])
        x_poly   = tf.squeeze(tf.matmul(x_poly, self.w1)) + self.b1
        att   = self.att_activate_1(x_poly)
        x_att = x_1st * att
        
        # second DP
        x_1st    = self.second_block(x_att)
        x_1st_re = self.reshape_2(x_1st)
        x_2nd    = x_1st_re * x_1st_re
        x_3rd    = x_2nd * x_1st_re
        x_poly   = self.concat_2([x_1st_re, x_2nd, x_3rd])
        x_poly   = tf.squeeze(tf.matmul(x_poly, self.w2)) + self.b2
        att   = self.att_activate_2(x_poly)
        x_att = x_1st * att
        
        x = self.activation(x_att)
        
        return x       
    
    def compute_output_shape(self, input_shape):
        output_shape = list(input_shape)
        output_shape[-1] = self.units
        return tuple(output_shape)
    
    def get_config(self):
        config = {
            'units': self.units,
            'num_poly': self.num_poly,
            'activation': tf.keras.activations.serialize(self.activation),
            'att_activation': tf.keras.activations.serialize(self.att_activation),
            'initializer': tf.keras.initializers.serialize(self.initializer),
            'regularizer': tf.keras.regularizers.serialize(self.regularizer),
        }
        base_config = super(StackDPDense, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

##########################  Model Class ######################################

class ADSurv(tf.keras.Model):
    def __init__(self,
                 num_hidden=32,
                 num_outputs=1,
                 num_layers=1,
                 dropout=0.,
                 kernel_initializer="glorot_uniform",
                 l2_regularizer = 1e-5,
                 encode_model="DeepSurv",
                 hidden_activation=None,
                 att_activation=None,
                 output_activation=None,
                 input_name=None,
                 name=None,
                 **kwargs):
        super(ADSurv, self).__init__(name=name, **kwargs)
        self.input_name = input_name

        if encode_model=="SPAN":
            self.dense_block_1 = StackDPDense(units=int(num_hidden),
                                              activation=hidden_activation,
                                              att_activation=att_activation,
                                              initializer=kernel_initializer,
                                              name="dense_1")
            self.dropout_1 = tf.keras.layers.Dropout(dropout, name="dropout_1")
            self.dense_block_2 = StackDPDense(units=int(num_hidden/2),
                                              activation=hidden_activation,
                                              att_activation=att_activation,
                                              initializer=kernel_initializer,
                                              name="dense_2")
            self.dropout_2 = tf.keras.layers.Dropout(dropout/1.42857, name="dropout_2")
        elif encode_model=="DeepSurv":
            self.dense_block_1 = tf.keras.layers.Dense(units=int(num_hidden),
                                                       activation=hidden_activation,
                                                       kernel_initializer=kernel_initializer,
                                                       name="dense_1")
            self.dropout_1 = tf.keras.layers.Dropout(dropout, name="dropout_1")
            self.dense_block_2 = tf.keras.layers.Dense(units=int(num_hidden/2),
                                                       activation=hidden_activation,
                                                       kernel_initializer=kernel_initializer,
                                                       name="dense_2")
            self.dropout_2 = tf.keras.layers.Dropout(dropout/1.42857, name="dropout_2")
        self.hr_output_layer = tf.keras.layers.Dense(units=num_outputs,
                                                     activation=output_activation,
                                                     kernel_initializer=kernel_initializer,
                                                     kernel_regularizer=tf.keras.regularizers.l2(l2_regularizer),
                                                     name='hr_prediction')
        self.cl_output_layer = tf.keras.layers.Dense(units=num_outputs,
                                                     activation="sigmoid",
                                                     kernel_initializer=kernel_initializer,
                                                     kernel_regularizer=tf.keras.regularizers.l2(l2_regularizer),
                                                     name='cl_prediction')
        
    def call(self, inputs):
        x = inputs[self.input_name]
        x = self.dense_block_1(x)
        x = self.dropout_1(x)
        x = self.dense_block_2(x)
        x = self.dropout_2(x)
        
        x_hr = self.hr_output_layer(x)
        x_cl = self.cl_output_layer(x)
        
        return [x_hr, x_cl]        

class MultiClinRad(tf.keras.Model):
    def __init__(self,
                 num_hidden=32,
                 num_outputs=1,
                 dropout=0.,
                 kernel_initializer="glorot_uniform",
                 l2_regularizer = 1e-5,
                 encode_model='DeepSurv',
                 hidden_activation=None,
                 att_activation=None,
                 output_activation=None,
                 input_names=None,
                 fusion_type=None,
                 name=None,
                 **kwargs):
        super(MultiClinRad, self).__init__(name=name, **kwargs)
        self.input_names = input_names
        self.fusion_type = fusion_type

        if encode_model=='SPAN':
            self.dense_block_clin_1 = StackDPDense(units=int(num_hidden),
                                              activation=hidden_activation,
                                              att_activation=att_activation,
                                              initializer=kernel_initializer,
                                              name="dense_clin_1")
            self.dropout_clin_1 = tf.keras.layers.Dropout(dropout, name="dropout_clin_1")            
            self.dense_block_rad_1 = StackDPDense(units=int(num_hidden),
                                              activation=hidden_activation,
                                              att_activation=att_activation,
                                              initializer=kernel_initializer,
                                              name="dense_rad_1")
            self.dropout_rad_1 = tf.keras.layers.Dropout(dropout, name="dropout_rad_1")
            
            self.dense_block_clin_2 = StackDPDense(units=int(num_hidden/1.42857),
                                              activation=hidden_activation,
                                              att_activation=att_activation,
                                              initializer=kernel_initializer,
                                              name="dense_clin_2")
            self.dropout_clin_2 = tf.keras.layers.Dropout(dropout/1.42857, name="dropout_clin_2")
            self.dense_block_rad_2 = StackDPDense(units=int(num_hidden/1.42857),
                                              activation=hidden_activation,
                                              att_activation=att_activation,
                                              initializer=kernel_initializer,
                                              name="dense_rad_2")
            self.dropout_rad_2 = tf.keras.layers.Dropout(dropout/1.42857, name="dropout_rad_2")
        elif encode_model=='DeepSurv':
            self.dense_block_clin_1 = tf.keras.layers.Dense(units=int(num_hidden),
                                                       activation=hidden_activation,
                                                       kernel_initializer=kernel_initializer,
                                                       name="dense_clin_1")
            self.dropout_clin_1 = tf.keras.layers.Dropout(dropout, name="dropout_clin_1")
            self.dense_block_clin_2 = tf.keras.layers.Dense(units=int(num_hidden/1.42857),
                                                       activation=hidden_activation,
                                                       kernel_initializer=kernel_initializer,
                                                       name="dense_clin_2")
            self.dropout_clin_2 = tf.keras.layers.Dropout(dropout/1.42857, name="dropout_clin_2")
            
            self.dense_block_rad_1 = tf.keras.layers.Dense(units=int(num_hidden),
                                                       activation=hidden_activation,
                                                       kernel_initializer=kernel_initializer,
                                                       name="dense_rad_1")
            self.dropout_rad_1 = tf.keras.layers.Dropout(dropout, name="dropout_rad_1")
            self.dense_block_rad_2 = tf.keras.layers.Dense(units=int(num_hidden/1.42857),
                                                       activation=hidden_activation,
                                                       kernel_initializer=kernel_initializer,
                                                       name="dense_rad_2")
            self.dropout_rad_2 = tf.keras.layers.Dropout(dropout/1.42857, name="dropout_rad_2")
        
        if fusion_type=="weight":
            self.pf = AEDmodule(int(num_hidden),
                                activation="sigmoid",
                                initializer=kernel_initializer,
                                regularizer=tf.keras.regularizers.l2(l2_regularizer),
                                name="aed")
        elif fusion_type=="concat":
            self.concat = tf.keras.layers.Concatenate(axis=-1)
        
        self.hr_output_layer = tf.keras.layers.Dense(units=num_outputs,
                                                     activation=output_activation,
                                                     kernel_initializer=kernel_initializer,
                                                     kernel_regularizer=tf.keras.regularizers.l2(l2_regularizer),
                                                     name='hr_prediction')
        self.cl_output_layer = tf.keras.layers.Dense(units=num_outputs,
                                                     activation="sigmoid",
                                                     kernel_initializer=kernel_initializer,
                                                     kernel_regularizer=tf.keras.regularizers.l2(l2_regularizer),
                                                     name='cl_prediction')
    
    
    def call(self, inputs):
        x1 = inputs[self.input_name[0]]
        x2 = inputs[self.input_name[1]]
        
        x1 = self.dense_block_clin_1(x1)
        x1 = self.dropout_clin_1(x1)
        x1 = self.dense_block_clin_2(x1)
        x1 = self.dropout_clin_2(x1)
        
        x2 = self.dense_block_rad_1(x2)
        x2 = self.dropout_rad_1(x2)
        x2 = self.dense_block_rad_2(x2)
        x2 = self.dropout_rad_2(x2)
        
        if self.fusion_type=="weight":
            weight = self.pf(x1)
            x = weight * x1 + (1-weight) * x2
        elif self.fusion_type=="concat":
            x = self.concat([x1, x2])
        
        y_hr = self.hr_output_layer(x)
        y_cl = self.cl_output_layer(x)

        return [y_hr, y_cl]
        