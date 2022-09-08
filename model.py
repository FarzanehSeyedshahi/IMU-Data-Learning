import tfquaternion as tfq
import tensorflow as tf

from keras.models import Sequential, Model
from keras.layers import Bidirectional, CuDNNLSTM, Dropout, Dense, Input, Layer, Conv1D, MaxPooling1D, concatenate, Add, Activation
from keras.initializers import Constant
from tensorflow.keras.optimizers import Adam
from keras.losses import mean_absolute_error
from keras import backend as K

from tensorflow import keras
from tensorflow.keras import layers

window_size_transformer = 600

def quaternion_phi_3_error(y_true, y_pred):
    return tf.acos(K.abs(K.batch_dot(y_true, K.l2_normalize(y_pred, axis=-1), axes=-1)))

#LQIP
def quaternion_phi_4_error(y_true, y_pred):
    return 1 - K.abs(K.batch_dot(y_true, K.l2_normalize(y_pred, axis=-1), axes=-1))

#log LQIP
def quaternion_log_phi_4_error(y_true, y_pred):
    return K.log(1e-4 + quaternion_phi_4_error(y_true, y_pred))

#LQME
def quat_mult_error(y_true, y_pred):
    q_hat = tfq.Quaternion(y_true)
    q = tfq.Quaternion(y_pred).normalized()
    q_prod = q * q_hat.conjugate()
    w, x, y, z = tf.split(q_prod, num_or_size_splits=4, axis=-1)
    return tf.abs(tf.multiply(2.0, tf.concat(values=[x, y, z], axis=-1)))

#mean LQME
def quaternion_mean_multiplicative_error(y_true, y_pred):
    return tf.reduce_mean(quat_mult_error(y_true, y_pred))


# Custom loss layer
class CustomMultiLossLayer(Layer):
    def __init__(self, nb_outputs=2, **kwargs):
    #def __init__(self, nb_outputs=3, **kwargs):
        self.nb_outputs = nb_outputs
        self.is_placeholder = True
        super(CustomMultiLossLayer, self).__init__(**kwargs)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'nb_outputs': self.nb_outputs
        })
        return config
        
    def build(self, input_shape=None):
        # initialise log_vars
        self.log_vars = []
        for i in range(self.nb_outputs):
            self.log_vars += [self.add_weight(name='log_var' + str(i), shape=(1,),
                                              initializer=Constant(0.), trainable=True)]
        super(CustomMultiLossLayer, self).build(input_shape)

    def multi_loss(self, ys_true, ys_pred):
        assert len(ys_true) == self.nb_outputs and len(ys_pred) == self.nb_outputs
        loss = 0

        #LMSE (TQMSE, SMSE)
        #for y_true, y_pred, log_var in zip(ys_true, ys_pred, self.log_vars):
        #    precision = K.exp(-log_var[0])
        #    loss += K.sum(precision * (y_true - y_pred)**2., -1) + log_var[0]

        #LTMAE
        precision = K.exp(-self.log_vars[0][0])
        loss += precision * mean_absolute_error(ys_true[0], ys_pred[0]) + self.log_vars[0][0]
        #LQMAE
        precision = K.exp(-self.log_vars[1][0])
        loss += precision * quaternion_mean_multiplicative_error(ys_true[1], ys_pred[1]) + self.log_vars[1][0]
        #LQIP
        #loss += precision * quaternion_phi_4_error(ys_true[1], ys_pred[1]) + self.log_vars[1][0]

        return K.mean(loss)

    def call(self, inputs):
        ys_true = inputs[:self.nb_outputs]
        ys_pred = inputs[self.nb_outputs:]
        loss = self.multi_loss(ys_true, ys_pred)
        self.add_loss(loss, inputs=inputs)
        # We won't actually use the output.
        return K.concatenate(inputs, -1)


def create_pred_model_6d_quat(window_size=200):
    #inp = Input((window_size, 6), name='inp')
    #lstm1 = Bidirectional(CuDNNLSTM(128, return_sequences=True))(inp)
    x1 = Input((window_size, 3), name='x1')
    x2 = Input((window_size, 3), name='x2')
    convA1 = Conv1D(128, 11)(x1)
    convA2 = Conv1D(128, 11)(convA1)
    poolA = MaxPooling1D(3)(convA2)
    convB1 = Conv1D(128, 11)(x2)
    convB2 = Conv1D(128, 11)(convB1)
    poolB = MaxPooling1D(3)(convB2)
    AB = concatenate([poolA, poolB])
    lstm1 = Bidirectional(CuDNNLSTM(128, return_sequences=True))(AB)
    drop1 = Dropout(0.25)(lstm1)
    lstm2 = Bidirectional(CuDNNLSTM(128))(drop1)
    drop2 = Dropout(0.25)(lstm2)    
    y1_pred = Dense(3)(drop2)
    y2_pred = Dense(4)(drop2)

    #model = Model(inp, [y1_pred, y2_pred])
    model = Model([x1, x2], [y1_pred, y2_pred])

    model.summary()
    
    return model


def create_train_model_6d_quat(pred_model, window_size=200):
    #inp = Input(shape=(window_size, 6), name='inp')
    #y1_pred, y2_pred = pred_model(inp)
    x1 = Input((window_size, 3), name='x1')
    x2 = Input((window_size, 3), name='x2')
    y1_pred, y2_pred = pred_model([x1, x2])
    y1_true = Input(shape=(3,), name='y1_true')
    y2_true = Input(shape=(4,), name='y2_true')
    out = CustomMultiLossLayer(nb_outputs=2)([y1_true, y2_true, y1_pred, y2_pred])
    #train_model = Model([inp, y1_true, y2_true], out)
    train_model = Model([x1, x2, y1_true, y2_true], out)
    train_model.summary()
    return train_model


def create_pred_model_3d(window_size=200):
    #inp = Input((window_size, 6), name='inp')
    #lstm1 = Bidirectional(CuDNNLSTM(128, return_sequences=True))(inp)
    x1 = Input((window_size, 3), name='x1')
    x2 = Input((window_size, 3), name='x2')
    convA1 = Conv1D(128, 11)(x1)
    convA2 = Conv1D(128, 11)(convA1)
    poolA = MaxPooling1D(3)(convA2)
    convB1 = Conv1D(128, 11)(x2)
    convB2 = Conv1D(128, 11)(convB1)
    poolB = MaxPooling1D(3)(convB2)
    AB = concatenate([poolA, poolB])
    lstm1 = Bidirectional(CuDNNLSTM(128, return_sequences=True))(AB)
    drop1 = Dropout(0.25)(lstm1)
    lstm2 = Bidirectional(CuDNNLSTM(128))(drop1)
    drop2 = Dropout(0.25)(lstm2)
    y1_pred = Dense(1)(drop2)
    y2_pred = Dense(1)(drop2)
    y3_pred = Dense(1)(drop2)

    #model = Model(inp, [y1_pred, y2_pred, y3_pred])
    model = Model([x1, x2], [y1_pred, y2_pred, y3_pred])

    model.summary()
    
    return model


def create_train_model_3d(pred_model, window_size=200):
    #inp = Input(shape=(window_size, 6), name='inp')
    #y1_pred, y2_pred, y3_pred = pred_model(inp)
    x1 = Input((window_size, 3), name='x1')
    x2 = Input((window_size, 3), name='x2')
    y1_pred, y2_pred, y3_pred = pred_model([x1, x2])
    y1_true = Input(shape=(1,), name='y1_true')
    y2_true = Input(shape=(1,), name='y2_true')
    y3_true = Input(shape=(1,), name='y3_true')
    out = CustomMultiLossLayer(nb_outputs=3)([y1_true, y2_true, y3_true, y1_pred, y2_pred, y3_pred])
    #train_model = Model([inp, y1_true, y2_true, y3_true], out)
    train_model = Model([x1, x2, y1_true, y2_true, y3_true], out)
    train_model.summary()
    return train_model


def create_model_6d_rvec(window_size=200):
    input_gyro_acc = Input((window_size, 6))
    lstm1 = Bidirectional(CuDNNLSTM(128, return_sequences=True))(input_gyro_acc)    
    drop1 = Dropout(0.25)(lstm1)
    lstm2 = Bidirectional(CuDNNLSTM(128))(drop1)    
    drop2 = Dropout(0.25)(lstm2)    
    output_delta_rvec = Dense(3)(drop2)
    output_delta_tvec = Dense(3)(drop2)

    model = Model(inputs = input_gyro_acc, outputs = [output_delta_rvec, output_delta_tvec])
    model.summary()
    model.compile(optimizer = Adam(0.0001), loss = 'mean_squared_error')
    
    return model


def create_model_6d_quat(window_size=200):
    input_gyro_acc = Input((window_size, 6))
    lstm1 = Bidirectional(CuDNNLSTM(128, return_sequences=True))(input_gyro_acc)    
    drop1 = Dropout(0.25)(lstm1)
    lstm2 = Bidirectional(CuDNNLSTM(128))(drop1)    
    drop2 = Dropout(0.25)(lstm2)    
    output_delta_p = Dense(3)(drop2)
    output_delta_q = Dense(4)(drop2)

    model = Model(inputs = input_gyro_acc, outputs = [output_delta_p, output_delta_q])
    model.summary()
    #model.compile(optimizer = Adam(0.0001), loss = 'mean_squared_error')
    model.compile(optimizer = Adam(0.0001), loss = ['mean_absolute_error', quaternion_mean_multiplicative_error])
    #model.compile(optimizer = Adam(0.0001), loss = ['mean_absolute_error', quaternion_phi_4_error])
    
    return model


def create_model_3d(window_size=200):
    input_gyro_acc = Input((window_size, 6))
    lstm1 = Bidirectional(CuDNNLSTM(128, return_sequences=True))(input_gyro_acc)    
    drop1 = Dropout(0.25)(lstm1)
    lstm2 = Bidirectional(CuDNNLSTM(128))(drop1)    
    drop2 = Dropout(0.25)(lstm2)    
    output_delta_l = Dense(1)(drop2)
    output_delta_theta = Dense(1)(drop2)
    output_delta_psi = Dense(1)(drop2)

    model = Model(inputs = input_gyro_acc, outputs = [output_delta_l, output_delta_theta, output_delta_psi])
    model.summary()
    model.compile(optimizer = Adam(0.0001), loss = 'mean_squared_error')
    
    return model


def create_model_2d(window_size=200):
    input_gyro_acc = Input((window_size, 6))
    lstm1 = Bidirectional(CuDNNLSTM(128, return_sequences=True))(input_gyro_acc)    
    drop1 = Dropout(0.25)(lstm1)
    lstm2 = Bidirectional(CuDNNLSTM(128))(drop1)
    drop2 = Dropout(0.25)(lstm2)    
    output_delta_l = Dense(1)(drop2)
    output_delta_psi = Dense(1)(drop2)
    model = Model(inputs = input_gyro_acc, outputs = [output_delta_l, output_delta_psi])
    model.summary()
    model.compile(optimizer = Adam(0.0001), loss = 'mean_squared_error')
    
    return model


##### Adding new paper changes

## Adding resnet model
def create_pred_resnet_model_6d_quat(window_size=200):
    x1 = Input((window_size, 3), name='x1')
    x2 = Input((window_size, 3), name='x2')

    convA1 = Conv1D(128, 11, activation='relu', padding = "same")(x1)
    conv_shortcut_A1 = convA1
    convA2 = Conv1D(128, 11, activation='relu', padding = "same")(convA1)
    convA3 = Conv1D(128, 11, activation='relu', padding = "same")(convA2)
    addA = Add()([convA3, conv_shortcut_A1])
    actA = Activation('relu')(addA)
    poolA = MaxPooling1D(3)(actA)

    convB1 = Conv1D(128, 11, activation='relu', padding = "same")(x2)
    conv_shortcut_B1 = convB1
    convB2 = Conv1D(128, 11, activation='relu', padding = "same")(convB1)
    convB3 = Conv1D(128, 11, activation='relu', padding = "same")(convB2)
    addB = Add()([convB3, conv_shortcut_B1])
    actB = Activation('relu')(addB)
    poolB = MaxPooling1D(3)(actB)

    AB = concatenate([poolA, poolB])
    lstm1 = Bidirectional(CuDNNLSTM(128, return_sequences=True))(AB)
    drop1 = Dropout(0.25)(lstm1)
    lstm2 = Bidirectional(CuDNNLSTM(128))(drop1)
    drop2 = Dropout(0.25)(lstm2)    
    y1_pred = Dense(3)(drop2)
    y2_pred = Dense(4)(drop2)
    model = Model([x1, x2], [y1_pred, y2_pred])
    model.summary()
    return model

## Adding 9d resnet model
def create_resnet_pred_model_9d_quat(window_size=200):
    
    x1 = Input((window_size, 3), name='x1')
    x2 = Input((window_size, 3), name='x2')
    x3 = Input((window_size, 3), name='x3')

    convA1 = Conv1D(128, 11, activation='relu', padding = "same")(x1)
    conv_shortcut_A1 = convA1
    convA2 = Conv1D(128, 11, activation='relu', padding = "same")(convA1)
    convA3 = Conv1D(128, 11, activation='relu', padding = "same")(convA2)
    addA = Add()([convA3, conv_shortcut_A1])
    actA = Activation('relu')(addA)
    poolA = MaxPooling1D(3)(actA)

    convB1 = Conv1D(128, 11, activation='relu', padding = "same")(x2)
    conv_shortcut_B1 = convB1
    convB2 = Conv1D(128, 11, activation='relu', padding = "same")(convB1)
    convB3 = Conv1D(128, 11, activation='relu', padding = "same")(convB2)
    addB = Add()([convB3, conv_shortcut_B1])
    actB = Activation('relu')(addB)
    poolB = MaxPooling1D(3)(actB)

    convC1 = Conv1D(128, 11, activation='relu', padding = "same")(x3)
    conv_shortcut_C1 = convC1
    convC2 = Conv1D(128, 11, activation='relu', padding = "same")(convC1)
    convC3 = Conv1D(128, 11, activation='relu', padding = "same")(convC2)
    addC = Add()([convC3, conv_shortcut_C1])
    actC = Activation('relu')(addC)
    poolC = MaxPooling1D(3)(actC)

    ABC = concatenate([poolA, poolB, poolC])
    lstm1 = Bidirectional(CuDNNLSTM(128, return_sequences=True))(ABC)
    drop1 = Dropout(0.25)(lstm1)
    lstm2 = Bidirectional(CuDNNLSTM(128))(drop1)
    drop2 = Dropout(0.25)(lstm2)    
    y1_pred = Dense(3)(drop2)
    y2_pred = Dense(4)(drop2)

    model = Model([x1, x2, x3], [y1_pred, y2_pred])
    model.summary()
    return model

## Adding Loss function to the 9d model Architecture
def create_train_resnet_or_without_model_9d_quat(pred_model, window_size):
    x1 = Input((window_size, 3), name='x1')
    x2 = Input((window_size, 3), name='x2')
    x3 = Input((window_size, 3), name='x3')
    y1_pred, y2_pred = pred_model([x1, x2, x3])
    y1_true = Input(shape=(3,), name='y1_true')
    y2_true = Input(shape=(4,), name='y2_true')
    out = CustomMultiLossLayer(nb_outputs=2)([y1_true, y2_true, y1_pred, y2_pred])
    train_model = Model([x1, x2, x3, y1_true, y2_true], out)
    train_model.summary()
    return train_model

#################################### Transformer models#####################################

def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout):
    #Normalization and attention
    x = layers.LayerNormalization(epsilon=1e-6)(inputs)
    x = layers.MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(x, x)
    x = layers.Dropout(dropout)(x)
    res = x + inputs

    # Feed Forward Part
    x = layers.LayerNormalization(epsilon=1e-6)(res)
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)

    return x+res

def create_transformer_pred_model_9d(
    head_size,
    window_size,
    num_heads,
    ff_dim,
    num_transformer_blocks,
    mlp_units,
    dropout,
    mlp_dropout,
):

    x1 = Input((window_size, 3), name='x1')
    x2 = Input((window_size, 3), name='x2')
    x3 = Input((window_size, 3), name='x3')

    convA1 = Conv1D(128, 11, activation='relu', padding = "same")(x1)
    conv_shortcut_A1 = convA1
    convA2 = Conv1D(128, 11, activation='relu', padding = "same")(convA1)
    convA3 = Conv1D(128, 11, activation='relu', padding = "same")(convA2)
    addA = Add()([convA3, conv_shortcut_A1])
    actA = Activation('relu')(addA)
    poolA = MaxPooling1D(3)(actA)

    convB1 = Conv1D(128, 11, activation='relu', padding = "same")(x2)
    conv_shortcut_B1 = convB1
    convB2 = Conv1D(128, 11, activation='relu', padding = "same")(convB1)
    convB3 = Conv1D(128, 11, activation='relu', padding = "same")(convB2)
    addB = Add()([convB3, conv_shortcut_B1])
    actB = Activation('relu')(addB)
    poolB = MaxPooling1D(3)(actB)

    convC1 = Conv1D(128, 11, activation='relu', padding = "same")(x3)
    conv_shortcut_C1 = convC1
    convC2 = Conv1D(128, 11, activation='relu', padding = "same")(convC1)
    convC3 = Conv1D(128, 11, activation='relu', padding = "same")(convC2)
    addC = Add()([convC3, conv_shortcut_C1])
    actC = Activation('relu')(addC)
    poolC = MaxPooling1D(3)(actC)

    ABC = concatenate([poolA, poolB, poolC])

    for _ in range(num_transformer_blocks):
        x = transformer_encoder(ABC, head_size, num_heads, ff_dim, dropout)

    x = layers.GlobalAveragePooling1D(data_format="channels_first")(x)
    for dim in mlp_units:
        x = layers.Dense(dim, activation="relu")(x)
        x = layers.Dropout(mlp_dropout)(x)
    y1_pred = Dense(3)(x)
    y2_pred = Dense(4)(x)


    model = Model([x1, x2, x3], [y1_pred, y2_pred])
    model.summary()
    # return keras.Model(inputs, outputs)
    return model
    
