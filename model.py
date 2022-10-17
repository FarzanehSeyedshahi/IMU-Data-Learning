from turtle import position
import tfquaternion as tfq
import tensorflow as tf
from util import *
import tensorflow_addons as tfa


from keras.models import Sequential, Model
from keras.layers import Bidirectional, Flatten, Reshape, Lambda, CuDNNLSTM, Dropout, Dense, Input, MaxPooling2D, Layer, Conv2D, Conv1D, MaxPooling1D, concatenate, Add, Activation
from keras.initializers import Constant
from tensorflow.keras.optimizers import Adam
from keras.losses import mean_absolute_error
from keras import backend as K

from tensorflow import keras
from tensorflow.keras import layers

window_size_transformer = 200

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
    # train_model.summary()
    return train_model

#################################### Transformer models#####################################
# class PositionalEncoding(torch.nn.Module):

#     def __init__(self, d_model, dropout=0.1, max_len=5000):
#         self.dropout = torch.nn.Dropout(p=dropout)

#         pe = torch.zeros(max_len, (d_model + 1) // 2 * 2)
#         position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
#         pe[:, 0::2] = torch.sin(position * div_term)
#         pe[:, 1::2] = torch.cos(position * div_term)
#         pe = pe.unsqueeze(0).transpose(0, 1)
#         self.register_buffer('pe', pe)

#     def forward(self, x):
#         r"""Inputs of forward function
#         Args:
#             x: the sequence fed to the positional encoder model (required).
#         Shape:
#             x: [sequence length, batch size, embed dim]
#             output: [sequence length, batch size, embed dim]
#         Examples:
#         """

#         x = x + self.pe[:x.size(0), :, :x.size(2)]
#         return self.dropout(x)

class PositionEncoding(layers.Layer):
    def __init__(self, maxlength, window_size, embed_dim, **kwargs):
        super(PositionEncoding, self).__init__()
        self.window_embed = layers.Embedding(window_size, embed_dim)

        pe = tf.zeros(maxlength, (embed_dim + 1) // 2 * 2)
        position = tf.range(0, maxlength)
        div_term = tf.math.exp(tf.range(0, embed_dim, 2).float() * (-tf.math.log(10000.0) / embed_dim))
        pe[:, 0::2] = tf.math.sin(position * div_term)
        pe[:, 1::2] = tf.math.cos(position * div_term)
        self.position_encode = pe
        super(PositionEncoding, self).__init__(**kwargs)

    def call(self, x):
        x = self.window_embed(x)
        return x + self.position_encode




class PositionEmbedding(layers.Layer):
    def __init__(self, maxlength, window_size, embed_dim, **kwargs):
        super(PositionEmbedding, self).__init__()
        self.window_embed = layers.Embedding(window_size, embed_dim)
        self.position_embed = layers.Embedding(maxlength, embed_dim)
        super(PositionEmbedding, self).__init__(**kwargs)

    def call(self, x):
        maxlength = tf.shape(x)[-1]
        positions = tf.range(0, maxlength)

        positions = self.position_embed(positions)
        x = self.window_embed(x)
        return x + positions



class TransformerBlock(layers.Layer):
    def __init__(self, embed_dims, num_heads, dropout, **kwargs):
        super(TransformerBlock, self).__init__()
        self.attention_layer =  layers.MultiHeadAttention(num_heads, embed_dims)
        self.dropout_layer_1 = layers.Dropout(dropout)
        self.norm_layer_1 = layers.LayerNormalization(epsilon=1e-6)
        self.feed_forward_network_layers = Sequential([
            layers.Dense(embed_dims, activation='relu'),
            layers.Dense(embed_dims)
        ])
        self.dropout_layer_2 = layers.Dropout(dropout)
        self.norm_layer_2 = layers.LayerNormalization(epsilon=1e-6)
        super(TransformerBlock, self).__init__(**kwargs)

    def call(self, inputs):
        # batch_size, sequence_length, embed_dime = tf.shape(inputs)[0], tf.shape(inputs)[1], tf.shape(inputs)[2]
        #Causal Masking
        # mask = causal_mask(batch_size, sequence_length, embed_dime)
        # mask = self.
        # Multiheaded Attention
        mha_out = self.attention_layer(inputs, inputs)
        dropout_1_out = self.dropout_layer_1(mha_out)
        norm_1_out = self.norm_layer_1(dropout_1_out)
        combined_1_out = norm_1_out + inputs
        #feedforward network
        ffn_out = self.feed_forward_network_layers(combined_1_out)
        dropout_2_out = self.dropout_layer_2(ffn_out)
        norm_2_out = self.norm_layer_2(dropout_2_out)
        combined_2_out = norm_2_out + combined_1_out

        return combined_2_out




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

    x1 = layers.BatchNormalization(trainable=True, epsilon=1e-9)(x1)
    x2 = layers.BatchNormalization(trainable=True, epsilon=1e-9)(x2)
    x3 = layers.BatchNormalization(trainable=True, epsilon=1e-9)(x3)

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

    x = concatenate([poolA, poolB, poolC])

    #Embedding
    # x = PositionEmbedding(maxlength=200, window_size=200, embed_dim=32)(x)

    # for _ in range(num_transformer_blocks):
    x = transformer_encoder(x, 32, num_heads, ff_dim, dropout)
    x = transformer_encoder(x, 32, num_heads, ff_dim, dropout)
    x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

    x = layers.GlobalAveragePooling1D(data_format="channels_first")(x)
    for dim in mlp_units:
        x = layers.Dense(dim, activation="relu")(x)
        x = layers.Dropout(mlp_dropout)(x)
    y1_pred = Dense(3)(x)
    y2_pred = Dense(4)(x)


    model = Model([x1, x2, x3], [y1_pred, y2_pred])
    # model.summary()
    # return keras.Model(inputs, outputs)
    return model
    


### Adding VIT model
def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation=tf.nn.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x


class Patches(layers.Layer):
    def __init__(self, patch_size, **kwargs):
        super(Patches, self).__init__()
        self.patch_size = patch_size
        super(Patches, self).__init__(**kwargs)


    def get_config(self):
        config = super().get_config()
        config.update({
            "patch_size": self.patch_size,
        })
        return config


    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size[0], self.patch_size[1], 1],
            strides=[1, self.patch_size[0], self.patch_size[1], 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches


class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim, **kwargs):
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )
        self.projection_dim = projection_dim
        super(PatchEncoder, self).__init__(**kwargs)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "num_patches": self.num_patches,
            "projection_dim": self.projection_dim
        })
        return config

    #Embeddig
    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded

    #Encoding
    # def call(self, patch):
    #     print(self.num_patches, self.projection_dim)
    #     pe_np = np.array((self.num_patches, (self.projection_dim + 1) // 2 * 2))
    #     # pe = tf.zeros([self.num_patches, (self.projection_dim + 1) // 2 * 2])
    #     # print(tf.shape(pe))
    #     position = tf.range(0, self.num_patches, dtype=tf.float32)
    #     print(tf.shape(position))

    #     div_term = tf.math.exp( tf.range(0, self.num_patches, dtype=tf.float32) * (-tf.math.log(10000.0) / self.projection_dim))
    #     print(tf.shape(div_term))
    #     print(tf.math.sin(position * div_term))
    #     pe_np[:, 0::2] = tf.math.sin(position * div_term)
    #     pe_np[:, 1::2] = tf.math.cos(position * div_term)

    #     pe = tf.convert_to_tensor(pe_np)
    #     encoded = self.projection(patch) + pe
    #     return encoded


def create_vit_classifier(input_shape, image_size, patch_size, projection_dim, num_heads):

    num_patches = (image_size[0] // patch_size[0]) * (image_size[1] // patch_size[1])

    transformer_units = [
        projection_dim * 2,
        projection_dim,
    ]  # Size of the transformer layers
    transformer_layers = 12
    # mlp_head_units = [2048, 1024]  # Size of the dense layers of the final classifier

    inputs = layers.Input(shape=input_shape)
    # Create patches.
    patches = Patches(patch_size)(inputs)
    # print("patches shape:",tf.shape(patches))

    # Encode patches.
    encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)
    # print("encoded_patches shape:",tf.shape(encoded_patches))


    # Create multiple layers of the Transformer block.
    for _ in range(transformer_layers):
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        # Create a multi-head attention layer.
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.1
        )(x1, x1)
        # Skip connection 1.
        x2 = layers.Add()([attention_output, encoded_patches])
        # Layer normalization 2.
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        # MLP.
        x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.1)
        # Skip connection 2.
        encoded_patches = layers.Add()([x3, x2])

    # Create a [batch_size, projection_dim] tensor.
    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    
    lstm1 = Bidirectional(CuDNNLSTM(128, return_sequences=True))(representation)
    drop1 = Dropout(0.25)(lstm1)
    lstm2 = Bidirectional(CuDNNLSTM(128))(drop1)
    drop2 = Dropout(0.25)(lstm2)    

    # Classify outputs.
    y_true1 = layers.Dense(3)(drop2)
    y_true2 = layers.Dense(4)(drop2)

    # Create the Keras model.
    model = keras.Model(inputs=inputs, outputs=[y_true1,y_true2])
    return model


def create_VIT_model_9d(pred_model, window_size):
    # x1 = Input((window_size, 3), name='x1')
    # x2 = Input((window_size, 3), name='x2')
    # x3 = Input((window_size, 3), name='x3')
    x = Input((window_size, 9), name='x')
    y1_pred, y2_pred = pred_model(x)
    y1_true = Input(shape=(3,), name='y1_true')
    y2_true = Input(shape=(4,), name='y2_true')
    out = CustomMultiLossLayer(nb_outputs=2)([y1_true, y2_true, y1_pred, y2_pred])
    train_model = Model([x, y1_true, y2_true], out)
    return train_model

def create_combined_transformer_pred_model_9d(window_size):

    x1 = Input((window_size, 3), name='x1')
    x2 = Input((window_size, 3), name='x2')
    x3 = Input((window_size, 3), name='x3')

    x1 = layers.BatchNormalization(trainable=True, epsilon=1e-9)(x1)
    x2 = layers.BatchNormalization(trainable=True, epsilon=1e-9)(x2)
    x3 = layers.BatchNormalization(trainable=True, epsilon=1e-9)(x3)

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

    x = concatenate([poolA, poolB, poolC])
    x = layers.Reshape((66, 384,1), input_shape=(66, 384))(x)

    image_size = (66, 384)
    patch_size = (6,32)
    num_patches = (image_size[0] // patch_size[0]) * (image_size[1] // patch_size[1])
    projection_dim = 64
    num_heads = 8
    transformer_units = [
        projection_dim * 2,
        projection_dim,
    ]  # Size of the transformer layers
    transformer_layers = 8
    mlp_head_units = [2048, 1024]  # Size of the dense layers of the final classifier

    patches = Patches(patch_size)(x)
    # # print("patches shape:",tf.shape(patches))

    # # Encode patches.
    encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)
    # # print("encoded_patches shape:",tf.shape(encoded_patches))


    # Create multiple layers of the Transformer block.
    for _ in range(transformer_layers):
        # Layer normalization 1.
        x_normalization = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        # Create a multi-head attention layer.
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.1
        )(x_normalization, x_normalization)
        # Skip connection 1.
        Adding_output = layers.Add()([attention_output, encoded_patches])
        # Layer normalization 2.
        x2_normalization = layers.LayerNormalization(epsilon=1e-6)(Adding_output)
        # MLP.
        x_mlp = mlp(x2_normalization, hidden_units=transformer_units, dropout_rate=0.1)
        # Skip connection 2.
        encoded_patches = layers.Add()([x_mlp, Adding_output])

    # # Create a [batch_size, projection_dim] tensor.
    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    # # print("representation shape:",tf.shape(representation))

    representation = layers.Flatten()(representation)
    # # print("representation shape:",tf.shape(representation))

    representation = layers.Dropout(0.5)(representation)
    # # Add MLP.
    features = mlp(representation, hidden_units=mlp_head_units, dropout_rate=0.5)
    # # print("features shape:",tf.shape(features))

    # # Classify outputs.
    y1_pred = layers.Dense(3)(features)
    y2_pred = layers.Dense(4)(features)

    # # Create the Keras model.
    model = keras.Model([x1, x2, x3], [y1_pred, y2_pred])
    return model

################################# Adding CWT ######################

def ReshapeLayer(x):
    
    shape = x.shape
    
    # 1 possibility: H,W*channel
    reshape = Reshape((shape[1],shape[2]*shape[3]))(x)
    
    # 2 possibility: W,H*channel
    # transpose = Permute((2,1,3))(x)
    # reshape = Reshape((shape[1],shape[2]*shape[3]))(transpose)
    
    return reshape

def create_pred_resnet_cwt(sample_size, window_size=200):
    
    x1 = Input((sample_size, window_size, 3), name='x1')
    x2 = Input((sample_size, window_size, 3), name='x2')
    x3 = Input((sample_size, window_size, 3), name='x3')

    convA1 = Conv2D(128, 11, activation='relu', padding = "same")(x1)
    conv_shortcut_A1 = convA1
    convA2 = Conv2D(128, 11, activation='relu', padding = "same")(convA1)
    convA3 = Conv2D(128, 11, activation='relu', padding = "same")(convA2)
    addA = Add()([convA3, conv_shortcut_A1])
    actA = Activation('relu')(addA)
    poolA = MaxPooling2D((3,3))(actA)

    convB1 = Conv2D(128, 11, activation='relu', padding = "same")(x2)
    conv_shortcut_B1 = convB1
    convB2 = Conv2D(128, 11, activation='relu', padding = "same")(convB1)
    convB3 = Conv2D(128, 11, activation='relu', padding = "same")(convB2)
    addB = Add()([convB3, conv_shortcut_B1])
    actB = Activation('relu')(addB)
    poolB = MaxPooling2D((3,3))(actB)

    convC1 = Conv2D(128, 11, activation='relu', padding = "same")(x3)
    conv_shortcut_C1 = convC1
    convC2 = Conv2D(128, 11, activation='relu', padding = "same")(convC1)
    convC3 = Conv2D(128, 11, activation='relu', padding = "same")(convC2)
    addC = Add()([convC3, conv_shortcut_C1])
    actC = Activation('relu')(addC)
    poolC = MaxPooling2D((3,3))(actC)

    ABC = concatenate([poolA, poolB, poolC])

    ABC = Lambda(ReshapeLayer)(ABC) # <========== pass from 4D to 3D
    ABC = Flatten()(ABC)
    # lstm1 = Bidirectional(CuDNNLSTM(128, return_sequences=True))(ABC)
    # drop1 = Dropout(0.25)(lstm1)
    # lstm2 = Bidirectional(CuDNNLSTM(128))(drop1)
    # drop2 = Dropout(0.25)(ABC)    
    y1_pred = Dense(3)(ABC)
    y2_pred = Dense(4)(ABC)

    model = Model([x1, x2, x3], [y1_pred, y2_pred])
    return model

def create_train_resnet_cwt(pred_model, sample_size, window_size):
    x1 = Input((sample_size, window_size, 3), name='x1')
    x2 = Input((sample_size, window_size, 3), name='x2')
    x3 = Input((sample_size, window_size, 3), name='x3')
    y1_pred, y2_pred = pred_model([x1, x2, x3])
    y1_true = Input(shape=(3,), name='y1_true')
    y2_true = Input(shape=(4,), name='y2_true')
    out = CustomMultiLossLayer(nb_outputs=2)([y1_true, y2_true, y1_pred, y2_pred])
    train_model = Model([x1, x2, x3, y1_true, y2_true], out)
    return train_model
