import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dense, Input, Conv1D, Flatten, Conv1DTranspose, Lambda, Dropout, Reshape, Add
from tensorflow.keras import backend as K

class MCDropout(tf.keras.layers.Layer):
    def __init__(self, rate, **kwargs):
        super().__init__(**kwargs)
        self.rate = rate

    def call(self, inputs, training=None):
        return tf.nn.dropout(inputs, rate=self.rate)

class SimpleConvVAE1D:
    def __init__(self, input_dim, latent_dim=3, actfn="relu", beta=tf.Variable(0.0), dropout_rate=0.1):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.actfn = actfn
        self.beta = beta
        self.dropout_rate = dropout_rate

    def sampling(self, args):
        z_mean, z_log_sigma = args
        epsilon = tf.random.normal(shape=tf.shape(z_mean))
        return z_mean + tf.exp(z_log_sigma) * epsilon

    def model_tf(self):
        inputs = Input(shape=(self.input_dim,))
        x = Reshape((self.input_dim, 1))(inputs)
        x = Conv1D(32, 5, activation=self.actfn, padding="same")(x)
        x = Conv1D(64, 5, activation=self.actfn, padding="same")(x)
        x = Flatten()(x)

        z_mean = Dense(self.latent_dim)(x)
        z_log_sigma = Dense(self.latent_dim)(x)
        z_mean = tf.clip_by_value(z_mean, -10.0, 10.0)
        z_log_sigma = tf.clip_by_value(z_log_sigma, -10.0, 10.0)

        z = Lambda(self.sampling)([z_mean, z_log_sigma])
        encoder = Model(inputs, [z_mean, z_log_sigma, z], name="encoder")

        latent_inputs = Input(shape=(self.latent_dim,))
        x = Dense(self.input_dim, activation=self.actfn)(latent_inputs)
        x = Reshape((self.input_dim, 1))(x)
        x = Conv1D(64, 5, activation=self.actfn, padding="same")(x)
        x = MCDropout(self.dropout_rate)(x)
        x = Conv1D(32, 5, activation=self.actfn, padding="same")(x)
        x = Conv1D(1, 3, activation="linear", padding="same")(x)
        outputs = Reshape((self.input_dim,))(x)
        decoder = Model(latent_inputs, outputs, name="decoder")

        vae = Model(inputs, decoder(z), name="vae")

        kl_loss = -0.5 * tf.reduce_mean(1 + z_log_sigma - tf.square(z_mean) - tf.exp(z_log_sigma))
        recon_loss = tf.reduce_mean(tf.square(inputs - decoder(z)))
        vae.add_loss(self.beta * kl_loss + recon_loss)
        return vae, encoder, decoder
    

def residual_block(x, filters, kernel_size, actfn="relu"):
    shortcut = x
    x = Conv1D(filters, kernel_size, padding="same", activation=actfn)(x)
    x = Conv1D(filters, kernel_size, padding="same")(x)
    return tf.keras.layers.add([x, shortcut])


class FullConvVAE1D:
    def __init__(self, input_dim, latent_dim=3, actfn="relu", beta=tf.Variable(0.0), dropout_rate=0.1,
                conv_filters=(64, 128), kernel_size=3, use_residual=False,
                use_dense_encoder=False, dense_units_encoder=128,
                use_dense_decoder=False, dense_units_decoder=128):
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.actfn = actfn
        self.beta = beta
        self.dropout_rate = dropout_rate
        self.conv_filters = conv_filters
        self.kernel_size = kernel_size
        self.use_residual = use_residual
        self.use_dense_encoder = use_dense_encoder
        self.dense_units_encoder = dense_units_encoder
        self.use_dense_decoder = use_dense_decoder
        self.dense_units_decoder = dense_units_decoder

    def sampling(self, args):
        z_mean, z_log_sigma = args
        epsilon = tf.random.normal(shape=tf.shape(z_mean))
        return z_mean + tf.exp(z_log_sigma) * epsilon

    def model_tf(self):
        inputs = Input(shape=(self.input_dim,))
        x = Reshape((self.input_dim, 1))(inputs)
        x = Conv1D(self.conv_filters[0], self.kernel_size, activation=self.actfn, padding="same")(x)
        if self.use_residual:
            x = residual_block(x, self.conv_filters[0], self.kernel_size, self.actfn)
        x = Conv1D(self.conv_filters[1], self.kernel_size, activation=self.actfn, padding="same")(x)
        if self.use_residual:
            x = residual_block(x, self.conv_filters[1], self.kernel_size, self.actfn)

        x = Flatten()(x)

        if self.use_dense_encoder:
            x = Dense(self.dense_units_encoder, activation=self.actfn)(x)

        z_mean = Dense(self.latent_dim)(x)
        z_log_sigma = Dense(self.latent_dim)(x)

        #z_mean = tf.clip_by_value(z_mean, -10.0, 10.0)
        #z_log_sigma = tf.clip_by_value(z_log_sigma, -10.0, 10.0)

        z = Lambda(self.sampling)([z_mean, z_log_sigma])
        encoder = Model(inputs, [z_mean, z_log_sigma, z], name="encoder")

        latent_inputs = Input(shape=(self.latent_dim,))

        x = latent_inputs
        if self.use_dense_decoder:
            x = Dense(self.dense_units_decoder, activation=self.actfn)(x)

        x = Dense(self.input_dim, activation=self.actfn)(x)
        x = Reshape((self.input_dim, 1))(x)

        x = Conv1D(self.conv_filters[1], self.kernel_size, activation=self.actfn, padding="same")(x)
        if self.use_residual:
            x = residual_block(x, self.conv_filters[1], self.kernel_size, self.actfn)
        x = MCDropout(self.dropout_rate)(x)
        x = Conv1D(self.conv_filters[0], self.kernel_size, activation=self.actfn, padding="same")(x)
        if self.use_residual:
            x = residual_block(x, self.conv_filters[0], self.kernel_size, self.actfn)

        x = Conv1D(1, 3, activation="linear", padding="same")(x)
        outputs = Reshape((self.input_dim,))(x)
        decoder = Model(latent_inputs, outputs, name="decoder")

        vae = Model(inputs, decoder(z), name="vae")

        kl_loss = -0.5 * tf.reduce_mean(1 + z_log_sigma - tf.square(z_mean) - tf.exp(z_log_sigma))
        recon_loss = tf.reduce_mean(tf.square(inputs - decoder(z)))
        vae.add_loss(self.beta * kl_loss + recon_loss)
        return vae, encoder, decoder


class UNetVAE1D:
    def __init__(self, input_dim, latent_dim=3, actfn="relu", beta=tf.Variable(0.0), dropout_rate=0.1,
                 conv_filters=(64, 128), kernel_size=3):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.actfn = actfn
        self.beta = beta
        self.dropout_rate = dropout_rate
        self.conv_filters = conv_filters
        self.kernel_size = kernel_size

    def sampling(self, args):
        z_mean, z_log_sigma = args
        epsilon = tf.random.normal(shape=tf.shape(z_mean))
        return z_mean + tf.exp(0.5 * z_log_sigma) * epsilon

    def model_tf(self):
        inputs = Input(shape=(self.input_dim,))
        x = Reshape((self.input_dim, 1))(inputs)

        c1 = Conv1D(self.conv_filters[0], self.kernel_size, activation=self.actfn, padding="same", name="conv1_skip")(x)
        c2 = Conv1D(self.conv_filters[1], self.kernel_size, activation=self.actfn, padding="same", strides=2)(c1)

        x_flat = Flatten()(c2)
        z_mean = Dense(self.latent_dim, name="z_mean")(x_flat)
        z_log_sigma = Dense(self.latent_dim, name="z_log_sigma")(x_flat)
        z = Lambda(self.sampling, name="z")([z_mean, z_log_sigma])

        encoder = Model(inputs, [z_mean, z_log_sigma, z], name="encoder")

        latent_inputs = Input(shape=(self.latent_dim,))
        x = Dense((self.input_dim // 2) * self.conv_filters[1], activation=self.actfn)(latent_inputs)
        x = Reshape((self.input_dim // 2, self.conv_filters[1]))(x)
        d1 = Conv1DTranspose(self.conv_filters[0], self.kernel_size, activation=self.actfn, padding="same", strides=2)(x)

        skip_input = Input(shape=(self.input_dim, self.conv_filters[0]))
        d1 = Add()([d1, skip_input])

        out = Conv1D(1, 1, activation="linear", padding="same")(d1)
        outputs = Reshape((self.input_dim,))(out)

        decoder = Model([latent_inputs, skip_input], outputs, name="decoder")

        vae_outputs = decoder([z, c1])
        vae = Model(inputs, vae_outputs, name="vae")

        kl_loss = -0.5 * K.mean(1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma))
        recon_loss = K.mean(K.square(inputs - vae_outputs))
        vae.add_loss(self.beta * kl_loss + recon_loss)

        return vae, encoder, decoder