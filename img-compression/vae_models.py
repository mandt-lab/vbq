import tensorflow as tf
import numpy as np
# for compression
from scipy.stats import norm

log2pi = np.log(2. * np.pi).astype('float32')


def log_normal_pdf(sample, mean, logvar):
    # compute normal logpdf, element-wise
    return -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi)


class StandardGaussianPrior:
    @staticmethod
    def logpdf(z):
        return log_normal_pdf(z, 0., 0.)

    @staticmethod
    def pdf(z):
        return np.exp(StandardGaussianPrior.logpdf(z))

    @staticmethod
    def inverse_cdf(xi):
        return norm.ppf(xi)


class FactoredGaussianPrior:  # currently only used by prior CDF quantizer for finding quantization points
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
        self.logvar = 2 * np.log(std)

    def logpdf(self, z):
        return log_normal_pdf(z, self.mean, self.logvar)

    def pdf(self, z):
        return np.exp(self.logpdf(z))

    def inverse_cdf(self, xi):
        # assert xi.shape[-len(self.mean.shape):] == self.mean.shape
        assert xi.shape[-1] == len(self.mean)
        return norm.ppf(xi, loc=self.mean, scale=self.std)


from learned_prior import BMSHJ2018Prior


class GaussianVAE(tf.keras.Model):
    def __init__(self, prior, inference_net, generative_net, decode_sigmoid=False):
        super(GaussianVAE, self).__init__()
        self.prior = prior
        self.inference_net = inference_net
        self.generative_net = generative_net
        self.decode_sigmoid = decode_sigmoid

    def encode(self, x):
        # Should compute means and logvars of posterior q(z|x) given a batch input x
        mean, logvar = tf.split(self.inference_net(x), num_or_size_splits=2, axis=-1)
        return mean, logvar

    def decode(self, z):
        # Should compute means of likelihood p(x|z) given a batch input z (currently assuming fixed logvar)
        # mean, logvar = tf.split(self.generative_net(z), num_or_size_splits=2, axis=-1)  # split along channels
        # mean = self.generative_net(z)
        # mean = tf.sigmoid(mean)  # to constraint mean to (0, 1), as done in the VAE paper on Frey dataset
        # return mean, logvar

        mean = self.generative_net(z)  # currently only predicting means (not variances) of p(x|z)
        if self.decode_sigmoid:
            mean = tf.sigmoid(mean)  # constraint mean to (0, 1), as done in the VAE paper on Frey dataset
        return mean

    def reparameterize(self, mean, logvar):
        """
        Given a batch of parameters of q(z|x), use reparameterization trick to sample a batch of z from corresponding
        q(z|x); result has same shape as mean and logvar (only one sample is generated for each q(z|x) distribution)
        :param mean:
        :param logvar:
        :return:
        """
        eps = tf.random.normal(shape=tf.shape(mean))
        return eps * tf.exp(logvar * .5) + mean

    # @tf.function
    def compute_loss(self, x, likelihood_variance):
        z_mean, z_logvar = self.encode(x)
        z = self.reparameterize(z_mean, z_logvar)
        # x_hat_mean, x_hat_logvar = model.decode(z)
        x_hat = self.decode(z)
        likelihood_logvar = np.log(likelihood_variance).astype('float32')
        x_hat_logvar = likelihood_logvar

        dims_except_batch = list(range(1, len(x.shape)))  # only want to keep batch (0th) dimension
        logpx_z = tf.reduce_sum(log_normal_pdf(x, x_hat, x_hat_logvar), axis=dims_except_batch)
        dims_except_batch = list(range(1, len(z.shape)))  # only want to keep batch (0th) dimension
        logpz = tf.reduce_sum(self.prior.logpdf(z), axis=dims_except_batch)
        logqz_x = tf.reduce_sum(log_normal_pdf(z, z_mean, z_logvar), axis=dims_except_batch)

        likelihood = tf.reduce_mean(logpx_z)
        kl = tf.reduce_mean(logqz_x - logpz)
        elbo = likelihood - kl
        return dict(loss=-elbo, elbo=elbo, likelihood=likelihood, kl=kl, x_hat=x_hat)

    @tf.function
    def compute_apply_gradients(self, x, optimizer, **kwargs):
        with tf.GradientTape() as tape:
            loss = self.compute_loss(x, **kwargs)['loss']
        gradients = tape.gradient(loss, self.trainable_variables)
        optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        # @tf.function
        # def sample(self, num_samples=None, eps=None):
        #     if eps is None:
        #         eps = tf.random.normal(shape=(num_samples, self.latent_dim))
        #     return self.decode(eps)


import tensorflow_compression as tfc


def get_BLS2017_neural_nets(num_filters, filter_dims, sampling_rates):
    """
    Create the inference and generative networks for a Gaussian VAE, using the GDN/IGDN architecture from Balle 2017.
    :param num_filters: iterable of ints indicating num filters per layer (original model used the same for all layers)
    :param filter_dims: iterable of ints indicating kernel widths in the forward (analysis) computation; the reverse is
        used for backward (synthesis) computation
    :param sampling_rates: iterable of ints indicating downsampling rates in the forward (analysis) computation; the
        reverse is used for backward (synthesis) upsampling computation
    :return:
    """
    num_layers = len(num_filters)
    layers = []
    for i in range(num_layers):
        if i != num_layers - 1:
            layer = tfc.SignalConv2D(num_filters[i], (filter_dims[i], filter_dims[i]), name="layer_%d" % i,
                                     corr=True,
                                     strides_down=sampling_rates[i],
                                     padding="same_zeros", use_bias=True,
                                     activation=tfc.GDN(name="gdn_%d" % i))
        else:
            layer = tfc.SignalConv2D(num_filters[i] * 2, (filter_dims[i], filter_dims[i]), name="layer_%d" % i,
                                     corr=True,
                                     strides_down=sampling_rates[i],
                                     padding="same_zeros", use_bias=False,
                                     activation=None)  # twice the usual number of filters for combined output for mean and logvar
        layers.append(layer)

    inference_net = tf.keras.Sequential(layers)

    layers = []
    for i in reversed(range(num_layers)):
        j = num_layers - 1 - i
        if i != 0:
            layer = tfc.SignalConv2D(num_filters[i], (filter_dims[i], filter_dims[i]), name="layer_%d" % j,
                                     corr=False,
                                     strides_up=sampling_rates[i],
                                     padding="same_zeros", use_bias=True,
                                     activation=tfc.GDN(name="igdn_%d" % j, inverse=True))
        else:
            layer = tfc.SignalConv2D(3, (filter_dims[i], filter_dims[i]), name="layer_%d" % j,
                                     corr=False,
                                     strides_up=sampling_rates[i],
                                     padding="same_zeros", use_bias=True,
                                     activation=None)
        layers.append(layer)

    generative_net = tf.keras.Sequential(layers)

    return inference_net, generative_net


class MNISTVAE(GaussianVAE):
    def __init__(self, latent_dim):
        self.latent_dim = latent_dim  # we'll use fixed-size dense latents
        self.IMAGE_DIM = 28
        inference_net = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(self.IMAGE_DIM, self.IMAGE_DIM, 1)),
                tf.keras.layers.Conv2D(
                    filters=32, kernel_size=3, strides=(2, 2), activation='relu'),
                tf.keras.layers.Conv2D(
                    filters=64, kernel_size=3, strides=(2, 2), activation='relu'),
                tf.keras.layers.Flatten(),
                # No activation
                tf.keras.layers.Dense(latent_dim + latent_dim),
            ]
        )

        generative_net = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
                tf.keras.layers.Dense(units=7 * 7 * 32, activation=tf.nn.relu),
                tf.keras.layers.Reshape(target_shape=(7, 7, 32)),
                tf.keras.layers.Conv2DTranspose(
                    filters=64,
                    kernel_size=3,
                    strides=(2, 2),
                    padding="SAME",
                    activation='relu'),
                tf.keras.layers.Conv2DTranspose(
                    filters=32,
                    kernel_size=3,
                    strides=(2, 2),
                    padding="SAME",
                    activation='relu'),
                # No activation
                tf.keras.layers.Conv2DTranspose(
                    # filters=2, kernel_size=3, strides=(1, 1), padding="SAME"),  # double filters for mean and logvar
                    filters=1, kernel_size=3, strides=(1, 1), padding="SAME"),  # logvar will be fixed for now
            ]
        )
        super(MNISTVAE, self).__init__(prior=StandardGaussianPrior(), inference_net=inference_net,
                                       generative_net=generative_net, decode_sigmoid=True)


class CIFARVAE(GaussianVAE):
    def __init__(self, latent_dim):
        self.latent_dim = latent_dim  # we'll use fixed-size dense latents
        self.IMAGE_DIM = 32
        inference_net = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(self.IMAGE_DIM, self.IMAGE_DIM, 3)),
                tf.keras.layers.Conv2D(
                    filters=32, kernel_size=3, strides=(2, 2), activation='relu'),
                tf.keras.layers.Conv2D(
                    filters=64, kernel_size=3, strides=(2, 2), activation='relu'),
                tf.keras.layers.Conv2D(
                    filters=128, kernel_size=3, strides=(2, 2), activation='relu'),
                tf.keras.layers.Flatten(),
                # No activation
                tf.keras.layers.Dense(latent_dim + latent_dim),
            ]
        )

        generative_net = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
                tf.keras.layers.Dense(units=8 * 8 * 16, activation=tf.nn.relu),
                tf.keras.layers.Reshape(target_shape=(8, 8, 16)),
                tf.keras.layers.Conv2DTranspose(
                    filters=128,
                    kernel_size=3,
                    strides=(2, 2),
                    padding="SAME",
                    activation='relu'),
                tf.keras.layers.Conv2DTranspose(
                    filters=64,
                    kernel_size=3,
                    strides=(2, 2),
                    padding="SAME",
                    activation='relu'),
                # No activation
                tf.keras.layers.Conv2DTranspose(
                    filters=3, kernel_size=3, strides=(1, 1), padding="SAME"),
            ]
        )

        super(CIFARVAE, self).__init__(prior=StandardGaussianPrior(), inference_net=inference_net,
                                       generative_net=generative_net, decode_sigmoid=True)
