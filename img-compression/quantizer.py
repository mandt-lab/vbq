# For each channel, we assume all the pixel (x,y) dimensions are i.i.d., and solve a scalar quantization problem

import numpy as np

np.random.seed(0)

import tensorflow as tf

import utils



class ChannelwisePriorCDFQuantizer:
    def __init__(self, num_channels, max_bits_per_coord, float_type='float32', int_type='int32'):
        super().__init__()
        self.max_bits_per_coord = max_bits_per_coord
        self.num_channels = num_channels
        self.float_type = float_type
        self.int_type = int_type
        self.quantization_levels = 2 ** (max_bits_per_coord + 1) - 1  # 2^0 + 2^1 + 2^2 + ... + 2^max_bits_per_coord

        self.raw_code_length_entropy_models = None
        self.entropy_models = None

    def build_code_points(self, prior_model, **kwargs):
        N = self.max_bits_per_coord
        num_channels = self.num_channels
        float_type = self.float_type

        all_bin_floats = np.hstack([utils.n_bit_binary_floats(n) for n in range(N + 1)])  # xi representatives
        all_bin_floats_rep = np.repeat(all_bin_floats[:, None],
                                       num_channels, axis=1)  # quantization_levels x num_channels
        all_code_points = prior_model.inverse_cdf(all_bin_floats_rep, **kwargs)  # quantization_levels x num_channels
        all_code_points = tf.cast(all_code_points, float_type)
        all_code_points = tf.transpose(all_code_points)  # num_channels x quantization_levels
        self.all_code_points = all_code_points
        self.code_points_by_channel = tf.sort(all_code_points, axis=1)  # MUST BE SORTED!!!

        # code_points_by_bits[c][b] gives a length-2^b array of quantization/code points for channel c when using b bits
        code_points_by_bits = []
        for c in range(num_channels):
            code_points = []
            for n in range(N + 1):
                code_points.append(all_code_points[c, 2 ** n - 1: 2 ** (n + 1) - 1])  # n-bit code points
            code_points_by_bits.append(code_points)
        self.code_points_by_bits = code_points_by_bits

        # build grids of sorted quantization points represented by increasing number of raw bits, to make it faster to
        # search for "best" n-bit quantization when encoding
        search_grids = []
        for c in range(num_channels):
            search_grid = []
            for n in range(N + 1):
                if n == 0:
                    n_bit_grid = np.pad([code_points_by_bits[c][n][0]] * 2, (2 ** (N - 1) - 1,), 'edge')
                else:
                    n_bit_grid = np.pad(code_points_by_bits[c][n], (2 ** (N - 1) - 2 ** (n - 1),), 'edge')
                search_grid.append(n_bit_grid)
            search_grid = np.array(search_grid)
            search_grids.append(search_grid)
        search_grids = np.array(search_grids)  # num_channels x (N+1) x 2^N
        _search_grids = tf.constant(search_grids, dtype=float_type)
        self._search_grids = _search_grids  # C x (N+1) x 2^N

    def get_all_N_bit_intervals(self, Z):
        """

        :param Z: batch of latent representations of shape B x C
        :return: two tensors of shape C x (N+1) x B
        """
        N = self.max_bits_per_coord
        # backend has to be tensorflow >= 1.15 for searchsorted to work below (numpy's version doesn't support matrices)
        Z_repeated = tf.repeat(tf.transpose(Z)[:, None, :], N + 1, axis=1)  # C x (N+1) x B
        right_endpoint_idx = tf.searchsorted(self._search_grids, Z_repeated, side='left')  # search inner-most dim
        right_endpoint_idx = tf.clip_by_value(right_endpoint_idx, 0, 2 ** N - 1)  # C x (N+1) x B
        left_endpoint_idx = tf.clip_by_value(right_endpoint_idx - 1, 0, 2 ** N - 1)
        right_endpoints = tf.gather(self._search_grids, right_endpoint_idx, batch_dims=2)
        left_endpoints = tf.gather(self._search_grids, left_endpoint_idx, batch_dims=2)

        return left_endpoints, right_endpoints

    def build_entropy_models(self, X, vae, lambs, add_n_smoothing):
        N = self.max_bits_per_coord
        float_type = self.float_type

        posterior_means, posterior_logvars = vae.encode(X)
        posterior_vars = tf.exp(posterior_logvars)
        C = posterior_vars.shape[-1]  # number of latent channels
        assert C == self.num_channels
        batch_means, batch_vars = map(lambda r: tf.reshape(r, (-1, C)),
                                      [posterior_means, posterior_vars])  # num_samples x C
        batch_stds = batch_vars ** 0.5

        raw_code_length_entropy_models = dict()
        # first compress with raw number of bits (i.e., the number of binary decimal places of \hat xis)
        Z_hat_dict, raw_num_bits_dict = self.compress_batch_channel_latents(batch_means, batch_stds, lambs,
                                                                            return_np=False)  # each dict entry is B x C

        for lamb in lambs:
            Z_hat = Z_hat_dict[lamb]
            raw_num_bits = raw_num_bits_dict[lamb]

            # build entropy model for raw number of bits
            raw_num_bits_counts_by_channel = np.array([np.bincount(raw_num_bits[:, c], minlength=N + 1)
                                                       for c in range(C)], dtype=float_type)  # C x (N+1)
            raw_num_bits_counts_by_channel += add_n_smoothing
            raw_num_bits_freqs_by_channel = raw_num_bits_counts_by_channel / np.sum(raw_num_bits_counts_by_channel,
                                                                                    axis=1)[:, None]
            raw_code_length_entropy_model = -np.log2(raw_num_bits_freqs_by_channel)
            raw_code_length_entropy_models[lamb] = raw_code_length_entropy_model

        self.raw_code_length_entropy_models = raw_code_length_entropy_models

        # now run compression again to build entropy model for code points, but using the newly-built
        # raw_code_length_entropy_models that corrects for raw code length overhead when solving the encoding
        # optimization problems

        entropy_models = dict()
        Z_hat_dict, cl_corrected_num_bits_dict = self.compress_batch_channel_latents(batch_means, batch_stds, lambs,
                                                                                     return_np=False)  # each dict entry is B x C
        for lamb in lambs:
            Z_hat = Z_hat_dict[lamb]
            raw_num_bits = raw_num_bits_dict[lamb]
            #
            # # build entropy model for raw number of bits
            # raw_num_bits_counts_by_channel = np.array([np.bincount(raw_num_bits[:, c], minlength=N + 1)
            #                                            for c in range(C)], dtype=float_type)  # C x (N+1)
            # raw_num_bits_counts_by_channel += add_n_smoothing
            # raw_num_bits_freqs_by_channel = raw_num_bits_counts_by_channel / np.sum(raw_num_bits_counts_by_channel,
            #                                                                         axis=1)[:, None]
            # raw_code_length_entropy_model = -np.log2(raw_num_bits_freqs_by_channel)
            # raw_code_length_entropy_models[lamb] = raw_code_length_entropy_model

            # build entropy model for code points directly
            qidx = tf.searchsorted(self.code_points_by_channel, tf.transpose(Z_hat))  # C x B; requires TF
            assert tf.reduce_all(tf.equal(tf.gather(self.code_points_by_channel, qidx, batch_dims=1),
                                          tf.transpose(Z_hat)))  # qidx is the tensor of quantization indices
            qidx_counts_by_channel = np.array(
                [np.bincount(qidx[c], minlength=self.quantization_levels) for c in range(C)],
                dtype=float_type)  # num_latent_channels by quantization_levels
            qidx_counts_by_channel += add_n_smoothing
            qidx_freqs_by_channel = qidx_counts_by_channel / np.sum(qidx_counts_by_channel,
                                                                    axis=1)[:, None]
            entropy_model = -np.log2(qidx_freqs_by_channel)

            entropy_models[lamb] = entropy_model

        self.entropy_models = entropy_models

        return None

    @property
    def lambs(self):
        return list(sorted(self.entropy_models.keys()))

    def compress_batch_channel_latents(self, batch_means, batch_stds, lambs, **kwargs):
        Z = batch_means  # B x C
        N = self.max_bits_per_coord
        B, C = Z.shape
        float_type = self.float_type
        int_type = self.int_type
        left_endpoints, right_endpoints = self.get_all_N_bit_intervals(Z)  # C x (N+1) x B
        left_endpoints = tf.transpose(left_endpoints, [1, 2, 0])
        right_endpoints = tf.transpose(right_endpoints, [1, 2, 0])  # (N+1) x B x C

        if not self.raw_code_length_entropy_models:  # use naive (raw) code lengths, i.e., the number of floating points
            code_lengths = tf.range(N + 1, dtype=int_type)[:, None, None] * tf.ones((N + 1, B, C),
                                                                                    dtype=int_type)  # (N+1) x B x C
            code_lengths = tf.concat([code_lengths, code_lengths[1:]], axis=0)  # (2N+1) x B x C
        else:  # add code length overhead
            raw_code_lengths = tf.repeat(tf.range(N + 1, dtype=int_type)[:, None], C, axis=1)  # (N+1) x C
            raw_code_lengths = tf.cast(raw_code_lengths, self.raw_code_length_entropy_models[lambs[0]].dtype)
            code_lengths = []
            for lamb in lambs:
                code_lengths_for_lamb = raw_code_lengths + tf.transpose(self.raw_code_length_entropy_models[lamb])
                code_lengths_for_lamb = tf.repeat(code_lengths_for_lamb[:, None, :], B, axis=1)  # (N+1) x B x C
                code_lengths_for_lamb = tf.concat([code_lengths_for_lamb, code_lengths_for_lamb[1:]],
                                                  axis=0)  # (2N+1) x B x C
                code_lengths.append(code_lengths_for_lamb)
            code_lengths = tf.stack(code_lengths)

        # only considering right_endpoints with >=1 bits b/c 0-bit endpoints coincide for both left and right end points
        code_points = tf.concat([left_endpoints, right_endpoints[1:]], axis=0)  # (2N+1) x B x C

        fun = utils.curry_normal_logpdf(loc=batch_means, scale=batch_stds, ignore_const=True, backend=tf)
        Z_hat_dict, raw_num_bits_dict = utils.batch_quantize_indep_dims(Z.shape, code_points, code_lengths, fun,
                                                                        lambs=lambs, backend=tf, **kwargs)
        return Z_hat_dict, raw_num_bits_dict  # each dict entry is B x C

    def compress_latents(self, posterior_means, posterior_logvars, lambs):
        float_type = self.float_type

        posterior_vars = tf.exp(posterior_logvars)
        C = posterior_vars.shape[-1]  # number of latent channels
        assert C == self.num_channels
        batch_means, batch_vars = map(lambda r: tf.reshape(r, (-1, C)),
                                      [posterior_means, posterior_vars])  # num_samples x C
        batch_stds = batch_vars ** 0.5

        Z_hat_dict, raw_num_bits_dict = self.compress_batch_channel_latents(batch_means, batch_stds, lambs,
                                                                            return_np=False)
        out_keys = ('Z_hat', 'raw_num_bits', 'num_bits_cl', 'num_bits')
        output = {key: dict() for key in out_keys}

        for lamb in lambs:
            tmp_res_for_lamb = dict()

            Z_hat = Z_hat_dict[lamb]
            raw_num_bits = raw_num_bits_dict[lamb]

            tmp_res_for_lamb['Z_hat'] = Z_hat
            tmp_res_for_lamb['raw_num_bits'] = raw_num_bits

            # # calculate code length using raw num bits + codelength overhead approach
            # raw_code_length_entropy_model = self.raw_code_length_entropy_models[lamb]  # C x (N+1)
            # raw_code_length_overhead_num_bits = tf.gather(raw_code_length_entropy_model,
            #                                               tf.transpose(raw_num_bits), batch_dims=1)  # C x B
            # raw_code_length_overhead_num_bits = tf.transpose(raw_code_length_overhead_num_bits)  # B x C
            # num_bits_cl = tf.cast(raw_num_bits, float_type) + \
            #               tf.cast(raw_code_length_overhead_num_bits, float_type)  # B x C

            # calculate code length using direct entropy coding approach
            I = tf.searchsorted(self.code_points_by_channel, tf.transpose(Z_hat))  # C x B; requires TF
            # assert tf.reduce_all(tf.equal(tf.gather(self.code_points_by_channel, I, batch_dims=1),
            #                               tf.transpose(Z_hat)))
            entropy_model = self.entropy_models[lamb]  # C x quantization_levels
            num_bits = tf.gather(entropy_model, I, batch_dims=1)  # gather across innermost axis
            num_bits = tf.transpose(num_bits)  # B x C

            # tmp_res_for_lamb['num_bits_cl'] = num_bits_cl
            if self.raw_code_length_entropy_models:
                tmp_res_for_lamb['num_bits_cl'] = raw_num_bits
            tmp_res_for_lamb['num_bits'] = num_bits

            for key in out_keys:
                # tmp_res_for_lamb[key] = np.reshape(item, posterior_means.shape)
                output[key][lamb] = np.reshape(tmp_res_for_lamb[key],
                                               posterior_means.shape)  # same shape as latents; np.reshape moves to CPU

        return output

    def compress(self, X, vae, lambs, clip=True):
        posterior_means, posterior_logvars = vae.encode(X)
        output = self.compress_latents(posterior_means, posterior_logvars, lambs)
        Z_hat_dict = output['Z_hat']
        Z_hat_batch = tf.stack([Z_hat_dict[lamb] for lamb in lambs])  # len(lambs) by latent_shape
        Z_hat_batch_shape = Z_hat_batch.shape  # len(lambs) by posterior_means.shape
        Z_hat_flat_batch = tf.reshape(Z_hat_batch, [-1, *posterior_means.shape[1:]])

        X_hat_batch = tf.reshape(vae.decode(Z_hat_flat_batch),
                                 [len(lambs), *X.shape])  # 0th dimension corresponds to different lamb used
        if clip:  # pixel float values
            X_hat_batch = np.clip(X_hat_batch, 0, 1)
        X_hat_dict = {lamb: X_hat_batch[i] for i, lamb in enumerate(lambs)}
        output['X_hat'] = X_hat_dict
        return output


class UniformQuantizer:
    def __init__(self, quantization_levels, int_type=np.int32):
        super(UniformQuantizer, self).__init__()
        self.quantization_levels = quantization_levels
        self.int_type = int_type

    def fit(self, samples, add_n_smoothing=1.):
        min = np.min(samples)
        max = np.max(samples)
        N = self.quantization_levels
        delta = (max - min) / N

        offset = min + delta / 2  # location of the first quantization code point
        code_points = offset + delta * np.arange(N)

        self.min, self.max, self.delta, self.code_points = min, max, delta, code_points

        I = np.clip(np.floor((samples - min) / delta), 0, N - 1)  # how many deltas are we away from min?
        counts = np.bincount(I.astype(self.int_type), minlength=N)
        zero_bins = (counts == 0)
        if np.any(zero_bins):
            counts += add_n_smoothing
        # print('%d (%g%%) bins with no data, add %g smoothing' % (
        #                 np.sum(zero_bins), np.mean(zero_bins), add_n_smoothing))
        freqs = counts / len(samples)
        self.code_lengths = - np.log2(freqs)

    def quantize(self, samples):
        min = self.min
        delta = self.delta
        N = self.quantization_levels
        I = np.clip(np.floor((samples - min) / delta), 0, N - 1)  # how many deltas are we away from min?
        offset = min + delta / 2  # location of the first quantization code point
        quantized = offset + delta * I

        code_lengths = self.code_lengths
        num_bits = np.take(code_lengths, I.astype(self.int_type))

        return quantized, I, num_bits


class KmeansQuantizer:
    def __init__(self, quantization_levels, int_type=np.int32):
        super().__init__()
        self.quantization_levels = quantization_levels
        self.int_type = int_type

    def fit(self, samples, add_n_smoothing=1.):
        from sklearn.cluster import KMeans

        N = self.quantization_levels
        q = KMeans(n_clusters=N)
        samples = samples.reshape((-1, 1))  # required by sklearn.cluster.KMeans API
        q.fit(samples)
        # self.q = q
        self.code_points = q.cluster_centers_.ravel()  # not guaranteed sorted

        I = q.labels_
        counts = np.bincount(I.astype(self.int_type), minlength=N)
        zero_bins = (counts == 0)
        if np.any(zero_bins):
            counts += add_n_smoothing
        # print('%d (%g%%) bins with no data, add %g smoothing' % (
        #                 np.sum(zero_bins), np.mean(zero_bins), add_n_smoothing))
        freqs = counts / len(samples)
        self.code_lengths = - np.log2(freqs)

    def quantize(self, samples):
        from scipy.cluster import vq
        I = vq.vq(samples, self.code_points)[0]
        quantized = np.take(self.code_points, I)
        code_lengths = self.code_lengths
        num_bits = np.take(code_lengths, I.astype(self.int_type))

        return quantized, I, num_bits


class ChannelwiseSimpleQuantizer:
    def __init__(self, scalar_quantizer_type, num_channels, quantization_levels):
        super().__init__()
        self.quantization_levels = quantization_levels
        self.num_channels = num_channels
        self._quantizers = [scalar_quantizer_type(quantization_levels) for _ in range(num_channels)]

    def fit_latents(self, posterior_means, add_n_smoothing):
        C = posterior_means.shape[-1]  # number of latent channels
        assert C == self.num_channels
        # means, vars = map(lambda r: np.reshape(r, (-1, C)), [posterior_means, posterior_vars])  # num_samples x C
        means = np.reshape(posterior_means, (-1, C))  # num_samples x C
        num_samples = len(means)

        code_points = []  # will be C x quantization_levels
        code_lengths = []  # will be C x quantization_levels
        for c in range(C):
            q = self._quantizers[c]
            q.fit(means[:, c], add_n_smoothing)
            code_points.append(q.code_points)
            code_lengths.append(q.code_lengths)

        code_points = np.array(code_points)
        code_lengths = np.array(code_lengths)

        self.code_points = code_points
        self.code_lengths = code_lengths

    def fit(self, X, vae, add_n_smoothing):
        """

        :param X: a batch of images (same format as used in training the VAE)
        :return:
        """
        posterior_means, posterior_logvars = vae.encode(X)
        self.fit_latents(posterior_means, add_n_smoothing)

    def compress_latents(self, posterior_means):
        C = posterior_means.shape[-1]  # number of latent channels
        assert C == self.num_channels
        # means, vars = map(lambda r: np.reshape(r, (-1, C)), [posterior_means, posterior_vars])  # num_samples x C
        means = np.reshape(posterior_means, (-1, C))  # num_samples x C
        # num_samples = len(means)
        quantized, I, num_bits = [], [], []
        for c in range(C):
            q = self._quantizers[c]
            quantized_, I_, num_bits_ = q.quantize(means[:, c])
            quantized.append(quantized_)
            I.append(I_)
            num_bits.append(num_bits_)

        quantized, I, num_bits = [np.array(r).transpose() for r in [quantized, I, num_bits]]  # num_samples x C

        Z_hat = quantized
        Z_hat = np.reshape(Z_hat, posterior_means.shape)
        num_bits = np.reshape(num_bits, posterior_means.shape)  # has same shape as latents

        output = dict(Z_hat=Z_hat, num_bits=num_bits)
        return output

    def compress(self, X, vae, clip=True):
        posterior_means, posterior_logvars = vae.encode(X)
        output = self.compress_latents(posterior_means)
        Z_hat = output['Z_hat']
        X_hat = vae.decode(Z_hat)
        if clip:  # pixel float values
            X_hat = np.clip(X_hat, 0, 1)
        output['X_hat'] = X_hat
        return output


class ChannelwiseSimpleQuantizerWrapper:
    """
    Thin wrapper around a list of ChannelwiseSimpleQuantizer, each with a different quantization_levels, to create the
    illusion of a single 'quantizer' object that conforms with the API of utils.evaluate_compression_quantizer
    """

    def __init__(self, scalar_quantizer_type, num_channels, quantization_levels):
        super().__init__()
        self.quantization_levels = quantization_levels
        self.num_channels = num_channels
        self._quantizers = [ChannelwiseSimpleQuantizer(scalar_quantizer_type, num_channels, l) for l in
                            quantization_levels]

    def fit(self, X, vae, add_n_smoothing):
        """

        :param X: a batch of images (same format as used in training the VAE)
        :return:
        """
        posterior_means, posterior_logvars = vae.encode(X)
        for i, q in enumerate(self._quantizers):
            # q.fit(X, vae, add_n_smoothing)
            q.fit_latents(posterior_means, add_n_smoothing)
            # print('fit quantizer with %d levels' % (self.quantization_levels[i]))

    def _compress(self, X, vae, clip=True):
        posterior_means, posterior_logvars = vae.encode(X)
        output = {'Z_hat': {}, 'num_bits': {}}
        quantization_levels = self.quantization_levels
        for i, l in enumerate(quantization_levels):
            q = self._quantizers[i]
            tmp = q.compress_latents(posterior_means)
            for field in output:
                output[field][l] = tmp[field]

        Z_hat_dict = output['Z_hat']  # list of len(quantization_levels)
        Z_hat_batch = np.stack([Z_hat_dict[l] for l in quantization_levels])  # len(quantization_levels) by latent_shape
        Z_hat_flat_batch = np.reshape(Z_hat_batch, [-1, *posterior_means.shape[1:]])

        X_hat_batch = np.reshape(vae.decode(Z_hat_flat_batch),
                                 [len(quantization_levels),
                                  *X.shape])  # 0th dimension corresponds to different quantization_levels used
        if clip:  # pixel float values
            X_hat_batch = np.clip(X_hat_batch, 0, 1)
        X_hat_dict = {l: X_hat_batch[i] for i, l in enumerate(quantization_levels)}
        output['X_hat'] = X_hat_dict

        return output

    def compress(self, X, vae, quantization_levels, clip=True):
        # dud to conform with utils.evaluate_compression_quantizer API
        assert quantization_levels == self.quantization_levels
        return self._compress(X, vae, clip)

