import numpy as np


def round_float(x, decimals, mode):
    """
    Given floating point array x, round it to the nearest specified number of decimal places
    :param x:
    :param decimals:
    :param mode: 'ceil' or 'floor', either rounding up or down
    :return:
    e.g. round_float(0.113, 2, 'ceil') = 0.12, round_float(0.115, 2, 'floor') = 0.11
    """
    if mode in ('floor', 'down'):
        round_fun = np.floor
    elif mode in ('ceil', 'up'):
        round_fun = np.ceil
    scale = 10. ** decimals
    res = round_fun(x * scale) / scale
    res = np.round(res, decimals=10)  # to reduce floating point error, 90.99999999999999 => 91
    return res


def n_bit_binary_floats(n):
    return [i * 2 ** (-n) + 2 ** (-n - 1) for i in range(2 ** n)]


def get_n_bit_interval(x, n):
    """
    Get the end points of the smallest interval containing x, using grid points of the form
    [0*2^{-n} + 2^{-n-1}, 1*2^{-n} + 2^{-n-1}, 2*2^{-n} + 2^{-n-1}, ..., (2^n-1)*2^{-n} + 2^{-n-1}]
    e.g., when n = 2, the grid points are [0.001, 0.011, 0.101, 0.111], so given x = 0.4375 (0.0111 in binary),
    the interval [0.011, 0.101] (actually it's decimal representation, [0.375, 0.625]) will be returned.
    In our smart encoding, a terminating 1 is always assumed, so we'll only have to code ['0.01', '0.10'] in practice.
    :param x: decimal floating point in [0, 1]
    :param n: int, bit width
    :return: (left endpoint in decimal float, right_endpoint in decimal float)
    """
    # assert 0 <= x <= 1
    if n == 0:
        return [0.5, 0.5]

    width = 2 ** (-n)
    offset = width / 2  # 2 ** (-n - 1)
    leftmost_grid_point = offset
    rightmost_grid_point = 1 - offset
    if x < leftmost_grid_point:
        left = right = leftmost_grid_point
    elif x > rightmost_grid_point:
        left = right = rightmost_grid_point
    else:
        x_shifted = x - offset  # put x in the 'ordinary' grid system with grid points [0*2^{-n}, 1*2^{-n}, ..., (2^n-1)*2^{-n}]
        left = truncate_float_to_n_bits(x_shifted, n)[0]  # truncate x_shifted \in [0,1] to first n bits
        left += offset  # corresponds to appending 1 in the binary string representation
        right = left + width

    # print(float_dec2bin(left), float_dec2bin(right))
    return (left, right)


def truncate_float_to_n_bits(x, n):  # TODO: slow, cythonize
    """
    Given decimal float x \in [0, 1], find the first n places of its binary representation ("bits") and the
    corresponding value in decimal float ("x_hat")
    :param x:
    :param n:
    :return:
    """
    bits = ''
    rem = float(x)
    for i in range(1, n + 1):
        bit_width = 2 ** (-i)
        if rem < bit_width:  # can't divide into the current bit-width
            bits += '0'
        else:
            bits += '1'
            rem -= bit_width
    x_hat = x - rem
    return x_hat, bits


# print(get_n_bit_interval(0.4375, 2))
# print(get_n_bit_interval(0.004375, 2))
# print(get_n_bit_interval(0.04375, 5))

from scipy.stats import norm


def encode_mode_1d(f, mode, nbits, squash, unsquash):
    # given function f and its full-precision mode, encode it such that squash(\hat mode) is a length n binary float
    # that maximizes fun(unsquash(squash(\hat mode))
    xi = squash(mode)  # float in (0, 1)
    left, right = get_n_bit_interval(xi, nbits)  # in terms of decimal floats

    mode_hat = unsquash(left)
    f_hat = f(mode_hat)
    if right != left:
        mode_right = unsquash(right)
        f_right = f(mode_right)
        if f_right > f_hat:
            f_hat = f_right
            mode_hat = mode_right

    return mode_hat, f_hat


def encode_mode_dp(f, mode, nbits, squash, unsquash, zero_bit_mode_hat):
    K = len(f)
    N = nbits
    T = np.empty([K, N + 1])  # DP value table
    T[:] = np.nan
    mode_hat_table = np.empty([K, N + 1])  # (k,n)th entry is the n-bit quantized value that maximizes f_k
    mode_hat_table[:] = np.nan  # so that we know when things have gone wrong

    # (k,n)th entry gives index (k',n') of the subproblem on the optimal path
    best_subproblem_idx_table = np.empty([K, N + 1, 2], dtype=int)

    def f_hat(k, n):
        if n == 0:
            mode_hat_k = zero_bit_mode_hat[k]
            f_hat_k = f[k](mode_hat_k)
        else:
            mode_hat_k, f_hat_k = encode_mode_1d(f[k], mode[k], n, squash[k], unsquash[k])
        mode_hat_table[k, n] = mode_hat_k
        return f_hat_k

    def get_entry(k, n):
        val = T[k, n]
        if not np.isnan(val):
            return val
        # not yet calculated
        if k == 0:  # base case
            val = f_hat(k, n)
        else:
            f_hat_ks = np.array([f_hat(k, m) for m in range(n + 1)])  # 0, 1, 2, ..., n
            T_k_minus_1 = np.array([get_entry(k - 1, m) for m in range(n, -1, -1)])
            subproblem_values = f_hat_ks + T_k_minus_1
            best_m = np.argmax(subproblem_values)  # optimal num bits to use for f_k
            best_subproblem_idx_table[k, n] = (k - 1, n - best_m)

            val = subproblem_values[best_m]

        T[k, n] = val
        return val

    obj = get_entry(K - 1, N)

    # tracing back to find the optimal solution
    k = K - 1
    num_bits_for_problem = N
    num_bits = []
    while k > 0:
        best_subproblem_idx = best_subproblem_idx_table[k, num_bits_for_problem]
        num_bits = [num_bits_for_problem - best_subproblem_idx[1]] + num_bits  # used by coordinate k
        k, num_bits_for_problem = best_subproblem_idx

    num_bits = [N - sum(num_bits)] + num_bits  # need to set coordinate 0
    num_bits = np.array(num_bits)
    mode_hat = np.array([mode_hat_table[k, n] for k, n in zip(range(K), num_bits)])

    return mode_hat, obj, num_bits


def encode_mode(f, mode, lamb, squash, unsquash, zero_bit_mode_hat, max_bits_per_coord=32):
    """

    :param f: length K list of functions whose sum is to be maximized (currently this corresponds to q(z|x))
    :param mode: length K vector of mode of f
    :param lamb: penalty coeff
    :param squash: length K list of functions, each R -> [0, 1]
    :param unsquash:
    :param zero_bit_mode_hat: special mode_hat to use when using 0 bits (corresponding to mode of p(z))
    :param max_bits_per_coord:
    :return:
    """
    PATIENCE = 3  # number of iterations where objective g(b) hasn't increased before stop trying larger b
    K = len(f)
    mode_hat = np.empty_like(mode)
    obj = 0
    num_bits = np.array([0] * K)
    for k in range(K):
        bad_count = 0
        mode_hat_k = None
        best_mode_hat_k = None
        best_g_hat_k = -float('inf')  # g_k(b_k) = fun_k(b_k) - lamb * b_k
        best_b = None
        for b in range(max_bits_per_coord + 1):
            if b == 0:
                mode_hat_k = zero_bit_mode_hat[k]
                g_hat_k = f[k](mode_hat_k)  # - lamb * 0
            else:
                mode_hat_k, f_hat_k = encode_mode_1d(f[k], mode[k], b, squash[k], unsquash[k])
                g_hat_k = f_hat_k - lamb * b

            if g_hat_k > best_g_hat_k:
                bad_count = 0
                best_g_hat_k = g_hat_k
                best_mode_hat_k = mode_hat_k
                best_b = b
            else:  # no improvement
                bad_count += 1

            if bad_count == PATIENCE:
                break
        mode_hat[k] = best_mode_hat_k
        obj += best_g_hat_k
        num_bits[k] = best_b

    return mode_hat, obj, num_bits


from numba import jit


# additional flags (nogil, cache) for more potential speedup: https://numba.pydata.org/numba-doc/dev/user/jit.html
@jit(nopython=True, cache=True)  # , parallel=True)
def get_all_N_bit_intervals(x, N, left_endpoints, right_endpoints):
    """
    Given length K vector x in the K-dimensional unit hypercube, and integer N >= 1, compute the n-bit interval
    containing x[k] for bit budget n=0,1,...,N and dimension k=0,1,..., K-1 separately.
    :param x:
    :param N
    :param left_endpoints: (N+1) x K  array, (n,k)th entry contains left end point of n-bit interval containing x[k]
    :param right_endpoints: (N+1) x K array, (n,k)th entry contains left right point of n-bit interval containing x[k]
    :return:
    """
    for k in range(x.shape[0]):
        # x_k = x[k]
        for n in range(N + 1):
            if n == 0:  # special case; interval doesn't actually contain x_k!!
                left = right = 0.5
                left_endpoints[n, k] = left
                right_endpoints[n, k] = right
                continue

            x_k = x[k]
            width = 2. ** (-n)
            offset = width * 0.5  # 2 ** (-n - 1)
            leftmost_grid_point = offset
            rightmost_grid_point = 1. - offset
            if x_k < leftmost_grid_point:
                left = right = leftmost_grid_point
            elif x_k > rightmost_grid_point:
                left = right = rightmost_grid_point
            else:
                x_k_shifted = x_k - offset
                # truncate x_k_shifted to first n bits
                # left = truncate_float_to_n_bits(x_k_shifted, n) + offset
                # copied from truncate_float_to_n_bits
                rem = x_k_shifted
                for i in range(1, n + 1):
                    bit_width = 2. ** (-i)
                    diff = rem - bit_width
                    if diff >= 0.:  # can divide into the current bit-width
                        rem = diff
                x_hat = x_k_shifted - rem
                # x_hat = truncate_float_to_n_bits(x_k_shifted, n)[0]
                left = x_hat + offset
                right = left + width
            left_endpoints[n, k] = left
            right_endpoints[n, k] = right


def encode_vectorized(fun, z, lamb, squash, unsquash, max_bits_per_coord=16):
    """
    Encode the mode z of a function, fun, as best as possible. Specifically, the caller provides three vector-valued
    functions, fun, squash, and unsquash, where fun is the objective function (a separate one in each dimension), squash
    maps full precision float z to [0, 1], unsquash (generally the inverse of squash) maps [0, 1] to (-∞, ∞), we attempt
    to minimize -\sum_k {fun(unsquash(xi_k) + lamb * bits(xi_k)}, where for each k, xi_k \in [0, 1] is a (decimal) float
    representable as a binary float, bits(xi_k) is the length (number of bits) of the binary representation of xi_k.
    We can solve the problem separately in each dimension: for bit budget = n, examine all xi_k with bits(xi_k) = n that
    maximizes fun_k(unsquash(xi_k)); fortunately we already know z_k is the full-precision mode of fun_k, so we simply
    need to test the two values of xi_k, left_endpoint and right_endpoint, such that both are representable with n bits
    (i.e., bits(left_endpoint) = bits(right_endpoint) = n), and left_endpoint is the largest such point less than z_k,
    and right_endpoint is the smallest such point greater than or equal to z_k; in other words, [left_endpoint, right_
    endpoint] is the "smallest" (2)n-bit interval containing z_k.
    :param fun: R^K -> R^K, should be able to broadcast to handle R^{MxK} -> R^{MxK} (applied across M rows)
    :param z: K-dimensional floating point numpy array, assumed to maximize fun
    :param squash: element-wise fun
    :param unsquash: element-wise fun
    :param max_bits_per_coord:
    :return:
    """
    K = len(z)
    np_arange_K = np.arange(K)
    N = max_bits_per_coord
    left_endpoints = np.empty([N + 1, K])
    right_endpoints = np.empty([N + 1, K])
    xi = squash(z)
    get_all_N_bit_intervals(xi, N, left_endpoints, right_endpoints)
    endpoints = np.stack([left_endpoints, right_endpoints], axis=0)
    unsquashed_endpoints = unsquash(endpoints)  # 2 x (N+1) x K
    F = fun(unsquashed_endpoints)  # 2 x (N+1) x K
    argmax = np.argmax(F, axis=0)  # (N+1) x K
    best_endpoints_idx = (argmax, np.arange(N + 1)[:, np.newaxis], np_arange_K[np.newaxis, :])
    F_max = F[best_endpoints_idx]  # (N+1) x K
    best_unsquashed_endpoints = unsquashed_endpoints[best_endpoints_idx]  # (N+1) x K
    regularized_F_max = F_max - lamb * np.arange(N + 1)[:, np.newaxis]  # (N+1) x K
    best_num_bits = np.argmax(regularized_F_max, axis=0)
    z_hat = best_unsquashed_endpoints[best_num_bits, np_arange_K]
    best_endpoints = endpoints[best_endpoints_idx]
    xi_hat = best_endpoints[best_num_bits, np_arange_K]
    f_z_hat = regularized_F_max[best_num_bits, np_arange_K]
    result = dict(z_hat=z_hat, score=np.sum(f_z_hat), num_bits=best_num_bits, xi_hat=xi_hat)
    return result


def curry_normal_logpdf(loc, scale, ignore_const=False, backend=np):
    """

    :param loc: K-vector
    :param scale: K-vector
    :return:
    """
    if backend is np:
        def f(z):  # input arg can be any tensor with K in the last dimension
            return norm.logpdf(z, loc=loc, scale=scale)
    else:  # assume backend is TF eager
        if ignore_const:
            def f(z):
                return -0.5 * ((z - loc) / scale) ** 2
        else:
            const = - backend.log(scale) - 0.5 * np.log(2 * np.pi)

            def f(z):
                return -0.5 * ((z - loc) / scale) ** 2 + const

    return f


def quantize_indep_dims(z, code_points, code_lengths, fun, lamb, backend=np):
    """
    Quantize a K-dimensional vector, z, with independent dimensions, to maximize RD objective
    fun(\hat z) - \lambda code_length(\hat z), for each dimension separately (in parallel).
    Here fun(\hat z) is conceptually a surrogate function for negative distortion(z, \hat z).
    :param z: K-dimensional floating point numpy array, assumed to maximize fun
    :param code_points: KxM-dimensional float numpy array of quantization representatives, one M-array for each of the K
    dimensions. MUST BE SORTED in increasing order in the innermost (code points) dimension!!!
    :param code_lengths: KxM-dimensional int numpy array of code lengths corresponding to code_points.
    :param fun: R^K -> R^K, fun(\hat z) computes coordinate-wise negative distortion to the target vector z. Should be
    able to broadcast to handle batch input like R^{... x K} -> R^{... x K} (note that the same target vector z is used
    to compute negative distortion w.r.t every R^K element of the input batch).
    :param lamb: scalar parameter that controls R-D trade-off.
    :return:
    """
    # backend = np
    scores = fun(backend.transpose(code_points)) - lamb * backend.transpose(code_lengths)  # M x K
    code_indices = backend.argmax(scores, axis=0)  # K-dim, values in [0, M-1]
    K, M = code_points.shape
    if backend is np:
        range_K = np.arange(K)
        z_hat = code_points[range_K, code_indices]  # K-dim
        num_bits = code_lengths[range_K, code_indices]  # K-dim
    else:  # assuming backend is Tensorflow
        range_K = backend.range(K, dtype='int64')
        idx = backend.transpose([range_K, code_indices])  # https://stackoverflow.com/a/42629665/4115369
        z_hat = backend.gather_nd(code_points, idx)
        num_bits = backend.gather_nd(code_lengths, idx)
        z_hat = z_hat.numpy()
        num_bits = num_bits.numpy()
    return z_hat, num_bits


def batch_quantize_indep_dims(Z_shape, code_points, code_lengths, fun, lambs, backend=np, return_np=True):
    """
    Quantize a batch of vectors, treating each vector dimension independently.
    :param Z_shape: B x K, i.e., batch_size x num_dims
    :param code_points: K x M, i.e., num_dims x num_quantization_levels; alternatively, a 3D tensor of M x B x K
    can be used when there's a different set of code_points for each element of the input batch
    :param code_lengths: K x M, i.e., num_dims x num_quantization_levels; alternatively, a 3D tensor of M x B x K
    can be used when there's a different set of code_lengths for each element of the input batch
    :param fun: R^{BxK} -> R^{BxK}, fun(\hat z) computes coordinate-wise negative distortion to the target batch vectors
    Z. Should be able to broadcast to handle input like R^{... B x K} -> R^{... B x K} (note that the same target
    batch vectors Z is used to compute negative distortion w.r.t every R^{... B x K} element of the input tensor).
    :param lambs: a list of lambda values
    :param backend:
    :return: two dicts, Z_hat and num_bits, such that Z_hat[lamb], num_bits[lamb] gives the quantized values and raw
     number of bits for the lamb used.
    """
    B, K = Z_shape
    if len(code_points.shape) == 3:
        P = code_points  # M x B x K; each slice corresponds to a set of scalar quantization points
        L = code_lengths
    else:
        assert len(code_points.shape) == 2
        P = backend.repeat(backend.transpose(code_points)[:, None, :], B, axis=1)  # M x B x K
        L = backend.repeat(backend.transpose(code_lengths)[:, None, :], B, axis=1)  # M x B x K
    fun_P = fun(P)
    L_cast = L if backend is np else backend.cast(L, P.dtype)
    Z_hat_dict = dict()
    num_bits_dict = dict()

    for i, lamb in enumerate(lambs):
        if len(L.shape) == 4:  # outermost dimension corresponds to different lambs
            scores = fun_P - lamb * L_cast[i]
        else:
            scores = fun_P - lamb * L_cast
        # if backend is np:
        #     scores = fun(P) - lamb * L
        # else:
        #     scores = fun(P) - lamb * backend.cast(L, P.dtype)
        code_indices = backend.argmax(scores, axis=0)  # B x K, values in [0, M-1]
        if backend is np:
            Z_hat = code_indices.choose(P)
            num_bits = code_indices.choose(L)
        else:  # assuming backend is Tensorflow
            P_flat = backend.reshape(P, [-1, B * K])  # M x (B*K)
            if len(L.shape) == 4:  # outermost dimension corresponds to different lambs
                L_flat = backend.reshape(L[i], [-1, B * K])  # M x (B*K)
            else:
                L_flat = backend.reshape(L, [-1, B * K])  # M x (B*K)
            code_indices_flat = backend.reshape(code_indices, (B * K,))
            range_BK = backend.range(B * K, dtype='int64')
            idx = backend.transpose([code_indices_flat, range_BK])  # https://stackoverflow.com/a/42629665/4115369
            Z_hat = backend.reshape(backend.gather_nd(P_flat, idx), [B, K])
            num_bits = backend.reshape(backend.gather_nd(L_flat, idx), [B, K])
            if return_np:
                Z_hat = Z_hat.numpy()
                num_bits = num_bits.numpy()

        Z_hat_dict[lamb] = Z_hat
        num_bits_dict[lamb] = num_bits

    return Z_hat_dict, num_bits_dict  # each dict entry has same shape as input Z


# utility methods for tensorflow training

def tf_read_img(tf, filename):
    """Loads a image file as float32 HxWx3 array; tested to work on png and jpg images."""
    string = tf.read_file(filename)
    image = tf.image.decode_image(string, channels=3)
    image = tf.cast(image, tf.float32)
    image /= 255
    return image


def tf_convert_float_to_uint8(tf, image):
    image = tf.round(image * 255)
    image = tf.saturate_cast(image, tf.uint8)
    return image


def tf_write_png(tf, filename, image):
    """Saves an image to a PNG file."""
    image = tf_convert_float_to_uint8(image)
    string = tf.image.encode_png(image)
    return tf.write_file(filename, string)


def read_npy_file_helper(file_name_in_bytes):
    # data = np.load(file_name_in_bytes.decode('utf-8'))
    data = np.load(file_name_in_bytes)  # turns out this works too without decoding to str first
    # assert data.dtype is np.float32   # needs to match the type argument in the caller tf.data.Dataset.map
    return data


# Image processing utils
def convert_RGB_to_L(x, keep_channel_dim=False, dtype=np.uint8):
    """
    Given RGB array in [0, 255] (uint8), convert to L (luma) component using JPEG conversion formula:
    https://en.wikipedia.org/wiki/YCbCr#JPEG_conversion
    i.e., y = x[...,0] * 0.299 + x[...,1] * 0.587 + x[...,2] * 0.114
    Result might differ slightly with that computed by orig.convert('L') due to floating point error & rounding to ints)
    :param x:
    :return:
    """
    y = x[..., 0] * 0.299 + x[..., 1] * 0.587 + x[..., 2] * 0.114
    y = np.round(y, decimals=10)  # to reduce floating point error, 90.99999999999999 => 91
    y = y.astype(dtype)
    if keep_channel_dim:
        y = np.expand_dims(y, axis=-1)  # append channel dim for grayscale img
    return y


def convert_RGB_to_CbCr(x, dtype=np.uint8):
    """
    Given RGB array in [0, 255] (uint8), convert to Cb and Cr components using JPEG conversion formula:
    https://en.wikipedia.org/wiki/YCbCr#JPEG_conversion
    Result might differ slightly with that computed by orig.convert('L') due to floating point error & rounding to ints)
    :param x:
    :return:
    """
    cb = 128 - 0.168736 * x[..., 0] - 0.331264 * x[..., 1] + 0.5 * x[..., 2]
    cr = 128 + 0.5 * x[..., 0] - 0.418688 * x[..., 1] - 0.081312 * x[..., 2]

    def proc(y):
        y = np.round(y, decimals=10)  # to reduce floating point error, 90.99999999999999 => 91
        y = np.clip(y, 0., 255.)
        y = y.astype(dtype)
        return y

    cb, cr = list(map(proc, (cb, cr)))
    C = np.stack((cb, cr))
    return C


def convert_to_db(d):
    # d should be float between 0 and 1
    return -10 * np.log10(1 - d)  # BMSHJ ICLR 2018, page 8


def evaluate_compression_quantizer(quantizer, vae, test_img_files, settings, model_input_float_type='float32',
                                   return_reconstructions=False, use_tf=False):
    """
    Compress a list of images using trained model (tf must be imported in eager mode) and report statistics.
    Currently assume input images all have different sizes (TODO: add flag allowing same-sized images for faster batch
    processing).
    :param model:
    :param test_img_files:
    :param settings: an iterable of scalars of compression settings ('lambs' for prior CDF quantizer and
    'quantization_levels' for uniform/kmeans quantizer)
    :param mode: if 'L', only luma components of images are used for comparisons; default 'RGB'
    :return:
    """
    from PIL import Image
    import img_comparison_metrics

    N = len(test_img_files)
    M = len(settings)

    results = {}
    results['reconstructions'] = []
    results['B'] = np.empty([N, M])  # total num bits
    results['BPP'] = np.empty([N, M])
    results['BPPCL'] = np.empty(
        [N, M])  # bpp computed with raw code length/num bits + entropy-coding code length approach
    results['BPL'] = np.empty([N, M])  # num bits per latent dim

    modes = ('RGB', 'Luma', 'Chroma')
    metrics = ('MSE', 'PSNR', 'MS-SSIM')
    if use_tf:
        metrics += ('SSIM',)
    results.update({'%s (%s)' % (metric, mode): np.empty([N, M]) for mode in modes for metric in metrics})

    for n, f in enumerate(test_img_files):  # TODO: parallelize
        orig = Image.open(f)
        img = orig.convert('RGB')  # get rid of alpha dimension from .png
        num_pixels = orig.size[0] * orig.size[1]
        x = np.asarray(img)  # H x W x 3, uint8
        x_float = (x / 255.)  # model requires float input
        X = x_float[None, ...].astype(model_input_float_type)  # batch of one; input to model
        tmp = quantizer.compress(X, vae, settings, clip=True)  # X is a batch of one

        img_hats = []
        x_hats = []
        for m, lamb in enumerate(settings):
            num_bits = tmp['num_bits'][lamb][0]  # batch of one
            nbits = np.sum(num_bits)
            results['B'][n, m] = np.sum(num_bits)
            results['BPP'][n, m] = nbits / num_pixels
            results['BPL'][n, m] = nbits / num_bits.size
            num_bits_cl = tmp.get('num_bits_cl', tmp['num_bits'])
            results['BPPCL'][n, m] = np.sum(num_bits_cl[lamb][0]) / num_pixels
            X_hat = tmp['X_hat'][lamb][0]  # H x W x 3 float
            x_hat = np.clip(np.round(X_hat * 255), 0, 255).astype(np.uint8)  # "saturate cast", H x W x 3, uint8
            img_hat = Image.fromarray(x_hat)
            img_hats.append(img_hat)
            x_hats.append(x_hat)

        # prepare for img quality comparisons
        # x = np.asarray(img)  # RGB H x W x 3, uint8
        x_yc = np.asarray(img.convert('YCbCr'))  # YCbCr, H x W x 3, uint8
        # x_hats = np.array([np.asarray(img_hat) for img_hat in img_hats])  # RGB, M x H x W x 3, uint8
        x_hats = np.asarray(x_hats)  # RGB, M x H x W x 3, uint8
        x_hats_yc = np.array(
            [np.asarray(img_hat.convert('YCbCr')) for img_hat in img_hats])  # YCbCr, M x H x W x 3, uint8

        if return_reconstructions:
            results['reconstructions'].append(x_hats)

        comparison_modes = ('RGB', 'Luma', 'Chroma')
        for mode in comparison_modes:
            if mode == 'RGB':
                x_comp = x
                # xs_comp = np.repeat(x[None, ...], repeats=N, axis=0)  # M x H x W x 3
                # x_hats = np.array([np.asarray(img_hat) for img_hat in img_hats])
                x_hats_comp = x_hats
            elif mode == 'Luma':
                x_comp = x_yc[..., 0:1]
                # xs_comp = np.repeat(x_yc[None, :, :, 0:1], repeats=N, axis=0)  # M x H x W x 1
                x_hats_comp = x_hats_yc[..., 0:1]  # M x H x W x 1
            elif mode == 'Chroma':
                x_comp = x_yc[..., 1:]
                # xs_comp = np.repeat(x_yc[None, :, :, 1:], repeats=N, axis=0)  # M x H x W x 2
                x_hats_comp = x_hats_yc[..., 1:]  # M x H x W x 2
            else:
                raise NotImplementedError
            xs_comp = np.repeat(x_comp[None, ...], repeats=M, axis=0)  # same shape as x_hats_comp

            if not use_tf:
                # mses = img_comparison_metrics.mse(xs_comp, x_hats_comp)
                results['MSE (%s)' % mode][n] = img_comparison_metrics.mse(xs_comp, x_hats_comp)
                # psnrs = img_comparison_metrics.psnr(xs_comp, x_hats_comp, max_val=255)
                results['PSNR (%s)' % mode][n] = img_comparison_metrics.psnr(xs_comp, x_hats_comp, max_val=255)
                # msssims = img_comparison_metrics.ms_ssim(xs_comp, x_hats_comp, max_val=255)
                results['MS-SSIM (%s)' % mode][n] = img_comparison_metrics.ms_ssim(xs_comp, x_hats_comp, max_val=255)
            else:
                import tensorflow as tf
                xs_comp = tf.convert_to_tensor(xs_comp)
                x_hats_comp = tf.convert_to_tensor(x_hats_comp)

                mses = tf.reduce_mean(
                    tf.squared_difference(tf.cast(xs_comp, 'float32'), tf.cast(x_hats_comp, 'float32')), axis=(1, 2, 3))
                psnrs = tf.image.psnr(x_hats_comp, xs_comp, 255)
                ssims = tf.image.ssim(x_hats_comp, xs_comp, 255)
                msssims = tf.image.ssim_multiscale(x_hats_comp, xs_comp,
                                                   255)  # can crash: https://github.com/tensorflow/tensorflow/issues/33840
                # Also crashes with "Invalid argument: Computed output size would be negative" when input images are too small (this
                # is expected behavior: the default setting (https://www.tensorflow.org/api_docs/python/tf/image/ssim_multiscale)
                # uses 5 2x downsampling scales and a filter size of 11, so for a small image, say 64x64, 3 2x scaling reduce it to
                # 8x8, which is smaller than kernel size 11)
                tensors = [mses,
                           psnrs,
                           ssims,
                           msssims,
                           ]
                if not tf.executing_eagerly():
                    # if sess is not None:
                    with tf.Session() as sess:
                        # with sess as sess:
                        tensors = sess.run(tensors)
                [mses, psnrs, ssims, msssims] = tensors
                results['MSE (%s)' % mode][n] = mses
                results['PSNR (%s)' % mode][n] = psnrs
                results['SSIM (%s)' % mode][n] = ssims
                results['MS-SSIM (%s)' % mode][n] = msssims

    tmp_dict = {}
    for key, val in results.items():  # convert MS-SSIM values to decibels for easier plotting
        if 'MS-SSIM' in key:
            tmp_dict['%s (dB)' % key] = convert_to_db(val)
    results.update(tmp_dict)

    return results


def evaluate_compression_jpg(img_file, quality=(1, 10), return_reconstructions=False, use_tf=False):
    """
    Compress an image with JPEG using varing quality settings for RD performance evalution.
    :param img_file:
    :param quality: list of integers; this is the quality parameter from 1 to 100 (higher the better quality)
    :param use_tf:
    :param float_type: floating type used by the arithmetic internally; uint8 gets converted to this type.
    :return:
    """
    from PIL import Image
    import img_comparison_metrics

    from io import BytesIO
    orig = Image.open(img_file)
    img = orig.convert('RGB')  # get rid of alpha dimension from .png; this is input to JPEG baselines

    M = len(quality)
    img_hats = []
    file_sizes = np.empty(M)
    for i, qual in enumerate(quality):
        img_file = BytesIO()
        img.save(img_file, 'jpeg', quality=qual)
        file_sizes[i] = img_file.tell()
        img_hat = Image.open(img_file).convert('RGB')
        img_hats.append(img_hat)
        img_file.close()

    num_pixels = orig.size[0] * orig.size[1]

    results = {}
    results['BPP'] = file_sizes * 8 / num_pixels

    # prepare for img quality comparisons
    x = np.asarray(img)  # RGB H x W x 3, uint8
    x_yc = np.asarray(img.convert('YCbCr'))  # YCbCr, H x W x 3, uint8

    x_hats = np.array([np.asarray(img_hat) for img_hat in img_hats])  # RGB, M x H x W x 3, uint8
    x_hats_yc = np.array([np.asarray(img_hat.convert('YCbCr')) for img_hat in img_hats])  # YCbCr, M x H x W x 3, uint8

    if return_reconstructions:
        results['reconstructions'] = x_hats

    comparison_modes = ('RGB', 'Luma', 'Chroma')
    for mode in comparison_modes:
        if mode == 'RGB':
            x_comp = x
            # xs_comp = np.repeat(x[None, ...], repeats=N, axis=0)  # M x H x W x 3
            # x_hats = np.array([np.asarray(img_hat) for img_hat in img_hats])
            x_hats_comp = x_hats
        elif mode == 'Luma':
            x_comp = x_yc[..., 0:1]
            # xs_comp = np.repeat(x_yc[None, :, :, 0:1], repeats=N, axis=0)  # M x H x W x 1
            x_hats_comp = x_hats_yc[..., 0:1]  # M x H x W x 1
        elif mode == 'Chroma':
            x_comp = x_yc[..., 1:]
            # xs_comp = np.repeat(x_yc[None, :, :, 1:], repeats=N, axis=0)  # M x H x W x 2
            x_hats_comp = x_hats_yc[..., 1:]  # M x H x W x 2
        else:
            raise NotImplementedError
        xs_comp = np.repeat(x_comp[None, ...], repeats=M, axis=0)  # same shape as x_hats_comp

        if not use_tf:
            # mses = img_comparison_metrics.mse(xs_comp, x_hats_comp)
            results['MSE (%s)' % mode] = img_comparison_metrics.mse(xs_comp, x_hats_comp)
            # psnrs = img_comparison_metrics.psnr(xs_comp, x_hats_comp, max_val=255)
            results['PSNR (%s)' % mode] = img_comparison_metrics.psnr(xs_comp, x_hats_comp, max_val=255)
            # msssims = img_comparison_metrics.ms_ssim(xs_comp, x_hats_comp, max_val=255)
            results['MS-SSIM (%s)' % mode] = img_comparison_metrics.ms_ssim(xs_comp, x_hats_comp, max_val=255)
        else:
            import tensorflow as tf
            xs_comp = tf.convert_to_tensor(xs_comp)
            x_hats_comp = tf.convert_to_tensor(x_hats_comp)

            mses = tf.reduce_mean(tf.squared_difference(tf.cast(xs_comp, 'float32'), tf.cast(x_hats_comp, 'float32')),
                                  axis=(1, 2, 3))
            psnrs = tf.image.psnr(x_hats_comp, xs_comp, 255)
            ssims = tf.image.ssim(x_hats_comp, xs_comp, 255)
            msssims = tf.image.ssim_multiscale(x_hats_comp, xs_comp,
                                               255)  # can crash: https://github.com/tensorflow/tensorflow/issues/33840
            # Also crashes with "Invalid argument: Computed output size would be negative" when input images are too small (this
            # is expected behavior: the default setting (https://www.tensorflow.org/api_docs/python/tf/image/ssim_multiscale)
            # uses 5 2x downsampling scales and a filter size of 11, so for a small image, say 64x64, 3 2x scaling reduce it to
            # 8x8, which is smaller than kernel size 11)
            tensors = [mses,
                       psnrs,
                       ssims,
                       msssims,
                       ]
            if not tf.executing_eagerly():
                # if sess is not None:
                with tf.Session() as sess:
                    # with sess as sess:
                    tensors = sess.run(tensors)
            [mses, psnrs, ssims, msssims] = tensors
            results['MSE (%s)' % mode] = mses
            results['PSNR (%s)' % mode] = psnrs
            results['SSIM (%s)' % mode] = ssims
            results['MS-SSIM (%s)' % mode] = msssims

    tmp_dict = {}
    for key, val in results.items():  # convert MS-SSIM values to decibels for easier plotting
        if 'MS-SSIM' in key:
            tmp_dict['%s (dB)' % key] = convert_to_db(val)
    results.update(tmp_dict)

    return results
