from __future__ import absolute_import, division, print_function, unicode_literals
# import numpy as np
# import tensorflow as tf

# tf.enable_eager_execution()
# assert tf.executing_eagerly(), 'should use eager execution'
import os

import numpy as np
import tensorflow.compat.v1 as tf

import tensorflow_compression as tfc
import vae_models
from vae_models import GaussianVAE, StandardGaussianPrior, BMSHJ2018Prior

from utils import tf_read_img, tf_convert_float_to_uint8, read_npy_file_helper

read_img = lambda img: tf_read_img(tf, img)
convert_float_to_uint8 = lambda img: tf_convert_float_to_uint8(tf, img)


def train(args):
    """Trains the model."""
    import glob

    if args.verbose:
        tf.logging.set_verbosity(tf.logging.INFO)
    else:
        tf.logging.set_verbosity(tf.logging.WARN)

    # Create input data pipeline.
    with tf.device("/cpu:0"):
        train_files = glob.glob(args.train_glob)
        if not train_files:
            raise RuntimeError(
                "No training images found with glob '{}'.".format(args.train_glob))
        train_dataset = tf.data.Dataset.from_tensor_slices(train_files)
        train_dataset = train_dataset.shuffle(buffer_size=len(train_files)).repeat()
        if 'npy' in args.train_glob:  # reading numpy arrays directly instead of from images
            train_dataset = train_dataset.map(  # https://stackoverflow.com/a/49459838
                lambda item: tuple(tf.numpy_function(read_npy_file_helper, [item], [tf.float32, ])),
                num_parallel_calls=args.preprocess_threads)
        else:
            train_dataset = train_dataset.map(
                read_img, num_parallel_calls=args.preprocess_threads)
        train_dataset = train_dataset.map(lambda x: tf.random_crop(x, (args.patchsize, args.patchsize, 3)))
        train_dataset = train_dataset.batch(args.batchsize)
        train_dataset = train_dataset.prefetch(32)

    # Get training patch from dataset.
    x = train_dataset.make_one_shot_iterator().get_next()

    # Instantiate model.
    model = create_model(args)
    loss_dict = model.compute_loss(x, args.likelihood_variance)
    train_loss = loss_dict['loss']
    x_hat = loss_dict['x_hat']
    train_mse = tf.reduce_mean(tf.squared_difference(x, x_hat))

    step = tf.train.create_global_step()
    nn_variables = model.inference_net.variables + model.generative_net.variables
    main_step = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(train_loss, global_step=step, var_list=nn_variables)
    if args.learned_prior:
        if not args.prior_lr:
            args.prior_lr = 1e-3
        prior_step = tf.train.AdamOptimizer(learning_rate=args.prior_lr).minimize(train_loss,
                                                                                  var_list=model.prior.variables)
        train_op = tf.group(prior_step, main_step)
    else:
        train_op = main_step

    # aux_optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)
    # aux_step = aux_optimizer.minimize(entropy_bottleneck.losses[0])
    # train_op = tf.group(main_step, aux_step, entropy_bottleneck.updates[0])


    hooks = [
        tf.train.StopAtStepHook(last_step=args.last_step),
        tf.train.NanTensorHook(train_loss),
    ]

    runname = get_runname(vars(args))
    save_dir = os.path.join(args.checkpoint_dir, runname)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    import json
    import datetime
    with open(os.path.join(save_dir, 'record.txt'), 'a') as f:  # keep more detailed record in text file
        f.write(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + '\n')
        f.write(json.dumps(vars(args), indent=4, sort_keys=True) + '\n')
        f.write('\n')
    with open(os.path.join(save_dir, 'args.json'), 'w') as f:  # will overwrite existing
        json.dump(vars(args), f, indent=4, sort_keys=True)

    if args.log_dir != '':
        tf.summary.scalar("loss", train_loss)
        tf.summary.scalar("elbo", loss_dict['elbo'])
        tf.summary.scalar("likelihood", loss_dict['likelihood'])
        tf.summary.scalar("kl", loss_dict['kl'])
        tf.summary.scalar("mse", train_mse)
        tf.summary.scalar("psnr", - 10 * (tf.log(train_mse) / np.log(10)))  # note MSE was computed on float images

        tf.summary.image("original", convert_float_to_uint8(x), max_outputs=2)
        tf.summary.image("reconstruction", convert_float_to_uint8(x_hat), max_outputs=2)
        summary_op = tf.summary.merge_all()
        tf_log_dir = os.path.join(args.log_dir, runname)
        summary_hook = tf.train.SummarySaverHook(save_secs=args.save_summary_secs, output_dir=tf_log_dir,
                                                 summary_op=summary_op)
        hooks.append(summary_hook)

    with tf.train.MonitoredTrainingSession(
            hooks=hooks, checkpoint_dir=save_dir,
            save_checkpoint_secs=args.save_checkpoint_secs, save_summaries_secs=args.save_summary_secs) as sess:
        while not sess.should_stop():
            sess.run(train_op)

    return model


def create_model(args):
    """
    Instantiate a model object given options.
    Creates a three-layer conv net (with custom GDN/IGDN activations) from Balle 2017.
    :param args.num_filters: iterable of ints indicating num filters per layer (original model used the same for all layers)
    :param args.filter_dims: length-3 tuple of kernel widths in the forward (analysis) computation; the reverse is
    used for backward (synthesis) computation
    :param args.sampling_rates: length-3 tuple of downsampling rates in the forward (analysis) computation; the reverse is
    used for backward (synthesis) computation
    :return:
    """
    if args.learned_prior:
        # prior uses the same number of distinct distributions as number of filters/channels in the last layer of
        # inference network
        prior = BMSHJ2018Prior(args.num_filters[-1], init_scale=args.prior_init_scale)
    else:
        prior = StandardGaussianPrior()
    inference_net, generative_net = vae_models.get_BLS2017_neural_nets(args.num_filters, args.filter_dims,
                                                                       args.sampling_rates)
    model = GaussianVAE(prior, inference_net, generative_net, decode_sigmoid=False)
    return model


def load_model(args):
    """
    Create model and ops necessary for compression, restore weights (from latest checkpoint); assuming tf non-eager
    :param args:
    :return:
    """
    x_input_op = tf.placeholder(tf.float32, [None, None, None, 3])  # always assume colored images input
    model = create_model(args)
    # model.build([None, None, None, 3])
    encode_op = model.encode(x_input_op)
    z_num_filters = args.num_filters[-1]
    z_input_op = tf.placeholder(tf.float32, [None, None, None, z_num_filters])
    decode_op = model.decode(z_input_op)

    # Open tf.session and restorem model
    latest_ckpt = tf.train.latest_checkpoint(checkpoint_dir=args.checkpoint_dir)
    # print(latest_ckpt)

    sess = tf.Session()
    tf.train.Saver(var_list=model.variables).restore(sess, save_path=latest_ckpt)

    return model, x_input_op, z_input_op, encode_op, decode_op


def get_runname(args_dict):
    """
    Given a dictionary of cmdline arguments, return a string that identifies the training run.
    :param args_dict:
    :return:
    """
    config_strs = []  # ['key1=val1', 'key2=val2', ...]

    for key, val in args_dict.items():
        if isinstance(val, (list, tuple)):
            val_str = '_'.join(map(str, val))
            config_strs.append('%s=%s' % (key, val_str))

    for key in ('learned_prior', 'likelihood_variance', 'last_step'):
        config_strs.append('%s=%s' % (key, args_dict[key]))
        if key == 'learned_prior' and args_dict[key]:
            for sub_key in ('prior_init_scale', 'prior_lr'):
                config_strs.append('%s=%s' % (sub_key, args_dict[sub_key]))

    script_name = os.path.splitext(os.path.basename(__file__))[0]  # current script name, without extension
    return '-'.join([script_name] + config_strs)


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()

    # High-level options.
    parser.add_argument(
        "--verbose", "-V", action="store_true",
        help="If set, use tf.logging.INFO; otherwise use tf.logging.WARN, for tf.logging")

    parser.add_argument(
        "--checkpoint_dir", default="checkpoints",
        help="Directory where to save/load model checkpoints.")

    parser.add_argument(
        "--seed", type=int, default=0,
        help="Seed for reproducibility")

    # Model architecture
    parser.add_argument("--num_filters", nargs='*', type=int, default=[32, 32, 32],
                        help="Number of filters per layer; should be 3 integers")
    parser.add_argument('--filter_dims', nargs='*', type=int, default=[9, 5, 5],
                        help='dimensions of filters in the encoding/analysis network; should be 3 integers')
    parser.add_argument('--sampling_rates', nargs='*', type=int, default=[4, 2, 2],
                        help='down sampling rates in the encoding/analysis network; should be 3 integers')
    parser.add_argument('--learned_prior', action="store_true", help='If set, will use learnable prior from BMSHJ2018')
    parser.add_argument('--prior_init_scale', type=float, default=10., help='scale for initializing learned prior')
    parser.add_argument('--prior_lr', type=float, default=1e-3, help='Learning rate for learned prior')

    # For training
    parser.add_argument(
        "--train_glob", default="images/*.png",
        help="Glob pattern identifying training data. This pattern must expand "
             "to a list of RGB images in PNG or JPEG format.")
    parser.add_argument(
        "--batchsize", type=int, default=100,
        help="Batch size for training.")
    parser.add_argument(
        "--patchsize", type=int, default=32,
        help="Size of image patches for training.")
    parser.add_argument(
        "--likelihood_variance", type=float, default=0.02, dest="likelihood_variance",
        help="Variance of the likelihood disribution p(x|z); similar to beta, lower => lower distortion & higher rate.")
    parser.add_argument(
        "--last_step", type=int, default=100000,
        help="Train up to this number of steps.")
    parser.add_argument(
        "--preprocess_threads", type=int, default=8,
        help="Number of CPU threads to use for parallel decoding of training "
             "images.")
    parser.add_argument(
        "--save_checkpoint_secs", type=int, default=300,
        help="Seconds elapsed b/w saving models."
    )
    parser.add_argument(
        "--save_summary_secs", type=int, default=60,
        help="Seconds elapsed b/w saving tf logging summaries."
    )
    parser.add_argument(
        "--log_dir", default="/tmp/tensorboard_logs",
        help="Directory for storing Tensorboard logging files; set to empty string '' to disable Tensorboard logging.")

    args = parser.parse_args()

    tf.set_random_seed(args.seed)
    np.random.seed(args.seed)

    train(args)
