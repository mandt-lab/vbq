#! /usr/bin/env python

from __future__ import absolute_import, division, print_function, unicode_literals
import sys
import os
from quantizer import ChannelwisePriorCDFQuantizer
import vae_models
import utils
import bls2017_vae
import pickle

import numpy as np
import tensorflow as tf
import os
import time
import pprint
import json
from collections import namedtuple

tf.enable_eager_execution()
assert tf.executing_eagerly(), 'should use eager execution'
os.environ['CUDA_VISIBLE_DEVICES'] = -1  # use CPU for simplicity; o/w may get OOM on GPU when batch size is too large

checkpoint_dir = sys.argv[1]
runname = sys.argv[2]
val_img_path = sys.argv[3]
test_img_files_glob = sys.argv[4]

save_dir = os.path.join(checkpoint_dir, runname)
save_path = os.path.join(save_dir, 'latent_means.npy')


def load_model_from_run(runname, checkpoint_dir, create_model, runname_glob=True, reset_default_graph=True):
    # will use latest checkpoint by default
    from glob import glob
    save_dir = os.path.join(checkpoint_dir, runname)
    if runname_glob:
        save_dir = glob(save_dir)
        if not len(save_dir) == 1:
            modified_time = [os.path.getmtime(d) for d in save_dir]
            latest_idx = np.argmax(modified_time)
            save_dir = save_dir[latest_idx]
            print('      Using latest:', save_dir)
        else:
            save_dir = save_dir[0]
    ckpt = tf.train.latest_checkpoint(checkpoint_dir=save_dir)
    print('    loaded', ckpt)
    with open(os.path.join(save_dir, 'args.json')) as f:
        # based on https://stackoverflow.com/questions/6578986/how-to-convert-json-data-into-a-python-object
        args = json.load(f, object_hook=lambda d: namedtuple('Args', d.keys())(*d.values()))
    if reset_default_graph:
        tf.reset_default_graph()
    model = create_model(args)
    model.load_weights(ckpt)
    return model, args


create_model = bls2017_vae.create_model
model, args = load_model_from_run(runname, checkpoint_dir, create_model)
vae = model
num_latent_channels = args.num_filters[-1]

val_images = np.load(val_img_path)

train_empirical_prior = True
# train_empirical_prior = False
if train_empirical_prior:
    means, logvars = model.encode(val_images)
    flat_means = np.reshape(means, [-1, means.shape[-1]])
    np.save(save_path, flat_means)
    print('    saved latent samples at', save_path)

    command = f"python learned_prior.py --num_channels {num_latent_channels} --dims 3 3 3 --init_scale 1 --lr 0.1 \
    --its 400 --checkpoint_dir {save_dir} --tol 1e-2 --logging_freq 50 \
    --data_path {save_path}"

    # launch subprocess to train prior model (GPU)
    import subprocess, os

    subprocess.Popen(command,
                     shell=True).wait()  # blocking; https://stackoverflow.com/questions/22698754/subprocess-calls-are-they-done-in-parallel
    # subprocess.Popen("python -c \"import os; print(os.environ['CUDA_VISIBLE_DEVICES'])\"", shell=True, env=sub_env)
    print('    trained empirical prior')

# build CDF quantizer model (may get OOM error if run on GPU)
# Build quantizers with learned priors
import learned_prior


def build_cdf_qzer(runname, checkpoint_dir, max_bits_per_coord, lambs, rebuild=True):
    # for runname in all_runnames:
    save_dir = os.path.join(checkpoint_dir, runname)
    quantizer_path = os.path.join(save_dir, 'ChannelwisePriorCDFQuantizer-max_bits=%d.pkl' % max_bits_per_coord)

    if rebuild or (not os.path.exists(quantizer_path)):
        vae = model  # already loaded outside function
        num_latent_channels = args.num_filters[-1]

        quantizer = ChannelwisePriorCDFQuantizer(num_latent_channels, max_bits_per_coord)
        prior, pargs = load_model_from_run('learned_prior*', save_dir, learned_prior.create_model, \
                                           runname_glob=True, reset_default_graph=False)

        quantizer.build_code_points(prior)
        quantizer.build_entropy_models(val_images, vae, lambs, add_n_smoothing=1)

        with open(quantizer_path, 'wb') as f:
            pickle.dump(quantizer, f)
        print('    built quantizer')
    else:
        print('    skipping', runname)


# lambs = [100, 30, 10, 3, 1, 0.3, 0.1, 0.03, 0.01]
# lambs = [0.03, 0.1, 30]
lambs = 2 ** np.linspace(-8, 7, 16)
print([float(l) for l in lambs])
max_bits_per_coord = 10
rebuild = True
# rebuild = False

build_cdf_qzer(runname, checkpoint_dir, max_bits_per_coord, lambs, rebuild=rebuild)
print('    built cdf quantizer')


# build uniform, kmeans, quantizer model
def build_simple_qzer(runname, checkpoint_dir, quantizer_type, rebuild=True):
    save_dir = os.path.join(checkpoint_dir, runname)
    quantizer_path = os.path.join(save_dir, '%s.pkl' % quantizer_type.__name__)

    if rebuild or (not os.path.exists(quantizer_path)):

        num_latent_channels = args.num_filters[-1]
        quantizer = ChannelwiseSimpleQuantizerWrapper(quantizer_type, num_latent_channels, quantization_levels)
        #         prior, pargs = load_model_from_run('learned_prior*', save_dir, learned_prior.create_model, \
        #                                            runname_glob=True, reset_default_graph=False)

        #         quantizer.build_code_points(prior)
        quantizer.fit(val_images, vae, add_n_smoothing=1)

        with open(quantizer_path, 'wb') as f:
            pickle.dump(quantizer, f)
        print('built quantizer at %s' % quantizer_path)
    else:
        print('skipping', runname)


quantization_levels = [2, 4, 6, 8, 16, 32, 64, 128, 256]
from quantizer import UniformQuantizer, KmeansQuantizer, ChannelwiseSimpleQuantizerWrapper

for quantizer_type in (UniformQuantizer, KmeansQuantizer):
    print('building ', quantizer_type)
    build_simple_qzer(runname, checkpoint_dir, quantizer_type, rebuild=rebuild)

print('    built simple quantizers')


# evaluate on Kodak

def eval_compression_qzer(quantizer_type_name, runname, checkpoint_dir, test_img_files, settings, save=True,
                          reconstructions=False, **kwargs):
    save_dir = os.path.join(checkpoint_dir, runname)
    quantizer_path = os.path.join(save_dir, f'{quantizer_type_name}.pkl')
    with open(quantizer_path, 'rb') as f:
        quantizer = pickle.load(f)

    results = utils.evaluate_compression_quantizer(quantizer, model, test_img_files, \
                                                   settings, model_input_float_type='float32',
                                                   return_reconstructions=reconstructions, **kwargs)
    if not save:
        return results
    else:
        if reconstructions:  # can create huge files
            # hack to deal with np.savez (ultimately np.array) can't handle jagged array like results['reconstructions']
            # save as dict instead
            tmp_dict = {}
            for test_img_file, rec in zip(test_img_files, results['reconstructions']):
                tmp_dict[test_img_file] = rec
            results['reconstructions'] = tmp_dict

        np.savez(os.path.join(save_dir, f'{quantizer_type_name}-compression_results.npz'),
                 settings=settings,
                 #             max_bits_per_coord=max_bits_per_coord,
                 #              entropy_models=entropy_models,
                 compressed_files=test_img_files,
                 quantizer_path=quantizer_path,
                 **results
                 )
    return results


import glob

test_img_files = glob.glob(test_img_files_glob)
quantizer_type_names = [
    'ChannelwisePriorCDFQuantizer-max_bits=%d' % max_bits_per_coord,
    UniformQuantizer.__name__,
    KmeansQuantizer.__name__
]

all_settings = [lambs, quantization_levels, quantization_levels]
# for i in reversed(range(len(quantizer_type_names))):
for i in range(len(quantizer_type_names)):
    quantizer_type_name = quantizer_type_names[i]
    settings = all_settings[i]
    # for quantizer_type_name, settings in zip(quantizer_type_names, all_settings):
    print('evaluating', quantizer_type_name)
    eval_compression_qzer(quantizer_type_name, runname, checkpoint_dir, \
                          test_img_files, settings, save=True, reconstructions=False, \
                          use_tf=True)

    print('evaled', quantizer_type_name)

print('    evaled on kodak')
