## Training
bls2017_vae.py trains a VAE using the same neural net architecture as in [Balle et. al., 2017](https://www.cns.nyu.edu/~lcv/iclr2017/). The code is adapted from https://github.com/tensorflow/compression/blob/a543a0a6ed205ddb6c35331510dcf2b2551143f4/examples/bls2017.py

To train the VAE in the color image compression experiment, run the following command:

	python bls2017_vae.py --verbose --seed 0 --checkpoint_dir /tmp/checkpoints --train_glob '/train_data_dir/*.png' --batchsize 64 --patchsize 256 --last_step 2000000 --preprocess_threads 4 --log_dir /tmp/tf_logs --save_checkpoint_secs 10800 --likelihood_variance 0.001 --num_filters 256 256 256 --filter_dims 9 5 5 --sampling_rates 4 2 2

where /tmp/checkpoints is the directory for saving model checkpoints, /tmp/tf_logs is the directory for saving tensorboard logging information (set this to the empty string '' to disable tensorboard logging), and '/train_data_dir/*.png' is the glob pattern that specifies training images.

The trained model is then stored in a new directory within /tmp/checkpoints called 'bls2017_vae-num_filters=256_256_256-filter_dims=9_5_5-sampling_rates=4_2_2-learned_prior=False-likelihood_variance=0.001-last_step=2000000'.
In practice, training for 2000000 steps is probably unnecessary as the model performance generally plateaus after about 1000000 steps.


## Evaluation
Run the following to estimate the empirical distributions of the posterior means (as mentioned in the main text, we obtain better performance (lower rate) with an "empirical prior" fitted on the posterior means) and evaluate various methods on test images:

	python post_process.py /tmp/checkpoints/ bls2017_vae-num_filters=256_256_256-filter_dims=9_5_5-sampling_rates=4_2_2-learned_prior=False-likelihood_variance=0.001-last_step=2000000 /path_to/val_imgs.npy '/test_data/*.png'

where /path_to/val_imgs.npy stores a numpy array of validation images (this can be a subset of training images) with shape N x H x W x 3 (this data is used to build entropy models of code distribution in the compression methods; we used about 500 training images in our experiments), and '/test_data/*.png' is the glob pattern for test (Kodak) images. We obtained the Kodak images from Balle et. al.'s website, https://www.cns.nyu.edu/~lcv/iclr2017/; we also scraped the site for their rate-distortion results on Kodak (stored in Balle2017_proposed_compression_stats_on_Kodak.npz).

The evaluation command creates the following numpy arrays in the model checkpoint directory:

ChannelwisePriorCDFQuantizer-max_bits=10-compression_results.npz  
KmeansQuantizer-compression_results.npz  
UniformQuantizer-compression_results.npz  

which correspond to rate-distortion results for our method, kmeans baseline, and uniform quantization baseline, respectively.

The rate-distortion performance for JPEG can be obtained by calling utils.evaluate_compression_jpg(test_img_file, quality) for each test image and quality setting. The results on Kodak are included for convenience in JPEG_compression_stats_on_Kodak.npz.


## Dependencies
The dependencies are listed in requiements.txt, and should be installed via pip (instead of conda) to ensure tensorflow is compatible with the tensorflow-compression module.


