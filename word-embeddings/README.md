# Compressing Word Embeddings

To reproduce the results in the paper, follow the steps below.

**Note:** If you only want to reproduce the plots with the exact same bitrates
and performance metrics (e.g., to use as a baseline), then you can skip steps 1
and 2 below and just run the cells in the jupyter notebook
`compress-trained-word-embeddings.ipynb` that load the plotted values from a
file and plot them.

1.  Download and preprocess the training data: follow the instructions in file
    `README.md` in the subdirectory `bayesian-skip-gram/preprocess-gbooks`.
2.  Train the Bayesian Skip-Gram model: This Requires Tensorflow version 1.
    More precisely, it was tested with tensorflow 1.15 on python 3.6.

    ```bash
    cd bayesian-skip-gram/models
    nohup python single-timestep.py \
        -E 100000 \
        --steps_per_checkpoint 1000 \
        -d 100 \
        -B 10000 \
        --lr0 0.1 \
        /path/to/npos_years1980-2008.bin.gz \
        /path/to/output/directory &
    ```

3.  Calculate compressed file sizes using the jupyter notebook `compress-trained-word-embeddings.ipynb`
