This is the implementation of Experiment 1 in the paper. 

### Files:
- **dist_comparison.py** is the main script.
- **traj_patterns.py** is the code for generating the synthetic dataset.
- **rae_dist.py** implements the proposed method.
- **hmm_dist.py** is the implementation for the HMM-based distance measure.

- **bases.pkl** and **trajs.pkl**. These are the data that can exactly reproduce the results. Since the data is generated randomly, we provide the original data for reproduction. You can uncomment the related code in **dist_comparison.py** to generate new data.
- **ratio_exp.npy** saves the final results of computed ratios, as shown in the paper. This is the generated experimental result.

Note that the computation procedure in **dist_comparison.py** is not optimized at all. It would be rather slow when computing distances between 100 trajectories, since we have to train the model for each trajectory and there are a lot of duplicate computation. If you want to have a quick run, please set the sample number as a small number such as 10.

### Dependencies:
- numpy
- scipy
- PyTorch (version >= 0.4.0) (for our method only)
- trajectory_distance (https://github.com/maikol-solis/trajectory_distance)
