This is the implementation of Experiment 2 in the paper. 

### Files
- **datasets.py** is the main script.
- **ptaet.py** implements the proposed method.
- **hmm_dist.py** is the implementation for the HMM-based distance measure.
- **cross_rocs.py**, **detrac_aucs.py**, **traffic_rocs.py**, and **vmt_rocs.py** are code files for result reporting.
- Other .py files are named with "dataset_xxx_method", where datasets are in \['cross', 'detrac', 'traffic' 'vmt'\] and methods are 'eucl' for Euclidean distance, 'hmm' for HMM-based distance, 'aet' for our method, and 'others' for other trajectory distances.

### Folders
- **data** contains the intermediate files generated during code running.
- **datasets** contains the datasets (or downloading addresses) and necessary files for reproduction.

### Dependencies
- numpy
- scipy
- matplotlib
- sklearn
- PyTorch (version >= 0.4.0) (for our method only)
- trajectory_distance (https://github.com/maikol-solis/trajectory_distance)
