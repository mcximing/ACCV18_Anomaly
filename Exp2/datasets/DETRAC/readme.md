The DETRAC dataset labeled for trajectory anomalies.

This is a video dataset originally introduced as a large-scale real-world benchmark for object detection and multi-object tracking [1]. 

We first generate trajectories based on the tracking ground-truth of the training set. Then, We divide the behaviors of vehicle trajectories to several types including Driving through, Changing lane, Turn, Wait and some ambiguities. In order to define a proper amount of anomalous samples, we label the trajectories with Turn and Wait behaviors as anomalies. Finally, there are 31 scenes labeled with anomalies out of the 60 training videos. The number of trajectory samples in each scene ranges from 8 to 89 and there are 3 anomalies out of 43 trajectories in each scene on average.

The data is in Python 3 .pkl format and can be loaded using pickle as follows:

```
dataset_path = '/path_to_dataset/'
with open(dataset_path+'data.pkl', 'rb') as fp:
    data = pickle.load(fp)
with open(dataset_path+'label.pkl', 'rb') as fp:
    label = pickle.load(fp)
```

Reference:

[1] UA-DETRAC: a new benchmark and protocol for multi-object detection and tracking.
Longyin Wen and Dawei Du and Zhaowei Cai and Zhen Lei and Ming-Ching Chang and Honggang Qi and Jongwoo Lim and Ming-Hsuan Yang and Siwei Lyu
arXiv:1511.04136 (2015)
url: http://detrac-db.rit.albany.edu/

