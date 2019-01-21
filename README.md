# Zero-Shot-Action-Recognition-with-Two-Stream-GCN

This is a TensorFlow implementation of Two-Stream Graph Convolutional Networks for the task of action recognition, as described in our paper:

*Junyu Gao, Tianzhu Zhang, Changsheng Xu*, I Know the Relationships: Zero-Shot Action Recognition via Two-Stream Graph Convolutional Networks and Knowledge Graphs (**AAAI 2019**)

The code is developed based on the TensorFlow framework and the Graph Convolutional Network (GCN) repo [GCN](https://github.com/tkipf/gcn) and [zero-shot-gcn](https://github.com/JudyYe/zero-shot-gcn).

![](https://i.imgur.com/Z1SDVuU.png)


## Prerequisite

- Construct a python enviorment with python3.6 
- Install TensorFlow >= 1.2.0
	- Note the ZSAR performance slightly varies in different versions.
	- pip install tensorflow (For CPU)
	- pip install tensorflow-gpu (For GPU)
- Install networkx 
	- pip install networkx


## Dataset Preparation

The processed data of [UCF101](http://crcv.ucf.edu/data/UCF101.php) can be downloaded from [Google Driver](https://drive.google.com/open?id=1-ICJ-ruQzIIXx2Rh1GXEX7wkse809OM5).
After downloading the file, unzip it to the folder ./


## Run the demo

```
python train_two_stream_gcn.py
```

The above code will automatically train the two-stream GCN model on UCF101 dataset. The test accuracies will be outputted for each training epoch.

## Cite

Please cite our paper if you use this code in your own work:

```
@inproceedings{junyu2019AAAI_TS-GCN,
  title={I Know the Relationships: Zero-Shot Action Recognition via Two-Stream Graph Convolutional Networks and Knowledge Graphs},
  author={Gao, Junyu and Zhang, Tianzhu and Xu,  Changsheng},
  booktitle={AAAI},
  year={2019}
}
```
