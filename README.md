# Paper of the source codes released:

Jing Ma, Wei Gao, and Kam-Fai Wong. Detect Rumors on Twitter by Promoting Information Campaigns with Generative Adversarial Learning. In The World Wide Web Conference (WWW '19).

# Datasets:

The datasets used in the experiments were based on the two publicly available Twitter datasets released by Ma et al. (2016) and Zubiaga et al. (2017):

Jing Ma, Wei Gao, Prasenjit Mitra, Sejeong Kwon, Bernard J Jansen, Kam-Fai Wong, and Meeyoung Cha. Detecting rumors from microblogs with recurrent neural networks. In Proceedings of IJCAI 2016.

Arkaitz Zubiaga, Maria Liakata, and Rob Procter. 2017. Exploiting context for rumour detection in social media. In International Conference on Social Informatics. Springer, 109–123.

In the 'resource' folder we provide the pre-processed data files used for our experiments. The raw datasets can be respectively downloaded from https://www.dropbox.com/s/46r50ctrfa0ur1o/rumdect.zip?dl=0. and https://figshare.com/articles/PHEME_dataset_of_rumours_and_non-rumours/4010619.

The datafile is in a tab-sepreted column format, where each row corresponds to a tweet. Consecutive columns correspond to the following pieces of information:

1: claim-id -- an unique identifier describing the claim;

2: index of time interval -- an index number that the current batch of tweets is belong to;

3: list-of-index-and-counts -- the rest of the line contains space separated index-count pairs, where a index-count pair is in format of "index:count", E.g., "index1:count1 index2:count2" (extracted from the "text" field in the json format from Twitter)


# Dependencies:
Please install the following python libraries:

numpy version 1.11.2

theano version 0.8.2

# Reproduce the experimental results
Run script "model/ Main_GAN_RNN_pheme.py" for GAN-RNN model on PHEME dataset or "model/ Main_GAN_RNN_twitter.py" on TWITTER dataset.

Alternatively, you can change the "fold" parameter to set each fold.

We also save our trained model at "param/param-GAN-RNN-*.npz", you can alternatively choose to load it or not.

#If you find this code useful, please let us know and cite our paper.

