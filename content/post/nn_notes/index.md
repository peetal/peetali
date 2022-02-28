---
title: Coursera Deep Learning Specialization Notes
summary: These notes include important concepts and model architectures on DNN, optimizers, hyperparameters tunning, CNN, YOLO algorithm for DeepCV, triplet loss and siamese network for one-shot learning, RNN with LSTM and GRU units, attention mechanism and Transformer architectures for NLP and CV. 

# Link this post with a project
projects: []

# Date published
date: "2022-01-05T00:00:00Z"

# Date updated
lastmod: "2022-01-05T00:00:00Z"

# Is this an unpublished draft?
draft: false

# Show this page in the Featured widget?
featured: false

# Featured image
# Place an image named `featured.jpg/png` in this page's folder and customize its options here.

reading_time: false
tags:

categories:
---
### Table of contents
1. [Intro to Neural Network and Deep Learning](#course1)
    1. [Logistic regression (forward & backward)](#log)
    2. [Neural network representation & activation functions](#nn)
    3. [Gradient descent, Back prop, & DNN](#dnn)
2. [Improving DNN: Hyperparameter tuning, Regularization, & Optimization](#course2)
    1. [General logic for improving DNN](#logic)
    2. [L2 regularization](#l2)
    3. [Dropout and Early-stop](#dropout)
    4. [Normalize inputs](#norminput)
    5. [Vanishing gradient & Initilization](#vanishGrad)
    6. [Gradient checking](#gradcheck)
    7. [Batch & mini-batch gradient descent](#batchgd)
    8. [Optimizers](#opt)
    9. [Learning rate decay](#alphadec)
    10. [Hyperparameter tunning process](#tunepro)
    11. [Batch Normalization](#batchNorm)
    12. [Softmax activation](#softmax)
3. [Structuring machine learning projects](#course3)
    1. [Speeding up the cycle: problem & solutions](#cycle)
    2. [Satificing & Optimizing metrics](#metric)
    3. [Train, Dev, & Test split](#split)
    4. [Error analysis](#error)
    5. [Training & Testing data from different distribution](#dist)
    6. [Transfer learning, Multitask learning, & End-to-end learning](#trick)
4. [Convolutional Neural Network](#course4)
    1. [Convolution as feature detector, padding, & stride](#conv)
    2. [Convolution over volume](#conv2)
    3. [Convolution layer](#convlayer)
    4. [Pooling layer](#pool)
    5. [Classic CNN architecture](#cnnarch)
    6. [Inception CNN and 1x1 convolution](#incep)
    7. [Mobile Net](#mob)
    8. [Sliding window for object detection](#window)
    9. [YOLO algorithm](#yolo)
    10. [Semantic segmentation](#semseg)
    11. [Siamese network & Triplet loss](#oneshot)
    12. [Neural style transfer](#nt)
5. [Recurrent Neural Network](#course5)
    1. [Use cases for RNN](#seq)
    2. [RNN forward & backward](#rnn)
    3. [Different types of RNN](#rnn2)
    4. [Language models & Sampling new text](#lang)
    5. [GRU & LSTM](#grulstm)
    6. [Deep RNN](#deep)
    7. [Word embeddings](#emb)
    8. [Learning embeddings: word2vec & Glove](#emb2)
    9. [Sentiment analysis](#sent)
    10. [Sequence to sequence models](#seq2seq)
    11. [Beam search](#beam)
    12. [Attention mechanism](#atten)
    13. [Transformer architecture](#trans)



## Intro to Neural Network and Deep Learning <a name="course1"></a>

### Logistic regression (forward & backward) <a name="log"></a>
<img src="dlnotes/course1/c1_logisticReg&NN1.png" alt="drawing" width="1000"/>
<img src="dlnotes/course1/c1_logisticReg&NN2.png" alt="drawing" width="1000"/>

### Neural network representation & activation functions  <a name="nn"></a>
<img src="dlnotes/course1/c1_logisticReg&NN3.png" alt="drawing" width="1000"/>
<img src="dlnotes/course1/c1_logisticReg&NN4.png" alt="drawing" width="1000"/>

### Gradient descent, Back prop, & DNN <a name="dnn"></a>
<img src="dlnotes/course1/c1_logisticReg&NN5.png" alt="drawing" width="1000"/>
<img src="dlnotes/course1/c1_logisticReg&NN6.png" alt="drawing" width="1000"/>
<img src="dlnotes/course1/c1_logisticReg&NN7.png" alt="drawing" width="1000"/>

## Improving DNN: Hyperparameter tuning, Regularization, & Optimization <a name="course2"></a>

### General logic for improving DNN <a name="logic"></a>
<img src="dlnotes/course2/c2_01_logic.png" alt="drawing" width="1000"/>

### L2 regularization <a name="l2"></a>
<img src="dlnotes/course2/c2_02_l2reg.png" alt="drawing" width="1000"/>

### Dropout and Early-stop <a name="dropout"></a>
<img src="dlnotes/course2/c2_03_dropout&earlystop.png" alt="drawing" width="1000"/>

### Normalize inputs <a name="norminput"></a>
<img src="dlnotes/course2/c2_04_normalizeInput.png" alt="drawing" width="1000"/>

### Vanishing gradient & Initilization <a name="vanishGrad"></a>
<img src="dlnotes/course2/c2_05_vanishingGrad.png" alt="drawing" width="1000"/>

### Gradient checking <a name="gradcheck"></a>
<img src="dlnotes/course2/c2_06_gradientCheck.png" alt="drawing" width="1000"/>

### Batch & mini-batch gradient descent <a name="batchgd"></a>
<img src="dlnotes/course2/c2_07_batchgd_1.png" alt="drawing" width="1000"/>
<img src="dlnotes/course2/c2_08_batchgd_2.png" alt="drawing" width="1000"/>

### Optimizers <a name="opt"></a>
<img src="dlnotes/course2/c2_09_optimizer_1.png" alt="drawing" width="1000"/>
<img src="dlnotes/course2/c2_10_optimizer_2.png" alt="drawing" width="1000"/>

### Learning rate decay <a name="alphadec"></a>
<img src="dlnotes/course2/c2_11_alphadecay.png" alt="drawing" width="1000"/>

### Hyperparameter tunning process <a name="tunepro"></a>
<img src="dlnotes/course2/c2_12_tunningprocess.png" alt="drawing" width="1000"/>

### Batch Normalization <a name="batchNorm"></a>
<img src="dlnotes/course2/c2_13_batchNorm1.png" alt="drawing" width="1000"/>
<img src="dlnotes/course2/c2_14_batchNorm2.png" alt="drawing" width="1000"/>

### Softmax activation <a name="softmax"></a>
<img src="dlnotes/course2/c2_15_softmax.png" alt="drawing" width="1000"/>

## Structuring machine learning projects <a name="course3"></a>

### Speeding up the cycle: problem & solutions <a name="cycle"></a>
<img src="dlnotes/course3/c3_01_structureProject1.png" alt="drawing" width="1000"/>
<img src="dlnotes/course3/c3_02_structureProject2.png" alt="drawing" width="1000"/>

### Satificing & Optimizing metrics <a name="metric"></a>
<img src="dlnotes/course3/c3_03_metric.png" alt="drawing" width="1000"/>

### Train, Dev, & Test split <a name="split"></a>
<img src="dlnotes/course3/c3_04_split.png" alt="drawing" width="1000"/>

### Error analysis <a name="error"></a>
<img src="dlnotes/course3/c3_05_error1.png" alt="drawing" width="1000"/>
<img src="dlnotes/course3/c3_06_error2.png" alt="drawing" width="1000"/>

### Training & Testing data from different distribution <a name="dist"></a>
<img src="dlnotes/course3/c3_07_dist.png" alt="drawing" width="1000"/>

### Transfer learning, Multitask learning, & End-to-end learning <a name="trick"></a>
<img src="dlnotes/course3/c3_08_trick.png" alt="drawing" width="1000"/>

## Convolutional Neural Network <a name="course4"></a>

### Convolution as feature detector, padding, & stride <a name="conv"></a>
<img src="dlnotes/course4/c4_01_convolution.png" alt="drawing" width="1000"/>

### Convolution over volume <a name="conv2"></a>
<img src="dlnotes/course4/c4_02_convolveVolume.png" alt="drawing" width="1000"/>

### Convolution layer <a name="convlayer"></a>
<img src="dlnotes/course4/c4_03_convLayer.png" alt="drawing" width="1000"/>

### Pooling layer <a name="pool"></a>
<img src="dlnotes/course4/c4_04_pooling.png" alt="drawing" width="1000"/>

### Classic CNN architecture <a name="cnnarch"></a>
<img src="dlnotes/course4/c4_05_cnnarch.png" alt="drawing" width="1000"/>

### Inception CNN and 1x1 convolution <a name="incep"></a>
<img src="dlnotes/course4/c4_06_incep.png" alt="drawing" width="1000"/>

### Mobile Net <a name="mob"></a>
<img src="dlnotes/course4/c4_07_mobilenet.png" alt="drawing" width="1000"/>

### Sliding window for object detection <a name="window"></a>
<img src="dlnotes/course4/c4_08_slidewindow.png" alt="drawing" width="1000"/>

### YOLO algorithm <a name="yolo"></a>
<img src="dlnotes/course4/c4_09_YOLO1.png" alt="drawing" width="1000"/>
<img src="dlnotes/course4/c4_10_YOLO2.png" alt="drawing" width="1000"/>

### Semantic segmentation <a name="semseg"></a>
<img src="dlnotes/course4/c4_11_semseg.png" alt="drawing" width="1000"/>

### Siamese network & Triplet loss <a name="oneshot"></a>
<img src="dlnotes/course4/c4_12_facereg1.png" alt="drawing" width="1000"/>
<img src="dlnotes/course4/c4_13_facereg2.png" alt="drawing" width="1000"/>

### Neural style transfer <a name="nt"></a>
<img src="dlnotes/course4/c4_14_nt1.png" alt="drawing" width="1000"/>
<img src="dlnotes/course4/c4_15_nt2.png" alt="drawing" width="1000"/>

## Recurrent Neural Network <a name="course5"></a>

### Use cases for RNN <a name="seq"></a>
<img src="dlnotes/course5/c5_01_sequence.png" alt="drawing" width="1000"/>

### RNN forward & backward <a name="rnn"></a>
<img src="dlnotes/course5/c5_02_rnn.png" alt="drawing" width="1000"/>

### Different types of RNN <a name="rnn2"></a>
<img src="dlnotes/course5/c5_03_rnnCat.png" alt="drawing" width="1000"/>

### Language models & Sampling new text <a name="lang"></a>
<img src="dlnotes/course5/c5_04_languageMod.png" alt="drawing" width="1000"/>

### GRU & LSTM <a name="gru&lstm"></a>
<img src="dlnotes/course5/c5_05_gru&lstm.png" alt="drawing" width="1000"/>

### Deep RNN <a name="deep"></a>
<img src="dlnotes/course5/c5_06_deeprnn.png" alt="drawing" width="1000"/>

### Word embeddings <a name="emb"></a>
<img src="dlnotes/course5/c5_07_wordembed.png" alt="drawing" width="1000"/>

### Learning embeddings: word2vec & Glove <a name="emb2"></a>
<img src="dlnotes/course5/c5_08_learnembed.png" alt="drawing" width="1000"/>

### Sentiment analysis <a name="sent"></a>
<img src="dlnotes/course5/c5_09_sentiment.png" alt="drawing" width="1000"/>

### Sequence to sequence models <a name="seq2seq"></a>
<img src="dlnotes/course5/c5_10_seq2seq.png" alt="drawing" width="1000"/>

### Beam search <a name="beam"></a>
<img src="dlnotes/course5/c5_11_beamsearch.png" alt="drawing" width="1000"/>

### Attention mechanism <a name="atten"></a>
<img src="dlnotes/course5/c5_12_attenMech.png" alt="drawing" width="1000"/>

### Transformer architecture <a name="trans"></a>
<img src="dlnotes/course5/c5_13_trans1.png" alt="drawing" width="1000"/>
<img src="dlnotes/course5/c5_14_trans2.png" alt="drawing" width="1000"/>
