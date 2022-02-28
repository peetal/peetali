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
3. 

## Intro to Neural Network and Deep Learning <a name="course1"></a>

### Logistic regression (forward & backward) <a name="log"></a>
<img src="dlnotes/course1/c1_logisticReg&NN1.png" alt="drawing" width="700"/>
<img src="dlnotes/course1/c1_logisticReg&NN2.png" alt="drawing" width="700"/>

### Neural network representation & activation functions  <a name="nn"></a>
<img src="dlnotes/course1/c1_logisticReg&NN3.png" alt="drawing" width="700"/>
<img src="dlnotes/course1/c1_logisticReg&NN4.png" alt="drawing" width="700"/>

### Gradient descent, Back prop, & DNN <a name="dnn"></a>
<img src="dlnotes/course1/c1_logisticReg&NN5.png" alt="drawing" width="700"/>
<img src="dlnotes/course1/c1_logisticReg&NN6.png" alt="drawing" width="700"/>
<img src="dlnotes/course1/c1_logisticReg&NN7.png" alt="drawing" width="700"/>

## Improving DNN: Hyperparameter tuning, Regularization, & Optimization <a name="course2"></a>

### General logic for improving DNN <a name="logic"></a>
<img src="dlnotes/course2/c2_01_logic.png" alt="drawing" width="700"/>

### L2 regularization <a name="l2"></a>
<img src="dlnotes/course2/c2_02_l2reg.png" alt="drawing" width="700"/>

### Dropout and Early-stop <a name="dropout"></a>
<img src="dlnotes/course2/c2_03_dropout&earlystop.png" alt="drawing" width="700"/>

### Normalize inputs <a name="norminput"></a>
<img src="dlnotes/course2/c2_04_normalizeInput.png" alt="drawing" width="700"/>

### Vanishing gradient & Initilization <a name="vanishGrad"></a>
<img src="dlnotes/course2/c2_05_vanishingGrad.png" alt="drawing" width="700"/>

### Gradient checking <a name="gradcheck"></a>
<img src="dlnotes/course2/c2_06_gradientCheck.png" alt="drawing" width="700"/>

### Batch & mini-batch gradient descent <a name="batchgd"></a>
<img src="dlnotes/course2/c2_07_batchgd_1.png" alt="drawing" width="700"/>
<img src="dlnotes/course2/c2_08_batchgd_2.png" alt="drawing" width="700"/>

### Optimizers <a name="opt"></a>
<img src="dlnotes/course2/c2_09_optimizer_1.png" alt="drawing" width="700"/>
<img src="dlnotes/course2/c2_10_optimizer_2.png" alt="drawing" width="700"/>

### Learning rate decay <a name="alphadec"></a>
<img src="dlnotes/course2/c2_11_alphadecay.png" alt="drawing" width="700"/>

### Hyperparameter tunning process <a name="tunepro"></a>
<img src="dlnotes/course2/c2_12_tunningprocess.png" alt="drawing" width="700"/>

### Batch Normalization <a name="batchNorm"></a>
<img src="dlnotes/course2/c2_13_batchNorm1.png" alt="drawing" width="700"/>
<img src="dlnotes/course2/c2_14_batchNorm2.png" alt="drawing" width="700"/>

### Softmax activation <a name="softmax"></a>
<img src="dlnotes/course2/c2_15_softmax.png" alt="drawing" width="700"/>
