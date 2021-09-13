---
title: Speeding Up linear SVM with precomputed kernel
subtitle: Precompute kernel matrices for model training and testing

# Summary for listings and search engines
summary: Precompute kernel matrices for model training and testing

# Link this post with a project
projects: []

# Date published
date: "2021-09-11T00:00:00Z"

# Date updated
lastmod: "2021-09-11T00:00:00Z"

# Is this an unpublished draft?
draft: false

# Show this page in the Featured widget?
featured: false

# Featured image
# Place an image named `featured.jpg/png` in this page's folder and customize its options here.
authors:
- admin

tags:

categories:
---

## **Main Take-aways**
Sklearn svm.SVC(kernel = "linear") has the notoriety of being slow. However, by using the "precomputed kernel" and compute the gram matrix separately, we could not only save time when training and testing data, but also save system memory when dealing with large matrices.  

Here I'm going to show a demo and some basic math to demonstrate this trick. 

## **1. Simulate data**
Here we have 40 samples and each sample has $10^7$ number of features. 
We also arbitrarily assign labels to all samples, with half being 0 and the resting being 1.  

We will be using the first 25 epcohs as training and the rest 15 as testing sets. 


```python
# simulate data
sim_data_features = [np.random.rand(10000000,) for i in range(40)]
sim_data_labels = np.tile([0,1],20)

# seperate training and testing
training_feature, testing_feature = sim_data_features[0:25], sim_data_features[25:40],
training_label, testing_label = sim_data_labels[0:25], sim_data_labels[25:40]
```

## **2. Train a SVC(kernel = "linear")**



```python
# train clf, time it
train_start = time.time()
clf = SVC(kernel='linear')
clf.fit(training_feature, training_label)
train_end = time.time()
print(f'svc linear kernel clf training took {train_end - train_start}')

# test clf, time it and print result 
test_start = time.time()
pred_lab = clf.predict(testing_feature)
test_end = time.time()
print(f'svc linear kernel clf testing took {test_end - test_start}')
print(f'Predicted labels are {pred_lab}')

print('---------------------------------------------')
print('printing the decision function outputs for all testing samples')
print(clf.decision_function(testing_feature))
```

    svc linear kernel clf training took: 16.423278093338013
    svc linear kernel clf testing took: 6.240693092346191
    Predicted labels are [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
    ---------------------------------------------
    printing the decision function outputs for all testing samples:
    [-0.03974077 -0.03935351 -0.03929058 -0.03904115 -0.0405302  -0.03887102
     -0.04086935 -0.04088076 -0.03897439 -0.03982852 -0.03934111 -0.04194098
     -0.04004195 -0.0407289  -0.04057421]


## **3. Train a SVC with precomputed gram matrix**


```python
# compute gram matrix for training 
gram_start = time.time()
gram_mat = np.matmul(np.array(training_feature), np.array(training_feature).T)
gram_end = time.time()
print('--------------------Training---------------------')
print(f'Computing gram matrix took {gram_end - gram_start}')

# train clf with precomputed gram matrix 
train_start = time.time()
clf = SVC(kernel='precomputed')
clf.fit(gram_mat, training_label)
train_end = time.time()
print(f'clf training took : {train_end - train_start}')
print(f'Total time for training: {train_end - gram_start}')
print('--------------------Testing---------------------')

# compute gram matrix for testing 
gram_start = time.time()
gram_test = np.matmul(np.array(testing_feature), np.array(training_feature).T)
gram_end = time.time()
print(f'Computing testing kernel took {gram_end - gram_start}')

# test clf with precomputed gram matrix 
test_start = time.time()
pred_lab = clf.predict(gram_test)
test_end = time.time()
print(f'svc precomputed kernel clf testing took {test_end - test_start}')
print(f'Total time for testing: {test_end - gram_start}')
print(f'Predicted labels are {pred_lab}')

print('---------------------------------------------')
print('printing the decision function outputs for all testing samples')
print(clf.decision_function(gram_test))
```

    --------------------Training---------------------
    Computing gram matrix took: 3.305522918701172
    clf training took: 0.18439912796020508
    Total time for training: 3.490262985229492
    --------------------Testing---------------------
    Computing testing kernel took: 2.019568920135498
    svc precomputed kernel clf testing took: 0.0002570152282714844
    Total time for testing: 2.0199968814849854
    Predicted labels are: [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
    ---------------------------------------------
    printing the decision function outputs for all testing samples:
    [-0.03974077 -0.03935351 -0.03929058 -0.03904115 -0.0405302  -0.03887102
     -0.04086935 -0.04088076 -0.03897439 -0.03982852 -0.03934111 -0.04194098
     -0.04004195 -0.0407289  -0.04057421]


## **4. Model with precomputed gram matrix is much more efficient**
#### 4.1 First let's look at the time it took for model training and testing. 

- It is clear to see that the two models were exactly the same, outputing the **same predictive labels**, with the **exact same decision function outputs** (with possible tidy difference). This suggests that the two classifiers share the **same weight matrix that defines the hyperplane**. Given that the two models are identical, the second approach is also much faster **(training time: 16.42s vs. 3.50s; testing time: 6.24s vs. 2.02s)**. 

#### 4.2 Second, let's look at the inputs to the classifiers:



```python
print('--------------------Training---------------------')
print(f'The shape of the each training sample is {training_feature[0].shape}')
print(f'The shape of the training gram matrix is {gram_mat.shape}')
print('--------------------Testing---------------------')
print(f'The shape of the each training sample is {testing_feature[0].shape}')
print(f'The shape of the training gram (kernel) matrix is {gram_test.shape}')
```

    --------------------Training---------------------
    The shape of the each training sample is (10000000,)
    The shape of the training gram matrix is (25, 25)
    --------------------Testing---------------------
    The shape of the each training sample is (10000000,)
    The shape of the training gram (kernel) matrix is (15, 25)


- As shown above, for the first model, during training, we need to input 25 $10^7$ vectors to the model. On the other hand, we only need to input a 25 x 25 matrix if we use a precomputed kernel. When it comes to testing, again, we do not need to deal with the crazy large vectors. Instead, only a 15 x 25 kernal matrix would do the work here. 
- As a result, not only does the precomputed kernel save us computation time, it should in principle be much more friendly with memory. 
- It is also worth noting that, it could be time/memory consuming when computing the gram (kernel) matrix, but such step could be further optimized. Thus, using svm.SVC(kernel = "precomputed") should have a better potential then svm.linearSVC. Details can be found on the sklearn documentation: https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html


## **5. Conceptually how gram matrix does the work**

### 5.1 Primary and dual ridge regression 
In ridge regression, we use regularized least square to find the weight matrix such that we want to minimize, with $\lambda$ penalizes the size of th weight matrix.
$$ \frac{1}{2}(Xw - y)^T(Xw - y) + \lambda||w||^2 $$
Taking the derivative of the loss with respect to the weight parameter gives us
$$ X^TXw + \lambda w - X^Ty = 0 $$
$$ (X^TX + \lambda I)w = X^Ty $$
Because $(X^TX + \lambda I)w$ is always invertable if $\lambda$ > 0, we have 
$$ w = (X^TX + \lambda I)^{-1}yX^T$$
Thus we show here that the weight matrix $w$ can be expressed as a linear combination of a vector $\alpha$ and $X$ 
$$ w = \sum_{i=1}^{n} \alpha_{i}x_{i} $$
$$ \alpha = (X^TX + \lambda I)^{-1}y = (G + \lambda I)^{-1}y $$

### 5.2 Think about the dual optimization problem in SVC
Indeed the equation is much more complex when it comes to SVC. However, from 5.1, intuitively we can see that the solution of $\alpha$ can be written as a function of the Gram matrix G, which is $X^TX$. I believe the time complexity of sklearn implementation of $svm.svc(kernel = "linear")$ would be $O(l^2n)$ with l being the number of sample and n being the number of features becuase svm.SVC allows more complicated kernels, while linear kernel could be an special case. With precomputed gram matrix, the time complexity now is bounded by O(l^3). Because $l \ll n$, the precomputed linear kernel is much faster. <br>

As my understanding in math grows (casue now obviously i suck), I will update this post with more concrete equations and interpretations. 


