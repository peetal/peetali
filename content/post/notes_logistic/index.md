---
title: Binary and Multinomial Logistic Regression as GLMs and Neural Networks
subtitle: Understand Logistic and softmax regression as generalized linear model and simple NN

# Summary for listings and search engines
summary: Understand Logistic and softmax regression as generalized linear model and simple NN

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



Take-aways: logistic regression is the special (binary) case of the softmax regression, whereas the logistic loss is the special (binary) case of the cross-entropy loss. Logistic regression can be viewed as a single neuron, with the sigmoid activation function whereas the softmax regression can be viewed as a single layer of neuron, with the softmax activation function.

### 1. Logistic regression as generalized linear model

We are familiar with logistic regression as a member of the exponential family where all PDFs take the form

$$p_Y(y;\eta) = \frac{b(y)e^{\eta^TT(y)}}{e^{a(\eta)}}$$

The PMF of the Bernoulli random variable, parameterized by the success probability $\phi$ is 

$$p_Y(y;\phi)=\phi^y(1-\phi)^{1-y}=e^{log(\frac{\phi}{1-\phi})y+log(1-\phi)}$$

Matching the two PDFs, we have 

$$\eta=log(\frac{\phi}{1-\phi})\Rightarrow \phi=\frac{1}{1+e^{-\eta}}$$

The point here is to identify the function $g$ that maps the natural parameter $\eta$, (i.e., $x^T\theta$), to the canonical parameter $\phi$, which parameterize the PMF of the Bernoulli random variable above. 

![Screen Shot 2021-09-14 at 9.37.16 PM.png](Notes%20Binary%20and%20Multinomial%20Logistic%20Regression%20a%20f6e3c3c7d67247fd9e5591c928480b35/Screen_Shot_2021-09-14_at_9.37.16_PM.png)

Borrowed from CSE229; Thanks to 吴恩达

Note that $g$ here is the sigmoid function, thus explaining why we are using a sigmoid function in logistic regression and why the coefficients we get are in terms of log odds. 

$$\frac{p(X)}{1-p(X)}=e^{X^T\theta}\Rightarrow log(\frac{p(X)}{1-p(X)})={X^T\theta}$$

### 2. Logistic regression as neural network

- Logistic regression can also be seen as a neural network with only one neuron. What makes it a NN is that the logistic regression consists of two components: a linear component and a non-linear component. Here the linear component is the affine function $X^Tw+b$ whereas the nonlinear component is a sigmoid activation function $\frac{1}{1+e^{-z}}$, where z in this case would be the affine function output.

![Screen Shot 2021-09-14 at 8.16.45 PM.png](Notes%20Binary%20and%20Multinomial%20Logistic%20Regression%20a%20f6e3c3c7d67247fd9e5591c928480b35/Screen_Shot_2021-09-14_at_8.16.45_PM.png)

logistic regression as NN

### 3. Loss function for logistic regression

We want to find the model parameter $\theta$ that maximize the likelihood of getting what we observed. $\theta$ can be computed using gradient ascent on the log likelihood function or gradient descent on the cost function. Here, given $h_\theta(x)=g(x^T\theta)=\frac{1}{1+e^{-\theta^Tx}}$ and $m$ being the number of samples,

$$L(\theta)=p(y|x;\theta)=\prod_{i=1}^{m}(h_\theta(x^{(i)}))^{y^{(i)}}(1-h_\theta(x^{(i)}))^{1-y^{(i)}} \\ logL(\theta)=\sum_{i=1}^{m}y^{(i)}logh_\theta(x^{(i)})+(1-y^{(i)})log(1-h_\theta(x^{(i)}))\\J(\theta)=\frac{1}{m}\sum_{i=1}^{m}-y^{(i)}logh_\theta(x^{(i)})-(1-y^{(i)})log(1-h_\theta(x^{(i)}))$$

### 4 Softmax as generalized linear model

The PMF of a single Multinomial trial, parameterized by the success probability $\phi={\phi_1,\phi_2,...,\phi_k}$ is 

$$\begin{aligned}
p(y ; \phi) &=\phi_{1}^{1\{y=1\}} \phi_{2}^{1\{y=2\}} \cdots \phi_{k}^{1\{y=k\}} \\
&=\phi_{1}^{1\{y=1\}} \phi_{2}^{1\{y=2\}} \cdots \phi_{k}^{1-\sum_{i=1}^{k-1} 1\{y=i\}} \\
&=\phi_{1}^{(T(y))_{1}} \phi_{2}^{(T(y))_{2}} \cdots \phi_{k}^{1-\sum_{i=1}^{k-1}(T(y))_{i}}
\end{aligned}$$

With some manipulation of of the PMF function, we would identify the function $g$ that maps natural parameter to the canonical parameter, and thus solving $h_{\theta}(x)$.

$$\eta_{i}=\log \frac{\phi_{i}}{\phi_{k}} \\\phi_{i}=\frac{e^{\eta_{i}}}{\sum_{j=1}^{k} e^{\eta_{j}}}\\h_{\theta}(x)=\left[\begin{array}{c}\frac{\exp \left(\theta_{1}^{T} x\right)}{\sum_{j=1}^{k} \exp \left(\theta_{j}^{T} x\right)} \\\frac{\exp \left(\theta_{2}^{T} x\right)}{\sum_{j=1}^{k} \exp \left(\theta_{j}^{T} x\right)} \\\vdots \\\frac{\exp \left(\theta_{k-1}^{T} x\right)}{\sum_{j=1}^{k} \exp \left(\theta_{j}^{T} x\right)}\end{array}\right]$$

### 5. Softmax regression as neural network

Softmax can also be seen as a neural network with only a single layer of neuron, with the number of neurons in this layer being the number of total classes. Each neuron in this layer has its own linear part, where the weight matrix $w_i$ and offset $b_i$ are unique. The outputs of these neurons would go into a non-linear activation function (can be viewed as the generalized version of the sigmoid function). This activation function $g$ is determined by mapping the natural parameter of the exponential family to the canonical parameter that parameterizes the probability distribution of a single multinomial trial. The predicted $\hat{y}$ is a $n$-vector where n equals the total number of neurons/classes. 

![Screen Shot 2021-09-15 at 3.02.08 PM.png](Notes%20Binary%20and%20Multinomial%20Logistic%20Regression%20a%20f6e3c3c7d67247fd9e5591c928480b35/Screen_Shot_2021-09-15_at_3.02.08_PM.png)

Softmax as neural network 

### 5. Loss function for softmax regression

We want to find the model parameter $\theta$ that maximize the likelihood of getting what we observed. $\theta$ can be computed using gradient ascent on the log likelihood function or gradient descent on the cost function. Here, n is the number of samples.

$$\begin{aligned}\ell(\theta) &=\sum_{i=1}^{n} \log p\left(y^{(i)} \mid x^{(i)} ; \theta\right) \\&=\sum_{i=1}^{n} \log \prod_{l=1}^{k}\left(\frac{e^{\theta_{l}^{T} x^{(i)}}}{\sum_{j=1}^{k} e^{\theta_{j}^{T} x^{(i)}}}\right)^{1\left\{y^{(i)}=l\right\}}\end{aligned} $$

Taking the negative of the MLE function, we could get the cross-entropy loss function:

$$J(\theta)=-\sum_{i=1}^{n}log\left(\frac{e^{\theta_{l}^{T} x^{(i)}}}{\sum_{j=1}^{k} e^{\theta_{j}^{T} x^{(i)}}}\right) = -log(\hat{y_i})$$