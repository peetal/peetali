---
title: Speeding up Computation of Pearson Correlation
subtitle: This post shows a simple but handy trick for efficiently speeding up pearson correlation computation. This is also the first trick used by FCMA to speed up its computation. Specifically, pearson correlation could be done use matrix multiplication after standardization (e.g., zscore)

# Link this post with a project
projects: []

# Date published
date: "2021-08-17T00:00:00Z"

# Date updated
lastmod: "2021-08-17T00:00:00Z"

# Is this an unpublished draft?
draft: false

# Show this page in the Featured widget?
featured: false

# Featured image
# Place an image named `featured.jpg/png` in this page's folder and customize its options here.
authors:
- admin

reading_time: false
tags:

categories:
---

## **Main Take-away:**
The simple but efficient trick to compute Pearson Correlation faster is to 
1. Scale the data
2. Perform matrix algebra to compute the correlation coefficient. <br>

This simple trick in principle can make the computation 3-4 times faster and can be useful when dealing with large vectors and/or need to compute r for multiple things in parallel. Here I'm going to demonstrate this trick first and then quickly explain why it is the case. 

## **1. Demo**
Let's generate two random vectors, each has the length of $10^8$


```python
sim_v1 = np.random.rand(100000000,)
sim_v2 = np.random.rand(100000000,)
```

To compute the pearson correlation between the two vectros, we could use the function pearsonr from scipy.stats


```python
start = time.time()
print(f'The pearson correaltion coefficient is {pearsonr(sim_v1, sim_v2)[0]}')
end = time.time()
print(f'Time elapsed: {end - start}')
```

    The pearson correaltion coefficient is 0.00014023607618081493
    Time elapsed: 2.6575210094451904


Or we could scale the two vectors first, and then compute the dot product between them


```python
sim_v1_scale = zscore(sim_v1)
sim_v2_scale = zscore(sim_v2)
N = len(sim_v1_scale)
start = time.time()
print(f'The pearson correaltion coefficient is {np.dot(np.array(sim_v1_scale), np.array(sim_v2_scale))/N}')
end = time.time()
print(f'Time elapsed: {end - start}')
```

    The pearson correaltion coefficient is 0.00014023607618081355
    Time elapsed: 1.0209388732910156


Interestingly, the two approaches to compute pearson r gives exactly the same output, but the second approach is much faster.  

## **2. Why it is the case**

The typical formula for computing pearson r is $$ r = \frac{\sum(x - \bar{x})(y - \bar{y})}{\sqrt{\sum(x - \bar{x})^2(y - \bar{y})^2}}$$
When both vectors are standarized, meaning both vectors center at 0 and have SDs equal to 1. Thus we have $\bar{x} = 0$; $\bar{y} = 0$ and $\frac{\sum(x - \bar{x})^2}{N} = 1$; $\frac{\sum(y - \bar{y})^2}{N} = 1$, with N being the length of the vector (assuming ddof = 0)
Now say we have two standarized N vectors $\tilde{x}$ and $\tilde{y}$. The pearson correlation between the two vectors can be computed as $$r = \frac{\sum(\tilde{x} - 0)(\tilde{y} - 0)}{\sqrt{N^2}} = \frac{\sum\tilde{x}\tilde{y}}{N} = \frac{\tilde{x}^T\tilde{y}}{N}$$
The time difference between the two approaches should not have a crazy difference as the complexity for both computations were bounded by O(n). However, matrix multiplication can benefit from modern CPUs parallel computing techniques such as SIMD (single instruction, multiple data). 
