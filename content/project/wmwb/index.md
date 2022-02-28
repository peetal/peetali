---
title: Predictive modeling of human cognitive function

# Summary for listings and search engines
summary: This project primarily focused on 2 hot problems in fMRI research. First, we provided insights on how to significantly increase effect size for fMRI research such that even with relatively small sample size, we would still have enough statistical power to detect such effects. Second, we explore an useful heuristic that guide feature selection to improve predictive models for human cognitive functions. 

# Link this post with a project
projects: []

# Date published
date: "2021-7-06T00:00:00Z"

# Date updated
lastmod: "2021-07-06T00:00:00Z"

# Is this an unpublished draft?
draft: false

# Show this page in the Featured widget?
featured: false

# Featured image
# Place an image named `featured.jpg/png` in this page's folder and customize its options here.
image:
  caption: ""
  focal_point: ""
  preview_only: true


# Custom links (optional).
#   Uncomment and edit lines below to show custom links.
links:
# - name: Follow
#   url: https://twitter.com
#   icon_pack: fab
#   icon: twitter
- name: Code 
  url: https://github.com/peetal/WMBW
  icon_pack: fab
  icon: github
url_code: ""
url_pdf: ""
url_slides: ""
url_video: ""

slides: ""
---

## 1. Background and Goals

### Aim 1: Improving fMRI effect size

- fMRI studies usually suffer from small sample sizes, such that given the typical effect sizes of fMRI studies, a sample size of ~30 people does not provide ample power to detect the real effect if there is any, creating potential spurious findings. One solution here is simple collect for data points. However, due to the nature and cost of fMRI studies, this simple approach turned out to be unpractical in many circumstances. **Another potential solution is that, by adapting certain fMRI data analysis techniques, we might be able to increase effect sizes to a degree such that even with relatively small sample size, we would still gain ample power to detect the effect.**

![Image borrowed from [https://www.dummies.com/article/academics-the-arts/science/biology/the-power-of-a-statistical-hypothesis-test-150310](https://www.dummies.com/article/academics-the-arts/science/biology/the-power-of-a-statistical-hypothesis-test-150310)](Predictive%20d65b2/Screen_Shot_2022-02-27_at_9.34.44_PM.png)

Image borrowed from [https://www.dummies.com/article/academics-the-arts/science/biology/the-power-of-a-statistical-hypothesis-test-150310](https://www.dummies.com/article/academics-the-arts/science/biology/the-power-of-a-statistical-hypothesis-test-150310)

### Aim 2: Improving predictive model for human cognitive functions

- One major line of works in cognitive neuroscience is to build more accurate predictive models for human cognitive functions based on collected neural activity data. Previous works tended to use the whole brain activities for constructing the model. However, the model may benefit from a the law of parsimony, such that a feature selection heuristic may be helpful for improving model performance. **Thus the second goal of this project is to explore a useful feature selection heuristic to build predictive models for human cognitive functions.**

## 2. Analyses and Results

### Clusterings of vertices of the same kind increase effect sizes

- Many works have attempted to construct brain parcellation maps based on its structural and functional architectures. For example, the Gordon parcellation shown below segmented the brain into 333 parcels (each parcel including many vertices) across 13 functional networks (Gordon et al., 2016). The parcellation scheme was defined by abrupt functional connectivity gradient changes in resting state functional connectivity patterns.
- For the current investigation, we predicted that by clustering together vertices of the same kind, the resulted unit (i.e., parcel) would have larger effect size. As we expected, we showed that parcel-level effect sizes are significantly greater than vertex-level effect sizes for all functional networks.

![Screen Shot 2022-02-28 at 10.16.51 AM.png](Predictive%20d65b2/Screen_Shot_2022-02-28_at_10.16.51_AM.png)

### Parcels with larger ES are more useful features for predictive modeling

- We predicted that parcels that showed a large, positive effect size (in working memory tasks) should be more useful features for building predictive models. To test this hypothesis, we binned parcels based on effect sizes and used the same number of parcels from each bin to predict human cognitive functions. Moreover, we constructed null distributions of the ***differences*** between bins if parcels were randomly selected. Thus we were able to test whether the observed predictive power differences between bins are truly significant.

![Screen Shot 2022-02-28 at 10.37.23 AM.png](Predictive%20d65b2/Screen_Shot_2022-02-28_at_10.37.23_AM.png)

- As expected, our results showed that parcel bins of larger effect sizes tend to show significantly greater predictive power. And parcels bins of positive effect size (load-activated parcels) tend to have better predictive power than those bins of negative effective size (load-deactivated parcels).

![Screen Shot 2022-02-28 at 10.50.22 AM.png](Predictive%20d65b2/Screen_Shot_2022-02-28_at_10.50.22_AM.png)

- To further test this heuristic of effect size, we built a series of predictive models by sequentially adding in the next most useful parcel. We compared our feature selection heuristic with step-wise forward selection, and showed that our feature selection heuristic started to perform comparably to step-wise forward selection and eventually became better.

![Screen Shot 2022-02-28 at 11.17.14 AM.png](Predictive%20d65b2/Screen_Shot_2022-02-28_at_11.17.14_AM.png)

## 3. Conclusion

- Per aim 1, we showed that by utilizing the pre-defined parcellation schemes, we could significantly increase effect sizes. Thus, fMRI studies could benefit from such method to gain ample power even with relatively small sample size.
- Per aim 2, we showed that the effect size can be used as an heuristic that guides feature selection for building more powerful predictive models for human cognitive functions.