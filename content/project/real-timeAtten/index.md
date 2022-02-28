---
title: Real-time Decoding of Attention

# Summary for listings and search engines
summary: This project aims to investigate the neural mechanism that sustain or switch between attentional states. The first experiment combined machine learning framework with non-parametric statistical testing and multivariate analyses to identify neuromarkers that characterize and differentiate between attentional states. The second experiment built upon these findings by using the pre-trained model and a real-time neuro-feedback framework to design a closed-loop attentional focus traninig framework for potential clinical uses. 

# Link this post with a project
projects: []

# Date published
date: "2021-12-11T00:00:00Z"

# Date updated
lastmod: "2021-12-11T00:00:00Z"

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
- name: Code and Pipeline
  url: https://github.com/peetal/rt-project
  icon_pack: fab
  icon: github
url_code: ""
url_pdf: ""
url_slides: ""
url_video: ""

slides: ""
---

## 1. Background

### Psychological phenomenon

- The same external stimuli can be both the target of perception and the trigger of episodic memory retrieval, depending on where attention points to. That is, if attention was deployed externally, the stimuli would lead to perception; if attention was deployed internally, the stimuli would lead to episodic memory retrieval.

<img src="Real-time%20%20800a1/Screen_Shot_2022-02-26_at_11.54.11_AM.png" alt="drawing" width="700"/>

### Goal of the current project

1. Identify the neural mechanism that deploys attention externally vs. internally. 
2. Using real-time neural feedback to allow for flexible attentional switch (this may help with rumination in depression, allowing patients to *NOT* focus their attention on depressive symptoms and the implications of these symptoms). 

## 2. Experimental design

### Experiment 1

- The first experiment aims to examine the neural configuration differences between external and internal attention. Participants first learnt a set of associations between a face (male or female) and a scene (natural or man-made) image. In the external attention condition, participants were asked to make male/female decision for faces and natural/man-made decisions for scenes. In the internal attention condition, participants were asked to retrieve the cue-associated image and make male/female or natural/man-made decisions on the retrieved image. Neural activities were measured using fMRI during the two conditions and a model was trained to classify the ongoing attentional state based on neural signals.

### Experiment 2 (in progress)

- The second experiment aims to perform real-time neuro-feedback training. While in the scanner, participants were presented with an external stimuli. The current attentional states were being quantified using the pre-trained model, and the classification confidence was used as neuro-feedback. Participants were asked to either sustain or switch between attention states.

<img src="Real-time%20%20800a1/Screen_Shot_2022-02-26_at_4.43.32_PM.png" alt="drawing" width="700"/>

## 3. Analyses pipeline

### Preprocessing pipeline (post fmriprep)

- The minimal preprocessed timeseries consist of 2 components of information. First, there is a “event-related” component that track external stimuli. Second, there is a “state-related” component that track ongoing cognitive states. Here I used Finite Impulse Response (FIR) model to *regress out* the stimuli-driven task related component from the data.

<img src="Real-time%20%20800a1/IMG_0052.png" alt="drawing" width="700"/>


### Full correlation matrix analysis

- We used the full correlation matrix analysis to examine and identify the difference in neural configurations between external and internal attention. FCMA here serves as an efficient feature selection tool that uses nested cross validation to selected the most important set of brain connections that characterize and differentiate external from internal attentional states.

<img src="Real-time%20%20800a1/Screen_Shot_2022-02-26_at_4.52.09_PM.png" alt="drawing" width="700"/>

## 4. Results

### Distinct brain configurations between external vs. internal attention

- First we showed that whole brain interaction patterns significantly different between external vs. internal attentional states. Importantly, such difference was not driven by cognitive demands (e.g., task accuracy), but specifically tracks attentional states.

<img src="Real-time%20%20800a1/Screen_Shot_2022-02-26_at_5.16.58_PM.png" alt="drawing" width="700"/>

### Network structure differences

- Through information mapping and clustering, we identified 16 brain clusters across 3 functional communities, whose connectivity patterns track external vs. internal attentional states. Specifically, control network regions (as shown in red color) showed stronger within-community interactions during external attention while the default mode network (as shown in blue color) showed stronger within-community interactions during internal attention. Moreover, the retrosplenial cortices (as shown in green color) coupled stronger with the control regions during internal attention, but with the default mode network regions during external attention.

<img src="Real-time%20%20800a1/Screen_Shot_2022-02-26_at_5.19.51_PM.png" alt="drawing" width="700"/>


### Importance of retrosplenial cortices

- We showed that the activity and interaction patterns of the retrosplenial cortices accurately and sensitively tacked ongoing cognitive states and state-related tasks. recent work have suggested that RSC could be the hub of connecting external and internal worlds, such that it integrates external cues with self-generated information to guide behavior. This view has its basis from the rodent literature. For example, it was shown that RSC in rodents integrates both the allocentric mapping (animal’s location in the external world) and egocentric frame (animal’s internal representation of the location) to navigate through a maze  by combining information from sensory inputs and the medial temporal network.

<img src="Real-time%20%20800a1/Screen_Shot_2022-02-26_at_5.24.36_PM.png" alt="drawing" width="700"/>


## 5. Conclusion and future direction

- Experiment 1 identified the strong neural markers for external vs. internal attention. Experiment 2 would use these heuristics to pre-train binary classification SVM models, and use the classification confidence (i.e., the decision function output) as neural-feedbacks to guide participants to sustain or switch between attention states