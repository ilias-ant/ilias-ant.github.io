---
title: "Adversarial validation: should i trust my validation set?"
date: "2023-06-16"
draft: true

--

A common workflow in machine learning projects (especially in Kaggle competitions) is:

- train your ML model in a training dataset.
- tune your ML model in a validation dataset (which is a discrete subset of the training dataset).
- finally, assess the actual generalization ability of your ML model in a held-out test dataset.

This strategy is widely accepted, as it forces the practicioner to  interact with the ever important test dataset 
only once, at the end of the model selection process - and purely for performance assessment purposes. Any feedback
originating from the interaction with the test dataset does not influence the model selection process, thus preventing
overfitting.

**However**, the success of this strategy heavily relies on the following assumption:

> The training and test datasets are drawn from the same underlying distribution.

But this asssumption may not always hold true...

[work in progress]
