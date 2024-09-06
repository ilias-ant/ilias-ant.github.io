---
title: "Do we even need ML?"
date: "2024-01-31"
draft: false
category: posts
tags:
  - machine-learning
  - problem-solving
keywords:
  - machine-learning
  
cover:
  image: "blog-posts/do-we-even-need-ml/cover.png"
  alt: "counting"
  caption: "choosing the right tools for the job - image created by DALL-E model."
  relative: false
  responsiveImages: true

#cover:
#  image: "blog-posts/adversarial-validation/cover.png"
#  alt: "counting"
#  caption: "training & test datasets are not always drawn from the same distribution."
#  relative: false
#  responsiveImages: true

---

Having worked in the industry for 6 years now, I've seen machine learning projects succeed but, most 
importantly, I've seen many more fail - I even failed some of them myself due to inexperience and/or poor judgement. 
Although each failure has its own story, reasons and learnings, some common denominators always exist. 

One such important denominator can be "boiled down" to a single assertion that - as an ML engineer - I am particularly 
fond of:

> **_"Chances are, you do NOT need ML to solve THAT problem."_**

I get it - building a machine learning model is exciting stuff. Observing it as it generates predictions on your 
company's production environment is even more exciting. 

But, there's "no free lunch".

Being a good engineer in the industry is typically not about building complex, state-of-the-art sophisticated solutions - 
this is not academia. It's all about generating constant and considerable value for the product and company that you 
work for, while being cost-effective. To do that, you should be dead certain which is the right tool and approach 
for each problem that you're required to solve. ML is just one of many tools in an engineer's toolbox.

Coincidentally, the last couple of weeks I have been reading [Chip Huyen](https://huyenchip.com/)'s great book called 
[Designing Machine Learning Systems](https://www.oreilly.com/library/view/designing-machine-learning/9781098107956/).
This realization, that ML is not always the right tool for the job, is one of the main takeaways from the book. 

Chip went a step further and created a list of 9 criteria that we should consider before commiting to an ML solution for
our problem. I found this list to be very accurate & useful, so decided to share it here. ML is the right approach when:

- **the system has the capacity to learn.**
- **there are patterns to learn, and they are complex.**
- **data are available, or it's possible to collect data.**
- **it's a predictive problem.**
- **unseen data shares patterns with the training data.**
- **it's repetitive.**
- **the cost of wrong predictions it's cheap.**
- **it's at scale.**
- **the patterns are constantly changing.**

Next time you're faced with a problem that you think ML might be the right tool for, consider these 9 criteria. Go 
through each one of them and see if they apply to your problem. Make sure that all of them are met before you commit to 
an ML solution. If not, chances are that there exists a simpler, more cost-effective solution for this problem. 
Take some time to think what this solution might be.