---
layout: post
title: Caution while using Large Language Models
date: 2024-04-20 21:01:00
description: Weird failures and behavior from LLMs.
tags: machine-learning large-language-models llm
categories: posts
thumbnail: assets/img/llm_failure_img1.jpeg
---


<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/llm_failure_img1.jpeg" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Image taken from [2].
</div>


In the past two years or so, everyone in the IA/ML is talking about Large Language Models (LLMs) and how to use them in a bunch of tasks. I'm also very interested in how these models can help us do better research and applications.

We're also seeing some clever applications of LLMs in Computer Vision beginning to appear. On example is the recent paper ScreenAI from Google [1] which uses LLMs (coupled with Vision models) to automatically generate a ton of useful training data.

But today I want to highlight another cool paper by Alzahrani et al., in which they found how sensible to small changes in prompt are LLMs benchmarks and rankings. To attest this, in multiple choice question benchmark (MMLU), they change the options from ["A", "B", "C", "D"] to something like ["$", "&", "#", "@"] and found that most LLMs perform worse merely due to this subtle change (it also completely change the ranking). Even worse, they notice that only swapping the position of answers could would also lead to a decrease in performance!

The authors also refer to a nice work (still unpublished) by Den et al. 2023 [3] that tried to find if, for some LLMs, there was leakage for MMLU. In their experiment, they remove (wrong) answers from the question's prompt and ask the LLM to predict the missing portion of the text. They found that most LLMs are able to recover these missing portions of the questions, which strongly suggest that somehow these benchmarks were used (perhaps not directly) in training.

I think the takeaway here is that we need more consistent LLMs or methods (which are being researched) and that we should be careful in using these rankings/benchmarks to drive our choice of which model to use.

## References

_[1] - Baechler, Gilles, et al. "ScreenAI: A Vision-Language Model for UI and Infographics Understanding." arXiv preprint arXiv:2402.04615 (2024)._

_[2] - When Benchmarks are Targets: Revealing the Sensitivity of Large Language Model Leaderboards: https://lnkd.in/dpBW3SAW_

_[3] - Investigating Data Contamination in Modern Benchmarks for Large Language Models: https://lnkd.in/dBRdVVav_


