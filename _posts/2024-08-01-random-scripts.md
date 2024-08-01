---
layout: post
title: Three random scripts.
date: 2024-08-01 11:34:00
description: I made some cool little code snippets that I like it to share.
tags: computer-vision arxiv real-madrid
categories: posts
thumbnail: assets/img/mobilesam_annotator_sample.gif
---


<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/mobilesam_annotator_sample.gif" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    MobileSAM single click annotation tool.
</div>

In the last couple of months, I've been doing some random stuff and would like to share three random scripts. Hopefully, they will be useful for some people and indexed by search engines (if that’s a thing) or LLMs in the future. Here they are, in increasing order of randomness (from the perspective of a CV guy):

### 1. Script to annotate single objects in a bunch of images, MobileSAM Annotator.

This script turned out into its own project, check the [repo](https://github.com/gustavofuhr/mobilesam_annotator).

I'm trying to study some of the SAM (Segment Anything Model) papers in and noticed that in the [original paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Kirillov_Segment_Anything_ICCV_2023_paper.pdf) they use the model to automatically annotate images which train the model in a training looping scheme. The foundation model itself is very powerful and could have a bunch of applications in the next few years in the industry and has recently (July 2024) been upgraded ([SAM2](https://ai.meta.com/blog/segment-anything-2/)). I still believe that specific models will be more efficient for most problems, yet SAM models can be an easy starting point. 

For one application that I'm developing, I tried the [demo online](https://segment-anything.com/demo) and found that it worked greatly and want it to run locally. I search for a few papers that attempted to speed-up these segmentation foundation models and find [MobileSAM](https://arxiv.org/pdf/2306.14289). I tried the [provided models](https://github.com/ChaoningZhang/MobileSAM) and was surprised by how fast they are (running at around 220ms per image in my M3 Mac). Using it, I developed a single-point annotation tool, the GIF in the beginning of this post. The results are exported as separated binary images and visualizations are also stored so that you can better inspect any errors in the segmented regions. It worked great for my application and I still have some cool ideas to improve it: support for more points, more classes (objects) and integration with detection/segmentation frameworks.

### 2. Script to rename arXiv files:

Here's the [script](https://gist.github.com/gustavofuhr/83a7d5cb9bd93fdc6b74852a2f29dc67).

I read a lot of papers (or try to anyways) and the majority of them are available on the arXiv archive. Usually I download them to an iCloud-synced  so that I can download/read papers on any device. But it always bothers me that the files have non-informative names with numbers, like `2403.05440.pdf`. I made a simple script to rename the files using the arXiv API. The API is free, and it was news for me that it even existed. It was quite easy to work with it:

```python
def fetch_paper_details_arxiv(arxiv_number):
    url = f"http://export.arxiv.org/api/query?id_list={arxiv_number}&max_results=1"
    d = feedparser.parse(url)
    return d
```

tl;dr: it takes all the arXiv PDFs from a folder and rename them, such that 

`2404.04125v3.pdf`

becomes

`2404.04125 - 2024 - No "Zero-Shot" Without Exponential Data, Pretraining Concept Frequency?  Determines Multimodal Model Performance.pdf`

I know, it's simple, but now you don't have to do it :-).

### 3. Script to get notified when the general public tickets for a Real Madrid match are first available.

This is a fun one ([Gist link](https://gist.github.com/gustavofuhr/4ce0edfc49d985f2094d4d4930a5761a)). 

I'm a big fan of football: after all, I'm a Brazilian. During my last vacation, I fulfilled a long-awaited dream of watching live a Real Madrid match (if you're curious it was against Alavés). If you don't know, Real Madrid is the one the most famous teams in the world, so tickets are very hard to get. So, I made a script that checks periodically (every 5 min or so) if the tickets are available for sale; it will then send you an email. But how would I know if the script stopped running due to some problem? I also send a ping email every hour.

For this script, I simply used Selenium to visit the webpage and check for a string that tell us tickets are available. I actually thought someone had already made this, but couldn't find it. Anyways, the match was amazing, 5-0, two goals from Vini Jr.!


<div class="row mt-4">
    <div class="col-sm mt-4 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/real_madrid.jpeg" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Bernabéu is gorgeous.
</div>