---
layout: post
title: The future of OCR is not to use OCR!
date: 2024-04-20 21:01:00
description: End-to-end document understanding will be huge.
tags: ocr document-understanding computer-vision
categories: posts
thumbnail: assets/img/ocr_future_img1.jpeg
---


<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/ocr_future_img1.jpeg" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Image taken from [1], Donut paper.
</div>

Around 3-4 years ago, some colleagues and I launched a product that cleverly used OCR outputs to retrieve information from document images. We usually explain that this product was intended for generic documents, meaning it was not trained for a specific layout or style. With a few simple rules (such as: give me the text below the label “Nome”) you can extract a structure JSON containing the most important information from these documents.

The set of possible “rules” was pre-defined and mainly related to the OCRs positions and text without fully understanding them. Looking back, we failed to estimate how hard it was for users to create good rules for extracting the required information. Also, the pipeline for extracting info was very complex, resembling this:

Get image -> Detect text boxes -> Read OCR -> Aggregate OCR into blocks, lines -> Apply rules for document -> Output JSON.

It was also difficult to adapt these pipelines to other important information that might be interesting to extract, such as checkmarks in forms and signatures.

Some very cool methods for OCR-free document understanding appeared in the past few years. One of the first was Donut [1], by Clova (which also provided awesome OCRs and text detectors in the past). The idea was to go from the image to a JSON output with a single end-to-end model:
Get image -> Apply fine-tuned model for a given task -> Output JSON.
The presented results are impressive. The model is composed of an image encoder and text decoder built using transformers. The training is done (as it seems to be the norm these days) in two stages. First, in the pre-training phase, the task is to predict masked text in images (an objective analogous to OCR). 

Later the model is fine-tuned to a specific task, such as:
- document classification, i.e.: what kind of document it is?
- key information extraction, e.g.: extract the store address from this invoice
- VQA (Visual Question Answering), a generic question such as “Which is the document’s holder birthday?”

This kind of solution is much more versatile and easy to implement in a product.

To not extend this further, these are some other cool related works that I found:
Dessurt [2]: similar to Donut, made around the same time, with less impressive results.
Pix2Struct [3]: an extension of Donut for dealing with any type of image (documents, screenshots, UI). The pre-training was done with 80M screenshots of HTML images and the object was to predict the HTML of these images. How clever was that?
ScreenAI [4]: a recent work from Google that tackles QA, UI navigation, etc. It uses LLMs to create huge annotated datasets. Can you imagine a Siri that actually can be useful and use apps?

That's it this week, let me know what you think.

## References

_[1] - Kim, Geewook, et al. "Ocr-free document understanding transformer." European Conference on Computer Vision. Cham: Springer Nature Switzerland, 2022._

_[2] - Davis, Brian, et al. "End-to-end document recognition and understanding with dessurt." European Conference on Computer Vision. Cham: Springer Nature Switzerland, 2022._

_[3] - Lee, Kenton, et al. "Pix2struct: Screenshot parsing as pretraining for visual language understanding." International Conference on Machine Learning. PMLR, 2023._

_[4] - Baechler, Gilles, et al. "ScreenAI: A Vision-Language Model for UI and Infographics Understanding." arXiv preprint arXiv:2402.04615 (2024)._