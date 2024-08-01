---
layout: post
title: Comprehensive study on object detection for iOS
date: 2024-02-11 21:01:00
description: How to find ourself in an array of frameworks and model parameters for SOTA results.
tags: computer-vision object-detection ios yolo rtmdet
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

### A bit of history

I was always interested in object detection since the first years of Deep Learning (around 10 years ago). In those days R-CNN was the first one to apply CNN for object detection, which was later improved in the Faster-RCNN which a bunch of people still uses today (no idea why though). Those early detectors usually created object proposal and evaluate them more or less like a simple classifier. Then, YOLO and SSD proposed, with great success, single-shot detectors with much better performance. I specially remember 
when YOLOv2 was launched, it worked wonderfully and we were able to [deploy several detectors in parallel in GPU](https://medium.com/@meerkat.cv/object-detection-and-augmented-reality-the-cup-noodles-band-case-be1567e81c72) for a project in early 2018. Since then a lot of things improved, YOLO is already [in version 10](https://arxiv.org/abs/2405.14458) and we have a bunch of frameworks to run these model incredibly fast in different hardwares, including Android and iOS devices.

#### Why still use an object detector these days?

Nowadays, when I think in object detector I remember a phrase from Redmond in the YOLOv3 papers (btw, this is still one my favorite paper/technical report ever):

> Boxes are stupid anyway though, I’m probably a true believer in masks except I can’t get YOLO to learn them.

It's always seems to me that instance segmentation are much better for almost all applications, but sometimes our project can present constrains that forces our hand. I see two advantages in using object detection: first, it's much easier to annotate datasets (using boxes rather than pixels); and second, the models tend to be much faster, especially if you're considering real-time operation on mobile devices, which is our goal here.

Allow me to make another small digression here about image annotation. I'm still 

========================================================================
pytorch mobile -->

nao dá para dizer que é só o libtorch, porque exporta com pytorch, e na hora de usar se usa o libtorch via cocoapods. mas é importante a gente entender se quando tu tá fazendo o export do modelo (em python) to tá indo para o torchscript. O torchscript vai ser interpretado, e tem um cara efficient pra isso em https://pytorch.org/tutorials/recipes/mobile_interpreter.html mas nota que deve ser exportado de maneira diferente.

How is ExecuTorch Different from PyTorch Mobile (Lite Interpreter)?
PyTorch Mobile uses TorchScript to allow PyTorch models to run on devices with limited resources. ExecuTorch has a significantly smaller memory size and a dynamic memory footprint resulting in superior performance and portability compared to PyTorch Mobile. Also, ExecuTorch does not rely on TorchScript, and instead leverages PyTorch 2 compiler and export functionality for on-device execution of PyTorch models.

Read more in-depth technical overview topics about ExecuTorch:

coreml -->

tem limitações nas versões de iOS que vão estar disponíveis, principalmente no caso de compressão de modelos e .mlpackage O melhor seria exportar para mlmodel.

interessante que convertendo para coreml a gente teve que fixar um tamanho
no caso do executorch tem como fazer input dinamico.

executorch -->

Seu eu achar o executorch meio merda para uns modelos mais complexos (mmdetection lib)
eu posso só falar sobre tamanho de modelo, facilidade de código, suporte a tipos de ops, opções de quantization.

is there a way to export to coreml comming from executorch. pelo que entendi tem um estágio intermediário que precisa do código do nn.module (ver um vídeo sobre)

executorch seria mais fácil começar pelo ultralytics, eu acho. Maaaas, é o mmdet.registry (or something like that) poderia nos trazer o module e a gente seguir daí... (interesting)


-->
IOS FRAMEWORKS SIZE

importante a gente saber o tamanho que adicionar via cocoapods aumenta pro caso do pytorch mobile, executorch, onnxruntime, coreml (?), etc.

--> 
A moral é que tem um monte de escolhas importantes na hora de tu portar um modelo para mobile. Eu estou escrevendo um post sobre isso, alguma das coisas que eu to tentando metrificar:
a) caminho para conversão do modelo (origem quase sempre Pytorch) é fácil/reproduzível?
b) qual o número de parâmetros do modelo (maior que 40M começa a complicar, pensando que ficaria uns 10MB quantizado em int8). Ou seja, qual tamanho do modelo?
c) Tempo de inferência no mobile?
d) Qual o requisito mínimo de iOS? (as vezes quantizar já te pede um iOS mais novo).
e) como é o código para integrar a app? tem bridge para Objective-c++ ou é tudo via Swift (que é bem mais fácil)?
f) qual o tamanho da biblioteca necessária para incluir o modelo? libtorch é maior que executorch que é maior que usar CoreML direto, por exemplo.
g) qual a licensa do modelo? Tenho visto muita gente largando só utilização para open source. Dá para re-implementar, mas né....
h) qual a maturidade/suporte do pipeline? executorch é uma lib super nova tá sempre mudando....

--> when possible, we should include some conversion code taken from ultralytics or other to show how it's simple/not.

--> which are the default way of saving pytorch models (without optimization/compile/export)

--> the executorch sample of bridging is much more interesting, I suppose I could start using it, instead of torchscript/pytorch mobile

--> parece que tem um jeito de fazer o CoreML baixar e compilar o modelo no background! Cooooool: https://developer.apple.com/documentation/coreml/downloading-and-compiling-a-model-on-the-user-s-device ; achei meio loco que o executorch tem isso como opção, não sei qual a diferença no caso dele.