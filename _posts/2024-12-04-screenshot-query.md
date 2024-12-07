---
layout: post
title: ScreenshotQuery, make queries to screenshots using Vision Language Models.
date: 2024-12-04 11:34:00
description: Describe and talk with your set of screenshots
tags: computer-vision VLM image-retrieval image-caption semantic-search
categories: posts
thumbnail: assets/img/screenshot_query_thumb.png
images:
  slider: true
---


<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include video.liquid path="assets/video/screenshot_query_03.mp4" class="img-fluid rounded z-depth-1" controls=true autoplay=true %}
    </div>
</div>
<div class="caption">
    A notebook to query images.
</div>

A couple of months ago I started a small project that I thought would be a simple task from my ever growing to-do list: processing approximately 1000 screenshots so I can
perform some kind of image retrieval using natural language on them. It seemed simple enough yet it had some details that I would like to share in this post.

Here's the [project at Github](https://github.com/gustavofuhr/screenshot_query).

🚨 Before we start: Screenshots are usually very personal and might contain a bunch of personal information. If you want to use this code, ensure that you are comfortable sending them to OpenAI or other LLMs.

#### Which problem are we tackling here?

It's important to clarify that we'll delegate the image captioning to an MLLM model, even if there are other [available models](https://paperswithcode.com/task/image-captioning) can run locally. If I started again I would probably try to use [BLIP-2](https://paperswithcode.com/paper/blip-2-bootstrapping-language-image-pre) to generate captions/embeddings.

So, our ultimate goal is to search images through textual content descriptions using natural language. Image retrieval (and scene classification) has a long history in Computer Vision -- remember the seminal [Bag of Visual Words](https://people.eecs.berkeley.edu/~efros/courses/AP06/Papers/csurka-eccv-04.pdf) model? *Yet, here we're only trying to use natural language to query a set of image descriptions; everything is text*. The most appropriate nomenclature I found was "similarity search", which is sometimes used in (now trending) RAG systems. 

Anyway, the next sections discuss the steps that I took to achieve the objective.   

### 1. Creating image descriptions

The first thing is to send to a MLLM (Multi-modal Large Language Model), such as GPT4-o, all the images with a simple prompt asking the model to describe the
image contents:

```python

def get_image_description_openai(base64_image):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"
    }

    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {
            "role": "user",
            "content": [
                {
                "type": "text",
                "text": "What’s in this image?"
                },
                {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}"
                }
                }
            ]
            }
        ],
        "max_tokens": 300
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload).json()
    return response["choices"][0]["message"]["content"]

```

You can find the whole script [here](https://github.com/gustavofuhr/screenshot_query/blob/main/generate_image_descriptions_openai.py). That may take a lot of time, but it can be resumed multiple times. It's also surprisingly cheap to send those images to OpenAI (for 1k images I guess it was around 2-3 USD total!). The script will create txt files describing the images, one for each image.

#### How good are the descriptions?

It's impressively good! Check these:

<swiper-container keyboard="true" navigation="true" pagination="true" pagination-clickable="true" pagination-dynamic-bullets="true" rewind="true" slideShadows="false">
  <swiper-slide>{% include figure.liquid loading="eager" path="assets/img/screenshot_query/sq_desc_02.png" class="img-fluid rounded" %}</swiper-slide>
  <swiper-slide>{% include figure.liquid loading="eager" path="assets/img/screenshot_query/sq_desc_01.png" class="img-fluid rounded z-depth-1" %}</swiper-slide>
  <swiper-slide>{% include figure.liquid loading="eager" path="assets/img/screenshot_query/sq_desc_03.png" class="img-fluid rounded z-depth-1" %}</swiper-slide>
  <swiper-slide>{% include figure.liquid loading="eager" path="assets/img/screenshot_query/sq_desc_04.png" class="img-fluid rounded z-depth-1" %}</swiper-slide>
</swiper-container>


### 2. Making queries 

I got two main ideas for this: 
- Create text embeddings for each description and perform an ANN (Approximate Nearest Neighbors) search for the query.
- Send the text containing all the image descriptions to the LLM with a prompt to retrieve the top-N most relevant image depictions.

#### 2.1 Text embeddings {#emb}

The idea here is to represent the whole description of the image contents as a vector. That will also be done later for the query so we can search the closest embeddings to the query, under a certain distance threshold.I chose to use ANN since K-nearest neighbors will be too slow for larger datasets, in case I want to process my entire gallery someday.

We'll explore two ways to create these embeddings, but first, let's talk about how we can store them and perform search. This is usually done with a vector 
database, and we can find several solutions for it: 
[Milvus](https://milvus.io/), [Vertex AI](https://cloud.google.com/vertex-ai/docs/vector-search/overview) from Google, [Weaviate](https://github.com/weaviate/weaviate) etc. I had some experience in the first two but want to try Weaviate since it appeared to have a very easy setup. For search, it appears
that Weaviate uses the [HNSW algorithm](https://arxiv.org/abs/1603.09320) for ANN (Approximate Nearest Neighbors) which is probably available in the other
solutions too -- It's worth mentioning that the awesome [Faiss](https://github.com/facebookresearch/faiss) has implementations of several ANN algorithms if one wants to explore different methods.

Using Weaviate was indeed very easy. Here's how to start its Docker container:
```bash
docker run -p 8080:8080 -p 50051:50051 cr.weaviate.io/semitechnologies/weaviate:1.26.3
```

One thing that I found kind of annoying is that they recently changed the interface of the Python library. The version used in the following snippets is `weaviate-client==4.7.1`. First, you need to create a class, which is like a collection or a table if you're not familiar with the term.

```python
client = weaviate.Client(WEAVIATE_URL)

class_obj = {
    "class": CLASS_NAME,
    "properties": [
        {"name": "filename", "dataType": ["string"]},
        {"name": "description", "dataType": ["string"]}
    ]
}
client.schema.create_class(class_obj)
```

After that you can populate the class with the embeddings that you want:

```python
client.data_object.create({"filename": "image.png", "description": "This is an image containing...}, 
                                CLASS_NAME, vector=description_embedding_vector)
```

For searching you just need to provide three parameters: the query embedding, the min distance and the 
limit of objects to return:

```python
response = collection.query.near_vector(
        near_vector=query_embedding,
        distance=distance,
        limit=n_images_limit,
    )
```

For creating the embeddings we explore two alternatives: SBERT and OpenAI API.

#### 2.1.1 SBERT embeddings

A couple of years ago, [BERT](https://arxiv.org/abs/1810.04805) was the main option for creating text embeddings. It was used in a bunch of downstream tasks quite successfully. However, for comparing sentences, which is our goal, it’s not well suited mainly because it doesn’t generate fixed-size embeddings. If you want to use BERT for our case (semantic search), it would take a lot of time. From the [SBERT paper](https://arxiv.org/pdf/1908.10084) abstract we have an impressive example:

>  Finding the most similar pair in a collection of 10,000 sentences requires about 50 million inference computations (~65 hours) with BERT. 

SBERT fixes this by focusing on semantic similarity tasks. With it, it's possible to generate fixed embeddings for sentences (or paragraphs). For the above example, SBERT can reduce the elapsed time from ~65 hours to 5 seconds with the same accuracy.

Here's the code: ([Reference](https://huggingface.co/sentence-transformers))

```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("all-MiniLM-L6-v2")
feat = model.encode(text)
```

This will generate embeddings with a size of 384.

#### 2.1.2 OpenAI embeddings

It’s expected that modern LLMs provide a way to get embeddings from text, and that is indeed the case. Since I wanted to use only the $10 USD credits I added to OpenAI for this project, I’m going to use their “text-embedding-3-small” model. Generating embeddings on the OpenAI API is super simple:

```python
payload = {
    "model": "text-embedding-3-small",
    "input": text
}

response = requests.post("https://api.openai.com/v1/embeddings", headers=headers, json=payload).json()
embedding = response["data"][0]["embedding"]
```

Notice that there are [larger models available](https://platform.openai.com/docs/guides/embeddings/embedding-models#embedding-models) for computing embeddings that should give us better results.
I just finished a small project on Text-to-SQL, and in that case, the use of larger models was indeed the 
deciding factor between success and failure (GPT4-o instead of GPT4-o-mini).

### 2.2 Results using embeddings

Once the embeddings are created and stored in the Weaviate collection, we can query them, as we explained in the [previous section](#emb). To see if it is working I'm showing results for 4 different queries:

- Image showing any type of scale models.
- Screenshots showing smartphone lock screens.
- Screenshots that show research papers.
- Which screenshots show 3D printing models/objects or 3D printers.


Here's some results:

SBERT embeddings:
<swiper-container keyboard="true" navigation="true" pagination="true" pagination-clickable="true" pagination-dynamic-bullets="true" rewind="true" slideShadows="false">
  <swiper-slide>{% include figure.liquid loading="eager" path="assets/img/screenshot_query/sbert/cropped_scale.png" class="img-fluid rounded" %}</swiper-slide>
  <swiper-slide>{% include figure.liquid loading="eager" path="assets/img/screenshot_query/sbert/cropped_screen.png" class="img-fluid rounded z-depth-1" %}</swiper-slide>
  <swiper-slide>{% include figure.liquid loading="eager" path="assets/img/screenshot_query/sbert/cropped_research.png" class="img-fluid rounded z-depth-1" %}</swiper-slide>
  <swiper-slide>{% include figure.liquid loading="eager" path="assets/img/screenshot_query/sbert/cropped_3D.png" class="img-fluid rounded z-depth-1" %}</swiper-slide>
</swiper-container>

It works, but there are some images with correct descriptions that should not be associated with the query. For instance, the first query has nothing to do with the screenshot of a research paper (4th screenshot in the first image), even though the description of the image is correct:

> The image seems to be a figure from a research paper titled "LOLNeRF: Learn from One Look." It illustrates a method for reconstructing 3D models from 2D images using a neural network. 
> Key components include:
> - **Authors**: The names of researchers associated with different institutions are listed.
> - **Method Overview**: It outlines a process involving coarse pose estimation and the use of a conditioned NeRF model for training and testing.
> - **Images**: Multiple images of faces appear to depict the results of the model, showing various views or representations of different individuals.
> The figure is likely meant to summarize the approach and findings of the research visually.

The query about research papers also did not return very good examples. It has some weird matches that are not articles/papers at all.

Open AI embeddings:
<swiper-container keyboard="true" navigation="true" pagination="true" pagination-clickable="true" pagination-dynamic-bullets="true" rewind="true" slideShadows="false">
  <swiper-slide>{% include figure.liquid loading="eager" path="assets/img/screenshot_query/openai/cropped_scale.png" class="img-fluid rounded" %}</swiper-slide>
  <swiper-slide>{% include figure.liquid loading="eager" path="assets/img/screenshot_query/openai/cropped_screen.png" class="img-fluid rounded z-depth-1" %}</swiper-slide>
  <swiper-slide>{% include figure.liquid loading="eager" path="assets/img/screenshot_query/openai/cropped_research.png" class="img-fluid rounded z-depth-1" %}</swiper-slide>
  <swiper-slide>{% include figure.liquid loading="eager" path="assets/img/screenshot_query/openai/cropped_3D.png" class="img-fluid rounded z-depth-1" %}</swiper-slide>
</swiper-container>

The results are similar but with fewer errors. That said, this entire analysis was qualitative. Conducting more structured experiments would be far more interesting (well, next time then).

### 2.3 Asking chatGPT to search

LLMs perform incredibly well on zero-shot tasks, but sometimes it’s hard to have an intuition if they will solve your problem. I tried to send descriptions of all images with an instruction prompt like this:

```python
initial_context = """
Your job is to find the top {N_TOP_IMAGES} best screenshots based on a query and the image descriptions. 

Bellow there a bunch of image descriptors preceded by their filename. Make sure that, in your answer
you only include the image filenames in order from most relevant to least relevant from the chosen {N_TOP_IMAGES}.

Example, given a query like this:

"Which screenshots appear to be tech products?""

You should answer like, without text before this and without numbers before the image names (THIS IS AN FORMAT EXAMPLE, THESE IMAGES DON'T EXISTS ):
IMAGE_01.PNG
IMAGE_02.jpeg
...

First there is a list of image filenames, only answer with these filenames in the order of relevance:

{image_filenames}

The following is a list of image descriptor (one for each image), you should use these to answer the query.

{image_descriptors}

"""
```

Only using prompt engineering like that may work, but has two annoying problems. First, it is very easy to reach a token limit, even for less than a thousand images (one may be tempted to use Gemini or other APIs that offer longer context windows, but that won't scale). Second, it is hard to force coherent outputs. For smaller models, it even returns image names that do not exist, and sometimes, the results come with a text preamble that changes each time and that needs to be removed for processing. In my research in text2SQL, that was mitigated using larger models (which don’t hallucinate as much) and by searching the part of the answer that actually corresponds to a formal SQL language.

## 3. That's it.

Using LLMs to help with side projects is awesome, especially if you are lazy. Things that would require two or three days of coding can be done in a few hours -- with less quality for sure, but still  a functional prototype that will give you an insight into whether your idea makes sense or not. That's where this project comes from, the whole "chatGPT should work for that" new attitude that is changing programming for good. It’s still not as good as 90% of people that hype it up say, but it’s fun.

I’m especially keen to ask the LLMs to structure the solution for the problem itself, so you can discover new ideas and tools that would be a pain to research by yourself.Asking how to solve the problem described here, chatGPT suggested using SBERT and Weaviate, which were outside my radar before that. Still, I think that the ideas here need a little more theoretical foundation so that we understand why sometimes completely unrelated texts may present near embeddings; maybe I will revisit this in the future.