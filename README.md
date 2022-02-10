# DL-TXST NewsImages: Contextual Feature Enrichment for Image-Text Re-matching

This repository contains Yuxiao Zhou's contribution to the [MediaEval 2021: NewsImages](https://multimediaeval.github.io/editions/2021/tasks/newsimages/) challenge, and also the code for our paper: [DL-TXST NewsImages: Contextual Feature Enrichment for Image-Text Re-matching](https://2021.multimediaeval.com/paper49.pdf).

## Introduction

In the project, we build multiple models to describe the connection between the textual content of articles and the images that accompany them, and employed these models to predict which image was published with a given news article in the provided dataset.

We evaluate our proposed model on the benchmark dataset derived from four months of webserver log files of a German  news publisher. The performance of the proposed model is  measured by image matching precision such as MRR and Mean Recall at different depths.

## Data Pipelines

![Data Pipeline](https://github.com/minazhou2020/NewsImage/blob/main/img/FinalProjectDiagram.png)

## Benchmark Dataset

The Multimedia Evaluation Benchmark (MediaEval) NewsImage task offered data covering four months of news from a German news publisher. The data contains information related to articles, images, and interactions with users. Each article and images has a reference number assigned. Articles’ meta data includes the URL, title, and a text snippet of at most 256 characters. 

The provided data consists of four batches of data in total. The first three are used for training, and the last one is used for testing. The training data contains the links between articles and images as well as the interaction statistics, while such links have been removed from test data.

## Requirement

* Programming Lanugage: Python3
* Data Exploring and Analysis: Pandas
* Deep Learning Model: Tensorflow, Keras, Pytorch, Scikit-Learn
* Computer Vision: OpenCV
* Natural Language Processing: nltk

## Results

![Results](https://github.com/minazhou2020/NewsImage/blob/main/img/result.png)

## Notebooks

 [Jupyter Notebooks](https://github.com/aarcosg/traffic-sign-detection/blob/master/Run_models_on_new_images.ipynb) are provided with instructions to run the models on new datasets.

## License

This code is distributed under MIT LICENSE

## Author

Yuxiao Zhou, [yuxiao.zh@gmail.com](mailto:yuxiao.zh@gmail.com)