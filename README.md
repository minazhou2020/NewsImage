# DL-TXST NewsImages: Contextual Feature Enrichment for Image-Text Re-matching

This repository contains Yuxiao Zhou's contribution to the [MediaEval 2021: NewsImages](https://multimediaeval.github.io/editions/2021/tasks/newsimages/) challenge, and also the code for our paper: [DL-TXST NewsImages: Contextual Feature Enrichment for Image-Text Re-matching](https://2021.multimediaeval.com/paper49.pdf).

## Introduction

In the project, we build multiple models to describe the connection between the textual content of articles and the images that accompany them, and employed these models to predict which image was published with a given news article in the provided dataset.

We evaluate our proposed model on the benchmark dataset derived from four months of webserver log files of a German  news publisher. The performance of the proposed model is  measured by image matching precision such as MRR and Mean Recall at different depths.

## Data Pipelines

![Data Pipeline](https://github.com/minazhou2020/NewsImage/blob/main/img/FinalProjectDiagram.png)

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

Run1 combines three different methods. Equal weights are assigned to the categorization-based method, and a combination of face-name matching and image captioning-based methods. The ranking of a candidate image in Run1 is: 

𝑅𝑅Run1 =0.5𝑅𝑅 Categorization +0.5 (𝑅𝑅Face+𝑅𝑅caption) 

Run2 combines all proposed methods. The first three models are ensembled using the same approach as in Run1. This ensembled model is used for creating the initial top 100 image list. Then we append the result, which is generated from the URL matchingbased method, to the end of the top 100 image list. 

Run3 is like Run2. The only difference is that we append the result of last method to the head of the top 100 image list. Since the image URL is an artificial feature, the results from Run1 and Run3 are not included in the result. 

Evaluation Result on Train and Validation Dataset

<img src="https://github.com/minazhou2020/NewsImage/blob/main/img/result.png" alt="Results from different Runs on Test Dataset" width="500"/>

Results from different Runs on Test Dataset 

<img src="https://github.com/minazhou2020/NewsImage/blob/main/img/result_1.png" alt="Results from different Runs on Test Dataset" width="500"/>

## Notebooks

 [Jupyter Notebooks](https://github.com/minazhou2020/NewsImage/tree/main/notebooks) are provided with instructions to run the models on new datasets.

## License

This code is distributed under MIT LICENSE

## Author

Yuxiao Zhou, [yuxiao.zh@gmail.com](mailto:yuxiao.zh@gmail.com)