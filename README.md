# DL-TXST NewsImages: Contextual Feature Enrichment for Image-Text Re-matching

This repository contains Yuxiao Zhou's contribution to the [MediaEval 2021: NewsImages](https://multimediaeval.github.io/editions/2021/tasks/newsimages/) challenge, and also the code for our paper: [DL-TXST NewsImages: Contextual Feature Enrichment for Image-Text Re-matching](https://2021.multimediaeval.com/paper49.pdf).

In this paper, we describe our multi-view approach to the news image re-matching to text for news articles run submission. Our feature pool consists of provided features, baseline text and image features using pre-trained and domain adapted modeling, and contextual features for the news and image article. We have evaluated multiple modeling approaches for the features and employed a deep multi-level encoding network to predict a probability-like matching score of images for a news article. Our best results are the ensemble of proposed models, and we found the URL for the image and related images provides the most discriminative context in this pairing task.

# Notebooks

- [Guide_for_Using_Classes.ipynb](https://github.com/minazhou2020/NewsImage/blob/main/notebooks/Guide_for_Using_Classes.ipynb): Jupyter notebook file explaining how to use the all python class
- [Data_Preprocessing.ipynb](https://github.com/minazhou2020/NewsImage/blob/main/notebooks/Data_Preprocessing.ipynb) : Jupyter notebook file explaining how to use the all Data_Preprocessing class
- [URL_Matching.ipynb](https://github.com/minazhou2020/NewsImage/blob/main/notebooks/URL_Matching.ipynb): Jupyter notebook file explaining how to use the all URL_Matching class
- [Image_Captioning based Model.ipynb](https://github.com/minazhou2020/NewsImage/blob/main/notebooks/Image_Captioning%20based%20Model.ipynb): Jupyter notebook file explaining how to use the all URL_Matching class
- [Face_Matching.ipynb](https://github.com/minazhou2020/NewsImage/blob/main/notebooks/Face_Matching.ipynb): Jupyter notebook file explaining how to use the all Face_Name_Matching class
- [Model_Ensembling.ipynb](https://github.com/minazhou2020/NewsImage/blob/main/notebooks/Model_Ensembling.ipynb): Jupyter notebook file explaining how to use the all Model_Ensembling class

# Python Classes

- [Class Description](https://github.com/minazhou2020/NewsImage/blob/main/Class_Description.md)
- [Data_Preprocessing.py](https://github.com/minazhou2020/NewsImage/blob/main/src/Data_Preprocessing.py)  
- [Face_Name_Matching.py](https://github.com/minazhou2020/NewsImage/blob/main/src/Face_Name_Matching.py)  
- [Image_Caption.py](https://github.com/minazhou2020/NewsImage/blob/main/src/Image_Caption.py)  
- [Model_Ensembling.py](https://github.com/minazhou2020/NewsImage/blob/main/src/Model_Ensembling.py)  
- [URL_Matching.py](https://github.com/minazhou2020/NewsImage/blob/main/src/URL_Matching.py)
- [Experiment.py](https://github.com/minazhou2020/NewsImage/blob/main/src/Experiment.py)  

# Data

The repository also includes some of the source and result datasets used in the project.

- [Processed data](https://git.txstate.edu/CS7311/FIREWHEEL/tree/master/Yuxiao/processed_data/data)
- [Processed images](https://git.txstate.edu/CS7311/FIREWHEEL/tree/master/Yuxiao/processed_data/img)
- [Results](https://git.txstate.edu/CS7311/FIREWHEEL/tree/master/Yuxiao/result)

### Requirements:

Python 3.7
tensorflow\
pytorch\
py-googletrans\
NLTK\
gensim\
cv2\
icrawler\
DeepFace\
scipy\
matplotlib\

##### 