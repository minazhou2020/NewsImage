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
\

### What you can do with this class:

#### Data Preprocessing
- combine the first two batches of files for training usage
- the third batch used for validation
- crawl training, valation, and test images from given URLs 
- extract features: image id, image url, article_title
- tranlsate article title into English using Google Translate API (https://github.com/ssut/py-googletrans)

#### URL Matching based Method
- Feature Extraction: extract article url and image url from provided file
- remove manually defined stop words from urls
- URL tokenization
- URL comparison: a pair of image url and article url is considered to be matched if they contains more than one common tokens.
- sort the potential matched image list by the number of same tokens
- Evaluate performance of this URL Matching based Method using MR100 on both training dataset and validation dataset

#### Image Captioning based Model
- acquired image caption from the pre-trained image captioning model (https://github.com/ruotianluo/ImageCaptioning.pytorch)
- caculate the wmd between the each pair of image caption and article title
- sort the potential matched image list by the wmd
- Evaluate performance of this Image Captioning based Method using MR100 on both training dataset and validation dataset

#### Face Matching
##### Step 1:  Create a specific training dataset for face-name matching
- A pair of article and image is used for this model training if the pair satisfies following two conditions: 1. the title of article include person's name, 2. the image is a human face image
- Extract person's name from article title
- Remove the image from this specific traning dataset if face can't be detected using multiple face detection frameworks
- Build connections between the extracted name and the corresponding human face image
- If there is no connected image for the extracted name in the training dataset, we crawl five face image using the extracted name as keyword from website.

##### Step 2: Face Name Matching
- Extract the person's names from testing article titles
- Find the corresponding face images from the training dataset which created in step 1
- Encode the face images into feature vectors
- Compare the corresponding face images with each test face image by caculating the cosine distance between two feature vectors
- Two face images are regared as matched if the cosine distance between two vectors is smaller or equal to 0.4 
- Sort the potential matcheing image list by both cosine distance and total matches

##### Step 3:Evaluation

##### Step 4: Prediction