# FakeNewsIdentifier

Overview

This project aims to detect fake news using natural language processing (NLP) techniques and machine learning models. The dataset consists of news articles labeled as real or fake, and the goal is to build a classification model that can accurately identify fake news based on textual features.


Make sure you have Python installed on your system along with the following libraries:
numpy
pandas
matplotlib
seaborn
nltk
scikit-learn
wordcloud

You can install these libraries using pip:
pip install numpy pandas matplotlib seaborn nltk scikit-learn wordcloud

Dataset
The dataset used for this project includes:

train.csv: Contains labeled training data with columns for title, author, text, and label (0 for real news, 1 for fake news).
test.csv: Contains unlabeled test data with columns for title, author, and text.

Usage
Data Loading and Preprocessing: Load the dataset using pandas and preprocess the text data by removing stopwords, punctuation, and performing lemmatization.
WordCloud Visualization: Generate WordClouds to visualize the most frequent words in real and fake news articles.
Feature Extraction: Vectorize the text data using CountVectorizer and TfidfTransformer to prepare it for model training.
Model Training: Train machine learning models such as Logistic Regression and Naive Bayes on the vectorized text data.
Model Evaluation: Evaluate the trained models using accuracy scores and confusion matrices.
Pipeline Creation: Create a pipeline that includes text preprocessing, feature extraction, and model training for future use.
Saving the Pipeline: Save the pipeline using joblib for making predictions on new data.

Files Included
fake_news_detection.ipynb: Jupyter Notebook containing the entire project code with detailed explanations and comments.
train.csv: Labeled training data.
test.csv: Unlabeled test data.
Running the Notebook

Open the fake_news_detection.ipynb file in Jupyter Notebook or JupyterLab.
Run each cell sequentially to execute the code and see the outputs.
Follow the instructions and comments in the notebook for detailed explanations of each step.
Author

[Victor Vo]

License

This project is licensed under the MIT License - see the LICENSE file for details.
