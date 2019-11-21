# Yelp Review Prediction
This repository hosts the code for the CS 544 end of semester project.

Python version: 3.7.5

Installation:

    pip install gensim
    pip install matplotlib
    pip install numpy
    pip install pandas
    pip install sklearn
    pip install 

To run our code:

`python3 run.py <location of folder containing yelp dataset>`

The above code first extracts review and business data from the original Yelp dataset and writes to "dataset.json". It then preprocesses the dataset and writes to "normalised_data.json. Finally, it generates three word2vec models for each of the normalised dataset.

The deep learning model implementation is to be continued.