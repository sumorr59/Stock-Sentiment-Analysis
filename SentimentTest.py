"""
A file for sentiment module testing
***Note*** : BERTModel (in models = ...) set size should be increased. Smaller sizes allow for quicker training, but
will have bad performance.
"""

#################################################### IMPORTS ###########################################################
from BertModel import BERTModel
from model_bayes_markov import NaiveBayesModel, MarkovModel
# Local Imports
from CSV_TrainingDataSource import CSVTrainingDataSource

################################################## RUN TEST ############################################################
training_data = CSVTrainingDataSource("stocks_twitter.csv")
training_data.load()

models = [NaiveBayesModel(), MarkovModel(), BERTModel(set_size=100)]

for model in models:
    model.build_from_data_source(training_data)

# Allow user to interact with the sentiment analysis
while True:
    sentence = input("Input sentence to analyze\n")
    labels = [model.classify(sentence) for model in models]
    print("       | Class | Std prob")
    print(f"Markov |   {labels[0][0]}   | {labels[0][1]}")
    print(f"NaiveB |   {labels[1][0]}   | {labels[1][1]}")
    print(f"BERT   |   {labels[2][0]}   | {labels[2][1]}")
