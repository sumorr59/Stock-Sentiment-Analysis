'''
A model interface for using given DataSources to make predictions for future input.
'''

#################################################### IMPORTS ###########################################################
from TrainingDataSource import TrainingDataSource

################################################### INTERFACE ##########################################################
# An interface for a model. A model is created from using a given DataSource and is able to then make predictions
# from future input
class Model:
    # Builds a model_info object from a given DataSource
    def build_from_data_source(self, tds: TrainingDataSource):
        raise NotImplementedError("The method not implemented")

    # Classifies the sentence; returns tuple of form (classification, prob(classification))
    # if prob(classification) cannot be calculated then return None
    def classify(self, sentence: str):
        raise NotImplementedError("The method not implemented")

    # Batch form of classification
    # Classifies the sentence; returns tuple of form (classification, prob(classification))
    # if prob(classification) cannot be calculated then return None
    def batch_classify(self, sentences: [str]):
        raise NotImplementedError("The method not implemented")
