"""
Defines an interface for training data sources and a class for single data entries.
"""

#################################################### IMPORTS ###########################################################
import pandas as pd

################################################ TRAINING DATA ENTRY ###################################################
# Representation of a single entry of data. Sentiment is either 0 or 1 --> bad or good
class TrainingDataEntry:
    # Initializes a data entry
    def __init__(self, sentence: str, sentiment: int):
        self.sentence = sentence
        self.sentiment = sentiment

# An interface for a source of data to be used in training models
class TrainingDataSource:
    # Loads and returns a dataframe of data i.e. collect data from an API (nltk, etc.), load file data into memory, etc.
    def load(self):
        raise NotImplementedError("The method not implemented")

    # Returns a list of Training Data Entries
    def list_data(self) -> [TrainingDataEntry]:
        raise NotImplementedError("The method not implemented")

    # Returns a dataframe of entries
    def df_data(self) -> pd.DataFrame:
        raise NotImplementedError("The method not implemented")



