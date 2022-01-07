"""
Implementation for a DataSource object that uses CSV data.
"""

#################################################### IMPORTS ###########################################################
from csv import reader
import pandas as pd
from TrainingDataSource import TrainingDataSource, TrainingDataEntry

############################################# CSV TRAINING SOURCE CLASS ################################################
# An implementation for a DataSource object that uses CSV data. For now used primarily for stocks_twitter.csv
# CSV file format must be --> SENTENCE, SENTIMENT
class CSVTrainingDataSource(TrainingDataSource):
    # Initialize an instance of the csv training source
    def __init__(self, file: str):
        self.data = []
        self.file = file

    # Loads the csv reader to get the data from the csv into the class
    def load(self):
        with open(self.file) as csv_file:
            csv_reader = reader(csv_file, delimiter=',')
            for line in csv_reader:
                self.data.append(TrainingDataEntry(line[0], int(line[1])))
        print(f'Data from {self.file} loaded')

    # List the data
    def list_data(self):
        return self.data

    # Returns a dataframe (DF) of the data
    def df_data(self) -> pd.DataFrame:
        df = pd.read_csv(self.file)
        return df.rename(columns={df.columns[0]: 'sentences', df.columns[1]: 'labels'})
