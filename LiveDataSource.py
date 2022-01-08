"""

Defines the interface for a live data source. A live data source loads the desired text from somewhere
into memory. The text source can vary, potentially being an API or scraped from a website

"""

#################################################### IMPORTS ###########################################################
import pandas as pd

################################################ LIVE DATA SOURCE ######################################################
# Interface for a source of data to be used to train Models
class LiveDataSource:
    # Prepares and returns this data sources data
    def load(self):
        raise NotImplementedError("The method not implemented")

    # Returns a list of data entries
    def list_data(self) -> [str]:
        raise NotImplementedError("The method not implemented")

    # Returns a dataframe of data
    def df_data(self) -> pd.DataFrame:
        raise NotImplementedError("The method not implemented")
