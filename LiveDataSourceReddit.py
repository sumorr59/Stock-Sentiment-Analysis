"""
An implementation for a live data source using the Reddit praw interface. You will need to create an app and use the
values in the initialization of the PRAW instance.
"""

#################################################### IMPORTS ###########################################################
import pandas as pd
import praw
from LiveDataSource import LiveDataSource

######################################## REDIIT IMPLEMENTATION LIVE DATA SOURCE ########################################
# Implementation of a live data source using the Reddit praw API
class LiveDataSourceReddit(LiveDataSource):
    def __init__(self, ticker: str):
        self.ticker = ticker
        self.data = pd.DataFrame(columns=["text"])
        self.reddit = praw.Reddit(
            client_id="your_id",
            client_secret="your_secret",
            user_agent="your_user_agent"
        )

    # Prepares and returns a dataframe from the source of the specific entry amount size
    def load_size(self, entries):
        titles = []
        for result in self.reddit.subreddit("all").search(self.ticker, limit=entries):
            titles.append(result.title)
        self.data = pd.DataFrame(titles, columns=["text"])
        return self.data

    # Prepares and returns a dataframe from the source
    def load(self):
        return self.load_size(entries=10)

    # Converts the data for this instance of the live data source to a list
    def list_data(self) -> [str]:
        return self.data["text"].tolist()

    # Returns the data from this instance of the live data source as a dataframe
    def df_data(self) -> pd.DataFrame:
        return self.data

