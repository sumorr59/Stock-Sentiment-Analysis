"""
Provides an example of how to use a live data source.
Note: this will not work after clone due to missing credentials.

Steps:
1. Create a new data source with the specified securities.
2. Load with specified entry size.
"""

from LiveDataSourceReddit import LiveDataSourceReddit
data_source = LiveDataSourceReddit(ticker="AMZN")
data_source.load_size(5)
print(data_source.list_data())
