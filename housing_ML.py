
#Example
import pandas as pd 
import kagglehub
from kagglehub import KaggleDatasetAdapter

file_name = 'melb_data.csv'

melbourne_housing_df = kagglehub.dataset_load(
    KaggleDatasetAdapter.PANDAS,
    "dansbecker/melbourne-housing-snapshot",
    file_name,
)

melbourne_housing_df.describe()