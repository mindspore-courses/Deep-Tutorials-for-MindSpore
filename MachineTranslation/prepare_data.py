"""prepare data for dataset"""

from utils import download_data, prepare_data

download_data(data_folder="/transformer data")

prepare_data(data_folder="/transformer data",
             euro_parl=True,
             common_crawl=True,
             news_commentary=True,
             min_length=3,
             max_length=150,
             max_length_ratio=2.,
             retain_case=True)
