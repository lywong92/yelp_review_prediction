from data_extractor import DataExtractor
#from model import build_model
from preprocess import write_to_file, preprocess_data
import sys

dataset_folder = sys.argv[1]
dataset_file = "dataset.json"
normalised_dataset_file = "normalised_data.json"

# extract data from review.json and business.json
data_extractor = DataExtractor(dataset_folder)
data_extractor.extract()
data_extractor.write_to_file()

# preprocess data and write final datasets to normalised_data.json
preprocess_data(dataset_file)

# build model
#build_model(normalised_dataset_file)
