import json
import os
import sys

class DataExtractor:
    def __init__(self, path):
        self.path = os.path.join(path, '')
        self.business_data_folder = os.path.join(self.path, 'business.json')
        self.review_data_folder = os.path.join(self.path, 'review.json')
        self.output_file_name = './dataset.json'
        self.output_data = {}
        self.unique_categories = {}
    
    def extract(self):
        with open(self.business_data_folder) as bd_handle:
            business = bd_handle.readline()

            while business:
                bd_data = json.loads(business)

                if bd_data['categories']:
                    self.output_data[bd_data['business_id']] = {
                        'categories': [cat.strip() for cat in bd_data['categories'].split(',')],
                        'reviews': []
                    }

                business = bd_handle.readline()

            bd_handle.close()
        
        with open(self.review_data_folder) as rd_handle:
            review = rd_handle.readline()

            while review:
                rv_data = json.loads(review)

                if rv_data['business_id'] in self.output_data:
                    review = {}
                    review['review_id'] = rv_data['review_id']
                    review['text'] = rv_data['text']
                    review['useful'] = rv_data['useful']
                    review['funny'] = rv_data['funny']
                    review['cool'] = rv_data['cool']
                    self.output_data[rv_data['business_id']]['reviews'].append(review)

                review = rd_handle.readline()

            bd_handle.close()

    def write_to_file(self):
        data_to_write = []
        for business_id in self.output_data:
            categories = self.output_data[business_id]['categories']

            if 'Restaurants' in categories and len(self.output_data[business_id]['reviews']) > 0:
                cat_to_write = ':'.join(categories)
                for review in self.output_data[business_id]['reviews']:
                    review_id = review['review_id']
                    text = review['text']
                    useful = str(review['useful'])
                    funny = str(review['funny'])
                    cool = str(review['cool'])
                    data_to_write.append({
                        'business_id': business_id,
                        'categories': cat_to_write,
                        'review_id': review_id,
                        'text': text,
                        'useful': useful,
                        'funny': funny,
                        'cool': cool
                    })           
        
        with open(self.output_file_name, 'w') as of_handle:
            json.dump(data_to_write, of_handle)
            of_handle.close()