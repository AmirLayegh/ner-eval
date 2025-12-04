from gliner2 import GLiNER2 
import os
import time
import dotenv
import json
dotenv.load_dotenv()

# extractor = GLiNER2.from_pretrained("fastino/gliner2-base-v1")

# text = "John Doe is a software engineer at Google. He lives in San Francisco."

# start_time = time.time()
# entities = extractor.extract_entities(text, ['company', 'person', 'location', 'product'])
# end_time = time.time()
# print(json.dumps(entities, indent=4))
# print(f"Time taken: {end_time - start_time} seconds")



class GliNER2Extractor:
    def __init__(self, model_id: str="fastino/gliner2-base-v1", file_path: str="data/science_ner_benchmark.json"):
        self.model_id = model_id
        self.file_path = file_path
        self.extractor = None
        self.data = None
        self.samples = None
        self.extractor = GLiNER2.from_pretrained(self.model_id)
    
    def set_data(self):
        with open(self.file_path, 'r') as f:
            data = json.load(f)
            self.data = data
            self.samples = data['samples']
            self.entity_types = data['metadata']['entity_types']

    def extract(self, text):
        
        entities = self.extractor.extract_entities(text, self.entity_types)
        return entities

    def run(self):
        self.set_data()
        results = []
        total_time = 0
        for sample in self.samples[:10]:
            text = sample['text']
            start_time = time.time()
            entities = self.extract(text)
            end_time = time.time()
            total_time += end_time - start_time
            results.append(entities)
        return results, total_time/len(self.samples[:10])