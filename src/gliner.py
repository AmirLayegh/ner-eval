from gliner2 import GLiNER2 
import os
import time
import dotenv
import json
dotenv.load_dotenv()

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
        for sample in self.samples:
            text = sample['text']
            start_time = time.time()
            entities = self.extract(text)
            end_time = time.time()
            total_time += end_time - start_time
            results.append(entities)
        return results, total_time/len(self.samples)

class GliNER2RE:
    def __init__(self, model_id: str="fastino/gliner2-base-v1", file_path: str="data/re/scientist_re_benchmark.json"):
        self.model_id = model_id
        self.file_path = file_path
        self.extractor = None
        self.data = None
        self.samples = None
        self.relation_types = None
        self.extractor = GLiNER2.from_pretrained(self.model_id)
    
    def set_data(self):
        with open(self.file_path, 'r') as f:
            data = json.load(f)
            self.data = data
            self.samples = data['samples']
            self.relation_types = data['metadata']['relation_types']

    def extract(self, text):
        # GLiNER2 returns: {'relation_extraction': {'rel_type': [(source, target), ...]}}
        result = self.extractor.extract_relations(text, self.relation_types)
        return result

    def run(self):
        self.set_data()
        results = []
        total_time = 0
        for sample in self.samples:
            text = sample['text']
            start_time = time.time()
            result = self.extract(text)
            end_time = time.time()
            total_time += end_time - start_time
            
            # Convert GLiNER2 output to list of triples
            # Handle both formats: 
            # - {'relation_extraction': {'rel_type': [(subj, obj), ...]}}
            # - {'rel_type': [(subj, obj), ...]}
            triples = []
            
            # Check if wrapped in 'relation_extraction' key
            if 'relation_extraction' in result:
                rel_data = result['relation_extraction']
            else:
                rel_data = result
            
            for rel_type, pairs in rel_data.items():
                for pair in pairs:
                    # Handle tuples of different lengths (might have confidence score)
                    if len(pair) >= 2:
                        subj, obj = pair[0], pair[1]
                        triples.append({
                            "subject": subj,
                            "relation": rel_type,
                            "object": obj
                        })
            results.append(triples)
        return results, total_time/len(self.samples)
