from gliner2 import GLiNER2 
import os
import time
import dotenv
import json
dotenv.load_dotenv()

extractor = GLiNER2.from_pretrained("fastino/gliner2-base-v1")

text = "John Doe is a software engineer at Google. He lives in San Francisco."

start_time = time.time()
entities = extractor.extract_entities(text, ['company', 'person', 'location', 'product'])
end_time = time.time()
print(json.dumps(entities, indent=4))
print(f"Time taken: {end_time - start_time} seconds")
