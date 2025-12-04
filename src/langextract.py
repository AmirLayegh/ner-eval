#  This script is used to test the langextract library.
#  This is mainly used for Named Entity Recognition (NER) but it can also be used for Relation Extraction (RE).
#  For RE, we need to prompt the model to extract the properties with a specific format which is not quite straightforward.
#  It is compulsory to provide the examples for the model to learn the format of the output.

import langextract as lx
import textwrap
import time
import json
import dotenv
dotenv.load_dotenv()

class LangExtractor:
    def __init__(self, model_id: str="gpt-4o", file_path: str="data/science_ner_benchmark.json"):
        self.model_id = model_id
        self.file_path = file_path
        self.data = None
        self.samples = None
        self.prompt = None
        self.examples = None

    def set_data(self):
        with open(self.file_path, 'r') as f:
            data = json.load(f)
            self.data = data
            self.samples = data['samples']
            self.entity_types = data['metadata']['entity_types']
    def set_prompt(self):
        self.prompt = textwrap.dedent("""
        Extract the entities with the types of the following: {self.entity_types}.
        Use exact text for extraction. Do not paraphrase or overlap entities.
        """)

    def set_examples(self):
        self.examples = [
            lx.data.ExampleData(
                text="They may also use Adenosine triphosphate, Nitric oxide, and ROS for signaling in the same ways that animals do.",
                extractions=[
                    lx.data.Extraction(
                        extraction_class="chemicalcompound",
                        extraction_text="Adenosine triphosphate",
                    ),
                    lx.data.Extraction(
                        extraction_class="chemicalcompound",
                        extraction_text="Nitric oxide",
                    ),
                    lx.data.Extraction(
                        extraction_class="chemicalcompound",
                        extraction_text="ROS",
                    ),
                ]
            ),
            lx.data.ExampleData(
                text="August Kopff, a colleague of Wolf at Heidelberg, then discovered 617 Patroclus eight months after Achilles, and, in early 1907, he discovered the largest of all Jupiter trojans, 624 Hektor.",
                extractions=[
                    lx.data.Extraction(
                        extraction_class="scientist",
                        extraction_text="August Kopff",
                    ),
                    lx.data.Extraction(
                        extraction_class="scientist",
                        extraction_text="Wolf",
                    ),
                    lx.data.Extraction(
                        extraction_class="location",
                        extraction_text="Heidelberg",
                    ),
                    lx.data.Extraction(
                        extraction_class="astronomicalobject",
                        extraction_text="617 Patroclus",
                    ),
                    lx.data.Extraction(
                        extraction_class="astronomicalobject",
                        extraction_text="Achilles",
                    ),
                    lx.data.Extraction(
                        extraction_class="misc",
                        extraction_text="Jupiter trojans",
                    ),
                    lx.data.Extraction(
                        extraction_class="astronomicalobject",
                        extraction_text="624 Hektor",
                    ),
                ]
            ),
        ]

    def extract(self, text):
        return lx.extract(text, self.prompt, examples=self.examples, model_id=self.model_id)
    
    def run(self):
        self.set_data()
        self.set_prompt()
        self.set_examples()
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

   
#             lx.data.Extraction(
#                 extraction_class="company",
#                 extraction_text="Neo4j",
#                 attributes={
#                     "relationship_with": "Emil Eifrem",
#                     "relationship_type": "FOUNDED_BY",
#                 }
#             ),
#             lx.data.Extraction(
#                 extraction_class="location",
#                 extraction_text="San Francisco",
#             ),
#             lx.data.Extraction(
#                 extraction_class="location",
#                 extraction_text="San Francisco",
#             ),
#         ]
#     ),
# ]



# text = "John Doe is a software engineer at Google. He lives in San Francisco."

# start_time = time.time()
# result = lx.extract(text, prompt, examples=examples, model_id="gemini-2.5-pro")
# end_time = time.time()
# print(result)
# print(f"Time taken: {end_time - start_time} seconds")