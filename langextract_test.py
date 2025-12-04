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
    def __init__(self, model_id="gemini-2.5-pro"):
        self.model_id = model_id

    def set_prompt(self):
        self.prompt = textwrap.dedent("""
        Extract the entities with the type company, person, location, product.
        Use exact text for extraction. Do not paraphrase or overlap entities.
        """)

    def set_examples(self):
        self.examples = [
            lx.data.ExampleData(
        text="Emil Eifrem is the CEO of Neo4j. He lives in San Francisco.",
        extractions=[
            lx.data.Extraction(
                extraction_class="person",
                extraction_text="Emil Eifrem",
            ),
            lx.data.Extraction(
                extraction_class="company",
                extraction_text="Neo4j",
            ),
            lx.data.Extraction(
                extraction_class="location",
                extraction_text="San Francisco",
            ),
            lx.data.Extraction(
                extraction_class="location",
                extraction_text="San Francisco",
            ),
        ]
    ),
    ]

    def extract(self, text):
        return lx.extract(text, self.prompt, examples=self.examples, model_id=self.model_id)
    
    def run(self):
        self.set_prompt()
        self.set_examples()
        return self.extract(text)

   
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



text = "John Doe is a software engineer at Google. He lives in San Francisco."

start_time = time.time()
result = lx.extract(text, prompt, examples=examples, model_id="gemini-2.5-pro")
end_time = time.time()
print(result)
print(f"Time taken: {end_time - start_time} seconds")