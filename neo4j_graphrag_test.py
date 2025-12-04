from neo4j_graphrag.experimental.components.entity_relation_extractor import LLMEntityRelationExtractor
from neo4j_graphrag.experimental.components.schema import (
    GraphSchema,
    NodeType,
    RelationshipType,
    PropertyType,
)
from neo4j_graphrag.experimental.components.types import TextChunk, TextChunks
from neo4j_graphrag.llm import OpenAILLM
import os
import asyncio
import json
import time
import dotenv
dotenv.load_dotenv()



class Neo4jGraphRagExtractor:
    def __init__(self, model_id: str="gpt-4o", file_path: str="data/science_ner_benchmark.json"):
        self.model_id = model_id
        self.file_path = file_path
        self.data = None
        self.samples = None
        self.entity_types = None
        self.extractor = None
        self.llm = OpenAILLM(
            model_name=self.model_id,
            api_key=os.getenv("OPENAI_API_KEY"),
        )

    def set_data(self):
        with open(self.file_path, 'r') as f:
            data = json.load(f)
            self.data = data
            self.samples = data['samples']
            self.entity_types = data['metadata']['entity_types']
    
    def set_schema(self):
        self.schema = GraphSchema(
            node_types=(
                NodeType(label=et) for et in self.entity_types
            )
        )
    
    def set_extractor(self):
        self.extractor = LLMEntityRelationExtractor(
            llm=self.llm,
            create_lexical_graph=False,
        )
    def run(self):
        self.set_data()
        self.set_schema()
        self.set_extractor()
        results = []
        total_time = 0
        for sample in self.samples[:10]:
            text = sample['text']
            chunks = TextChunks(chunks=[TextChunk(text=text, index=0)])
            start_time = time.time()
            result = asyncio.run(self.extractor.run(chunks=chunks, schema=self.schema))
            end_time = time.time()
            nodes_result = []
            for node in result.nodes:
                nodes_result.append({ "label": node.label, "properties": node.properties })
            results.append(nodes_result)
            total_time += end_time - start_time
        return results, total_time/len(self.samples[:10])

# Initialize the LLM
# llm = OpenAILLM(
#     model_name="gpt-4o",
#     api_key=os.getenv("OPENAI_API_KEY"),
# )

# # Create the extractor
# extractor = LLMEntityRelationExtractor(
#     llm=llm,
#     create_lexical_graph=False,  # Set to False since we only want entities/relations
# )

# # Test text - same as gliner2_test.py for comparison
# text = "John Doe is a software engineer at Google. He lives in San Francisco."

# # Prepare text chunks (required format for the extractor)
# chunks = TextChunks(chunks=[
#     TextChunk(text=text, index=0),
# ])

# async def run_extraction():
#     start_time = time.time()
#     # Pass the schema to the run() method
#     result = await extractor.run(
#         chunks=chunks,
#         schema=schema
#     )
#     end_time = time.time()
    
#     print("Extracted Nodes:")
#     for node in result.nodes:
#         print(f"  - {node.label}: {node.properties}")
    
#     print("\nExtracted Relationships:")
#     for rel in result.relationships:
#         print(f"  - {rel.type}: {rel.start_node_id} -> {rel.end_node_id}")
    
#     print(f"\nTime taken: {end_time - start_time:.4f} seconds")
#     return result

# # Run the async extraction
# if __name__ == "__main__":
#     result = asyncio.run(run_extraction())
