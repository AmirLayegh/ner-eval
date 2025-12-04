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

# Define the schema with node types, relationship types, and patterns
schema = GraphSchema(
    node_types=(
        NodeType(
            label="Person",
            description="An individual human being",
            properties=[
                PropertyType(name="name", type="STRING", description="The person's name"),
            ],
        ),
        NodeType(
            label="Company",
            description="A business organization",
            properties=[
                PropertyType(name="name", type="STRING", description="The company name"),
            ],
        ),
        NodeType(
            label="Location",
            description="A geographical place",
            properties=[
                PropertyType(name="name", type="STRING", description="The location name"),
            ],
        ),
    ),
    # relationship_types=(
    #     RelationshipType(label="WORKS_AT", description="Person works at a company"),
    #     RelationshipType(label="LIVES_IN", description="Person lives in a location"),
    #     RelationshipType(label="LOCATED_IN", description="Company is located in a location"),
    # ),
    # patterns=(
    #     ("Person", "WORKS_AT", "Company"),
    #     ("Person", "LIVES_IN", "Location"),
    #     ("Company", "LOCATED_IN", "Location"),
    # ),
)

# Initialize the LLM
llm = OpenAILLM(
    model_name="gpt-4o",
    api_key=os.getenv("OPENAI_API_KEY"),
)

# Create the extractor
extractor = LLMEntityRelationExtractor(
    llm=llm,
    create_lexical_graph=False,  # Set to False since we only want entities/relations
)

# Test text - same as gliner2_test.py for comparison
text = "John Doe is a software engineer at Google. He lives in San Francisco."

# Prepare text chunks (required format for the extractor)
chunks = TextChunks(chunks=[
    TextChunk(text=text, index=0),
])

async def run_extraction():
    start_time = time.time()
    # Pass the schema to the run() method
    result = await extractor.run(
        chunks=chunks,
        schema=schema
    )
    end_time = time.time()
    
    print("Extracted Nodes:")
    for node in result.nodes:
        print(f"  - {node.label}: {node.properties}")
    
    print("\nExtracted Relationships:")
    for rel in result.relationships:
        print(f"  - {rel.type}: {rel.start_node_id} -> {rel.end_node_id}")
    
    print(f"\nTime taken: {end_time - start_time:.4f} seconds")
    return result

# Run the async extraction
if __name__ == "__main__":
    result = asyncio.run(run_extraction())
