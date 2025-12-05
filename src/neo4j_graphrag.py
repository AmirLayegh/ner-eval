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
        for sample in self.samples:
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
        return results, total_time/len(self.samples)


class Neo4jGraphRagRE:
    def __init__(self, model_id: str="gpt-4o", file_path: str="data/re/scientist_re_benchmark.json"):
        self.model_id = model_id
        self.file_path = file_path
        self.data = None
        self.samples = None
        self.relation_types = None
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
            self.relation_types = data['metadata']['relation_types']
    
    def set_schema(self):
        # Generic Entity type - we only care about relation extraction
        self.schema = GraphSchema(
            node_types=[NodeType(label="Entity")],
            relationship_types=[
                RelationshipType(label=rt) for rt in self.relation_types
            ]
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
        for sample in self.samples:
            text = sample['text']
            chunks = TextChunks(chunks=[TextChunk(text=text, index=0)])
            start_time = time.time()
            result = asyncio.run(self.extractor.run(chunks=chunks, schema=self.schema))
            end_time = time.time()
            # Extract relationships as (subject, relation, object) triples
            rels_result = []
            for rel in result.relationships:
                # Find the source and target node names
                source_node = next((n for n in result.nodes if n.id == rel.start_node_id), None)
                target_node = next((n for n in result.nodes if n.id == rel.end_node_id), None)
                if source_node and target_node:
                    rels_result.append({
                        "subject": source_node.properties.get("name", ""),
                        "relation": rel.type,
                        "object": target_node.properties.get("name", "")
                    })
            results.append(rels_result)
            total_time += end_time - start_time
        return results, total_time/len(self.samples)