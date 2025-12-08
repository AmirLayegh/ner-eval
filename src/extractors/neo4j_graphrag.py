import asyncio
import os
from dotenv import load_dotenv

from neo4j_graphrag.experimental.components.entity_relation_extractor import LLMEntityRelationExtractor
from neo4j_graphrag.experimental.components.schema import GraphSchema, NodeType, RelationshipType
from neo4j_graphrag.experimental.components.types import TextChunk, TextChunks
from neo4j_graphrag.llm import OpenAILLM

from src.extractors.base import BaseRelationExtractor, BaseEntityExtractor
from src.models import Triple, Entity

load_dotenv()


class Neo4jGraphRAGRelationExtractor(BaseRelationExtractor):
    """Neo4j GraphRAG-based relation extractor using LLM."""
    
    def __init__(self, model_id: str = "gpt-4o"):
        self._model_id = model_id
        self._llm = OpenAILLM(
            model_name=self._model_id,
            api_key=os.getenv("OPENAI_API_KEY"),
        )
        self._extractor = LLMEntityRelationExtractor(
            llm=self._llm,
            create_lexical_graph=False,
        )

    @property
    def model_id(self) -> str:
        return self._model_id

    def _build_schema(self, relation_types: list[str]) -> GraphSchema:
        """Build GraphSchema from relation types."""
        return GraphSchema(
            node_types=[NodeType(label="Entity")],
            relationship_types=[
                RelationshipType(label=rt) for rt in relation_types
            ]
        )

    def extract(self, text: str, relation_types: list[str]) -> list[Triple]:
        schema = self._build_schema(relation_types)
        
        chunks = TextChunks(chunks=[TextChunk(text=text, index=0)])
        
        result = asyncio.run(self._extractor.run(chunks=chunks, schema=schema))
        
        triples = []
        for rel in result.relationships:
            source_node = next((n for n in result.nodes if n.id == rel.start_node_id), None)
            target_node = next((n for n in result.nodes if n.id == rel.end_node_id), None)
            
            if source_node and target_node:
                triples.append(Triple(
                    subject=source_node.properties.get("name", ""),
                    relation=rel.type,
                    object=target_node.properties.get("name", "")
                ))
        
        return triples

class Neo4jGraphRagEntityExtractor(BaseEntityExtractor):
    """Neo4j GraphRAG-based entity extractor using LLM."""
    
    def __init__(self, model_id: str = "gpt-4o"):
        self._model_id = model_id
        self._llm = OpenAILLM(
            model_name=self._model_id,
            api_key=os.getenv("OPENAI_API_KEY"),
        )
        self._extractor = LLMEntityRelationExtractor(
            llm=self._llm,
            create_lexical_graph=False,
        )

    @property
    def model_id(self) -> str:
        return self._model_id

    def _build_schema(self, entity_types: list[str]) -> GraphSchema:
        """Build GraphSchema from entity types."""
        return GraphSchema(
            node_types=[NodeType(label=et) for et in entity_types]
            # relationship_types=[
            #     RelationshipType(label=rt) for rt in relation_types
            # ]
        )

    def extract(self, text: str, entity_types: list[str]) -> list[Entity]:
        schema = self._build_schema(entity_types)
        chunks = TextChunks(chunks=[TextChunk(text=text, index=0)])
        result = asyncio.run(self._extractor.run(chunks=chunks, schema=schema))
        entities = []
        for node in result.nodes:
            entities.append(Entity(text=node.properties.get("name", ""), type=node.label))
        return entities