from gliner2 import GLiNER2
from src.extractors.base import BaseRelationExtractor, BaseEntityExtractor
from src.models import Triple, Entity


class GLiNERRelationExtractor(BaseRelationExtractor):
    """GLiNER2-based relation extractor."""
    
    def __init__(self, model_id: str = "fastino/gliner2-base-v1"):
        self._model_id = model_id
        self._extractor = GLiNER2.from_pretrained(self._model_id)

    @property
    def model_id(self) -> str:
        return self._model_id

    def extract(self, text: str, relation_types: list[str]) -> list[Triple]:
        result = self._extractor.extract_relations(text, relation_types)
        
        # Convert GLiNER2 output to list[Triple]
        # GLiNER2 returns: {'relation_extraction': {'rel_type': [(subj, obj), ...]}}
        # or: {'rel_type': [(subj, obj), ...]}
        triples = []
        
        if 'relation_extraction' in result:
            rel_data = result['relation_extraction']
        else:
            rel_data = result
        
        for rel_type, pairs in rel_data.items():
            for pair in pairs:
                if len(pair) >= 2:
                    triples.append(Triple(
                        subject=pair[0],
                        relation=rel_type,
                        object=pair[1]
                    ))
        
        return triples

class GLiNEREntityExtractor(BaseEntityExtractor):
    """GLiNER2-based entity extractor."""

    def __init__(self, model_id: str = "fastino/gliner2-base-v1"):
        self._model_id = model_id
        self._extractor = GLiNER2.from_pretrained(self._model_id)

    @property
    def model_id(self) -> str:
        return self._model_id

    def extract(self, text: str, entity_types: list[str]) -> list[Entity]:
        result = self._extractor.extract_entities(text, entity_types)
        
        # Convert GLiNER2 output to list[Entity]
        # GLiNER2 returns: {'entities': {'type': ['text1', 'text2'], ...}}
        entities = []
        
        if isinstance(result, dict) and 'entities' in result:
            for entity_type, entity_texts in result['entities'].items():
                for ent_text in entity_texts:
                    entities.append(Entity(text=ent_text, type=entity_type))
        
        return entities
