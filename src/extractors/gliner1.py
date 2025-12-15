from gliner import GLiNER
from src.extractors.base import BaseRelationExtractor, BaseEntityExtractor
from src.models import Triple, Entity


class GLiNER1RelationExtractor(BaseRelationExtractor):
    """GLiNER1-based relation extractor."""
    
    def __init__(self, model_id: str = "knowledgator/gliner-relex-large-v0.5"):
        self._model_id = model_id
        self._extractor = GLiNER.from_pretrained(self._model_id)

    @property
    def model_id(self) -> str:
        return self._model_id

    def extract(self, text: str, relation_types: list[str]) -> list[Triple]:
        _, relations = self._extractor.inference(
            texts=[text],
            labels=["entity"],
            relations=relation_types,
            threshold=0.5,
            adjacency_threshold=0.55,
            relation_threshold=0.8,
            return_relations=True,
            flat_ner=False
        )
        
        triples = []
        for relation in relations[0]:
            triples.append(Triple(
                subject=relation['head']['text'],
                relation=relation['relation'],
                object=relation['tail']['text']
            ))
        
        return triples
    
class GLiNER1EntityExtractor(BaseEntityExtractor):
    """GLiNER-based entity extractor."""

    def __init__(self, model_id: str = "urchade/gliner_medium-v2.1", threshold: float = 0.5):
        self._model_id = model_id
        self._threshold = threshold
        self._extractor = GLiNER.from_pretrained(self._model_id)

    @property
    def model_id(self) -> str:
        return self._model_id

    def extract(self, text: str, entity_types: list[str]) -> list[Entity]:
        result = self._extractor.predict_entities(text, entity_types, threshold=self._threshold)
        
        # Convert GLiNER output to list[Entity]
        # GLiNER returns: [{'text': 'entity_text', 'label': 'entity_type', 'start': int, 'end': int}, ...]
        entities = []
        
        for entity in result:
            entities.append(Entity(text=entity["text"], type=entity["label"]))
        
        return entities