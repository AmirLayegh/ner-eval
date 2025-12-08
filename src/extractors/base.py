from abc import ABC, abstractmethod
from src.models import Triple, Entity


class BaseRelationExtractor(ABC):
    """Abstract base class for relation extraction models.
    
    All relation extractors must implement the `extract` method.
    Extractors should ONLY do inference - no data loading, no file I/O.
    """
    
    @property
    @abstractmethod
    def model_id(self) -> str:
        """Return the model identifier (e.g., 'gliner2-base-v1', 'gpt-4o')."""
        pass
    
    @abstractmethod
    def extract(self, text: str, relation_types: list[str]) -> list[Triple]:
        """Extract relation triples from text.
        
        Args:
            text: The input text to extract relations from.
            relation_types: List of relation types to extract.
            
        Returns:
            List of Triple objects representing (subject, relation, object).
        """
        pass

class BaseEntityExtractor(ABC):
    """Abstract base class for named entity recognition models.
    
    All NER extractors must implement the `extract` method.
    """
    
    @property
    @abstractmethod
    def model_id(self) -> str:
        """Return the model identifier."""
        pass
    
    @abstractmethod
    def extract(self, text: str, entity_types: list[str]) -> list[Entity]:
        """Extract named entities from text.
        
        Args:
            text: The input text to extract entities from.
            entity_types: List of entity types to extract.
            
        Returns:
            List of Entity objects with text and type.
        """
        pass
