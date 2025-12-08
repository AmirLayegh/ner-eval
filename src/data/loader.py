import json
from pydantic import BaseModel
from src.models import BenchmarkData, Sample, Triple, BenchmarkMetadata

class BenchmarkDataLoader:
    """Loads and validates benchmark data using Pydantic models."""
    
    def __init__(self, file_path: str):
        self.file_path = file_path
        self._data: BenchmarkData = self._load_data()
    
    def _load_data(self) -> BenchmarkData:
        with open(self.file_path, 'r') as f:
            raw_data = json.load(f)
        return BenchmarkData.model_validate(raw_data)
    
    @property
    def samples(self) -> list[Sample]:
        return self._data.samples
    
    @property
    def entity_types(self) -> list[str] | None:
        return self._data.metadata.entity_types
    
    @property
    def relation_types(self) -> list[str] | None:
        return self._data.metadata.relation_types
    
    def __len__(self) -> int:
        return len(self._data.samples)


class NERBenchmarkDataLoader(BenchmarkDataLoader):
    """Loads and validates NER benchmark data using Pydantic models."""
    
    def __init__(self, file_path: str):
        super().__init__(file_path)
    
    @property
    def samples(self) -> list[Sample]:
        return self._data.samples
    
    def __len__(self) -> int:
        return len(self._data.samples)