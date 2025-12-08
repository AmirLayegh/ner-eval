import time
from src.extractors.base import BaseRelationExtractor, BaseEntityExtractor
from src.models import REBenchmarkResult, ReSampleResult, NERBenchmarkResult, NERSampleResult
from src.data.loader import BenchmarkDataLoader


class REBenchmarkRunner:
    """Runs relation extraction benchmark on any extractor."""
    
    def __init__(self, data_loader: BenchmarkDataLoader, extractor: BaseRelationExtractor):
        self._data_loader = data_loader
        self._extractor = extractor

    def run(self) -> REBenchmarkResult:
        results = []
        total_time = 0.0
        total_triples = 0
        
        relation_types = self._data_loader.relation_types
        
        for sample in self._data_loader.samples:
            start_time = time.time()
            predicted = self._extractor.extract(sample.text, relation_types)
            elapsed = time.time() - start_time
            
            total_time += elapsed
            total_triples += len(sample.triples)
            
            results.append(ReSampleResult(
                id=sample.id,
                text=sample.text,
                ground_truth=sample.triples,
                predicted=predicted,
                time_seconds=elapsed
            ))
        
        return REBenchmarkResult(
            model_id=self._extractor.model_id,
            results=results,
            total_samples=len(results),
            total_triples=total_triples,
            total_time_seconds=total_time,
            average_time_per_sample=total_time / len(results) if results else 0.0
        )

class NERBenchmarkRunner:
    """Runs named entity recognition benchmark on any extractor."""
    
    def __init__(self, data_loader: BenchmarkDataLoader, extractor: BaseEntityExtractor):
        self._data_loader = data_loader
        self._extractor = extractor

    def run(self) -> NERBenchmarkResult:
        results = []
        total_time = 0.0
        total_entities = 0

        entity_types = self._data_loader.entity_types

        for sample in self._data_loader.samples:
            start_time = time.time()
            predicted = self._extractor.extract(sample.text, entity_types)
            elapsed = time.time() - start_time
            
            total_time += elapsed
            total_entities += len(sample.entities)
            
            results.append(NERSampleResult(
                text=sample.text,
                ground_truth=sample.entities,
                predicted=predicted,
                time_seconds=elapsed
            ))
        
        return NERBenchmarkResult(
            model_id=self._extractor.model_id,
            results=results,
            total_samples=len(results),
            total_entities=total_entities,
            total_time_seconds=total_time,
            average_time_per_sample=total_time / len(results) if results else 0.0
        )
