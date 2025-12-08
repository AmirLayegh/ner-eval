from pydantic import BaseModel


# --- Core Data Models ---

class Triple(BaseModel):
    """A single relation triple (subject, relation, object)."""
    subject: str
    relation: str
    object: str


class Entity(BaseModel):
    """A single named entity with text and type."""
    text: str
    type: str


class Sample(BaseModel):
    """A single benchmark sample with text and ground truth."""
    id: str | None = None  # Optional - NER samples don't have IDs
    text: str
    triples: list[Triple] = []   # For RE tasks
    entities: list[Entity] = []  # For NER tasks


class BenchmarkMetadata(BaseModel):
    """Metadata about the benchmark dataset."""
    entity_types: list[str] | None = None
    relation_types: list[str] | None = None
    total_samples: int | None = None
    total_triples: int | None = None
    name: str | None = None
    description: str | None = None


class BenchmarkData(BaseModel):
    """Complete benchmark dataset structure."""
    metadata: BenchmarkMetadata
    samples: list[Sample]


# --- Relation Extraction Results ---

class ReSampleResult(BaseModel):
    """A single relation extraction sample result."""
    id: str
    text: str
    ground_truth: list[Triple]
    predicted: list[Triple]
    time_seconds: float


class REBenchmarkResult(BaseModel):
    """Complete relation extraction benchmark result."""
    model_id: str
    results: list[ReSampleResult]
    average_time_per_sample: float
    total_samples: int
    total_triples: int
    total_time_seconds: float


# --- NER Results ---

class NERSampleResult(BaseModel):
    """A single NER sample result."""
    text: str
    ground_truth: list[Entity]
    predicted: list[Entity]
    time_seconds: float


class NERBenchmarkResult(BaseModel):
    """Complete NER benchmark result."""
    model_id: str
    results: list[NERSampleResult]
    average_time_per_sample: float
    total_samples: int
    total_entities: int
    total_time_seconds: float


# --- Evaluation Metrics ---

class EvaluationMetrics(BaseModel):
    """Evaluation metrics (precision, recall, F1)."""
    model_id: str
    precision: float
    recall: float
    f1: float
    true_positives: int
    false_positives: int
    false_negatives: int