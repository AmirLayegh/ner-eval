import langextract as lx
from src.extractors.base import BaseEntityExtractor
from src.models import Entity
import textwrap

class LangExtractEntityExtractor(BaseEntityExtractor):
    """LangExtractor-based entity extractor."""
    
    def __init__(self, model_id: str = "gpt-4o"):
        self._model_id = model_id
        self._extractor = lx
        self._prompt = None
        self._examples = None

    @property
    def model_id(self) -> str:
        return self._model_id
    
    def _set_prompt(self, entity_types: list[str]):
        self._prompt = textwrap.dedent(f"""
        Extract the entities with the types of the following: {", ".join(entity_types)}.
        Use exact text for extraction. Do not paraphrase or overlap entities.
        """)

    def _set_examples(self):
        self._examples = [
        lx.data.ExampleData(
            text="They may also use Adenosine triphosphate, Nitric oxide, and ROS for signaling in the same ways that animals do.",
            extractions=[
                lx.data.Extraction(
                    extraction_class="chemicalcompound",
                    extraction_text="Adenosine triphosphate",
                ),
                lx.data.Extraction(
                    extraction_class="chemicalcompound",
                    extraction_text="Nitric oxide",
                ),
                lx.data.Extraction(
                    extraction_class="chemicalcompound",
                    extraction_text="ROS",
                ),
            ]
        ),
        lx.data.ExampleData(
            text="August Kopff, a colleague of Wolf at Heidelberg, then discovered 617 Patroclus eight months after Achilles, and, in early 1907, he discovered the largest of all Jupiter trojans, 624 Hektor.",
            extractions=[
                lx.data.Extraction(
                    extraction_class="scientist",
                    extraction_text="August Kopff",
                ),
                lx.data.Extraction(
                    extraction_class="scientist",
                    extraction_text="Wolf",
                ),
                lx.data.Extraction(
                    extraction_class="location",
                    extraction_text="Heidelberg",
                ),
                lx.data.Extraction(
                    extraction_class="astronomicalobject",
                    extraction_text="617 Patroclus",
                ),
                lx.data.Extraction(
                    extraction_class="astronomicalobject",
                    extraction_text="Achilles",
                ),
                lx.data.Extraction(
                    extraction_class="misc",
                    extraction_text="Jupiter trojans",
                ),
                lx.data.Extraction(
                    extraction_class="astronomicalobject",
                    extraction_text="624 Hektor",
                ),
            ]
        ),
    ]

    def extract(self, text: str, entity_types: list[str]) -> list[Entity]:
        results = self._extractor.extract(text, self._prompt, examples=self._examples, model_id=self._model_id, fence_output=True)
        entities = []
        for extraction in results.extractions:
            entities.append(Entity(text=extraction.extraction_text, type=extraction.extraction_class))
        return entities
