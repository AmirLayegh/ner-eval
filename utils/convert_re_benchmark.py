"""
Convert the scientist relation extraction dataset to a benchmark JSON format.
Filters only entity-to-entity relations and normalizes underscore notation.
"""
import json

# Entity-to-Entity relations only (no literals like dates, numbers, etc.)
ENTITY_TO_ENTITY_RELATIONS = [
    "almaMater", "award", "birthPlace", "capital", "citizenship", "country", 
    "currency", "deathPlace", "demonym", "governmentType", "isPartOf", 
    "knownFor", "language", "leader", "nationality", "officialLanguage",
    "professionalField", "religion", "residence", "timeZone"
]


def normalize_entity(entity: str) -> str:
    """Convert underscore notation to spaces and clean up."""
    # Remove surrounding quotes if present
    if entity.startswith('"') and entity.endswith('"'):
        entity = entity[1:-1]
    # Replace underscores with spaces
    entity = entity.replace("_", " ")
    # Handle special cases like "Leningrad, USSR" -> "Leningrad, USSR"
    entity = entity.replace(", ", ", ")
    return entity.strip()


def convert_dataset(input_path: str, output_path: str):
    samples = []
    skipped_samples = 0
    total_triples = 0
    filtered_triples = 0
    
    with open(input_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            data = json.loads(line)
            
            # Filter triples to only entity-to-entity relations
            filtered = []
            for triple in data['triples']:
                total_triples += 1
                if triple['rel'] in ENTITY_TO_ENTITY_RELATIONS:
                    filtered.append({
                        "subject": normalize_entity(triple['sub']),
                        "relation": triple['rel'],
                        "object": normalize_entity(triple['obj'])
                    })
                    filtered_triples += 1
            
            # Only include samples that have at least one entity-to-entity relation
            if filtered:
                samples.append({
                    "id": data['id'],
                    "text": data['sent'],
                    "triples": filtered
                })
            else:
                skipped_samples += 1
    
    # Create benchmark structure
    benchmark = {
        "metadata": {
            "name": "Scientist Relation Extraction Benchmark",
            "description": "Entity-to-entity relation extraction dataset filtered from ont_18_scientist_train",
            "relation_types": sorted(ENTITY_TO_ENTITY_RELATIONS),
            "total_samples": len(samples),
            "total_triples": filtered_triples
        },
        "samples": samples
    }
    
    with open(output_path, 'w') as f:
        json.dump(benchmark, f, indent=2, ensure_ascii=False)
    
    print(f"Conversion complete!")
    print(f"  Input triples: {total_triples}")
    print(f"  Filtered triples (entity-to-entity): {filtered_triples}")
    print(f"  Skipped samples (no entity relations): {skipped_samples}")
    print(f"  Output samples: {len(samples)}")
    print(f"  Output file: {output_path}")


if __name__ == "__main__":
    convert_dataset(
        "data/re/ont_18_scientist_train.jsonl",
        "data/re/scientist_re_benchmark.json"
    )

