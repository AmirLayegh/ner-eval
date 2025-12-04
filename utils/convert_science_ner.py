"""Convert BIO-format science_ner.txt to JSON NER format."""

import json

samples = []
current_tokens = []
current_entities = []
current_entity = None
entity_types = set()

with open("data/science_ner.txt") as f:
    for line in f:
        line = line.strip()
        
        if not line:  # Empty line = end of sentence
            if current_tokens:
                text = " ".join(current_tokens)
                # Fix spacing around punctuation
                for p in [".", ",", ")", ":", ";", "?"]:
                    text = text.replace(f" {p}", p)
                text = text.replace("( ", "(")
                text = text.replace("' s", "'s")
                
                if current_entity:
                    current_entities.append(current_entity)
                    current_entity = None
                
                samples.append({"text": text, "entities": current_entities})
                current_tokens = []
                current_entities = []
            continue
        
        parts = line.split("\t")
        if len(parts) != 2:
            continue
            
        token, tag = parts
        current_tokens.append(token)
        
        if tag.startswith("B-"):
            if current_entity:
                current_entities.append(current_entity)
            etype = tag[2:]
            entity_types.add(etype)
            current_entity = {"text": token, "type": etype}
        elif tag.startswith("I-") and current_entity:
            current_entity["text"] += " " + token
        else:
            if current_entity:
                current_entities.append(current_entity)
                current_entity = None

# Handle last sentence
if current_tokens:
    text = " ".join(current_tokens)
    if current_entity:
        current_entities.append(current_entity)
    samples.append({"text": text, "entities": current_entities})

output = {
    "metadata": {
        "name": "Science NER Benchmark",
        "entity_types": sorted(entity_types)
    },
    "samples": [s for s in samples if s["entities"]]  # Only keep samples with entities
}

with open("data/science_ner_benchmark.json", "w") as f:
    json.dump(output, f, indent=2)

print(f"Converted {len(output['samples'])} samples")
print(f"Entity types: {sorted(entity_types)}")

