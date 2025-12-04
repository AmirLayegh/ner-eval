from gliner2_test import GliNER2Extractor
from langextract_test import LangExtractor
from neo4j_graphrag_test import Neo4jGraphRagExtractor
gliner2_extractor = GliNER2Extractor()
langextract_extractor = LangExtractor()
neo4j_graphrag_extractor = Neo4jGraphRagExtractor()
gliner2_results, gliner2_time = gliner2_extractor.run()
langextract_results, langextract_time = langextract_extractor.run()
neo4j_graphrag_results, neo4j_graphrag_time = neo4j_graphrag_extractor.run()
print(f"GLiNER2 time: {gliner2_time} seconds")
print(f"LangExtract time: {langextract_time} seconds")
print(f"Neo4j GraphRag time: {neo4j_graphrag_time} seconds")
print(f"GLiNER2 results: {gliner2_results}")
print(f"LangExtract results: {langextract_results}")
print(f"Neo4j GraphRag results: {neo4j_graphrag_results}")