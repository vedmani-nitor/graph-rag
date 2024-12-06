~~Use neo4j GraphRag library to write a custom retriever. https://neo4j.com/developer-blog/enhancing-hybrid-retrieval-graphrag-python-package/~~



~~create full text index for Chunk Nodes and implement hybrid search.~~
~~https://neo4j.com/developer-blog/hybrid-retrieval-graphrag-python-package/~~
~~https://neo4j.com/docs/cypher-manual/current/indexes/semantic-indexes/full-text-indexes/~~

~~make sure that only description metadata~~ 

Add FULLTEXT INDEX CREATION to the custom_neo4j_property_graph.py file to index CHUNK nodes text property. 
```CREATE FULLTEXT INDEX text_index FOR (n:Chunk) ON EACH [n.text]``` otherwise the hybrid search will not work.

check which index works better ```CREATE FULLTEXT INDEX text_index FOR (n:__Node__) ON EACH [n.text, n.description]```