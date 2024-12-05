import neo4j

driver = neo4j.GraphDatabase.driver("neo4j://localhost:7687", auth=("neo4j", "12345678"))

driver.verify_connectivity()
