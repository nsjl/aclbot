# Copyright 2026 National Institute of Advanced Industrial Science and Technology (AIST)
# Licensed under the Apache License, Version 2.0

import os
import sys
from neo4j import GraphDatabase
from rich.console import Console
from rich.table import Table

# Parse optional limit argument
LIMIT = int(sys.argv[1]) if len(sys.argv) > 1 else 20

# Neo4j connection setup
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
console = Console()

# Define query templates with dynamic limit injection
CATEGORIES = {
    "Area": f"""
        MATCH (p:Paper)-[:HAS_AREA]->(a:Area)
        RETURN a.name AS name, COUNT(*) AS count
        ORDER BY count DESC LIMIT {LIMIT}
    """,
    "ContributionType": f"""
        MATCH (p:Paper)-[:HAS_CONTRIBUTIONTYPE]->(c:ContributionType)
        RETURN c.name AS name, COUNT(*) AS count
        ORDER BY count DESC LIMIT {LIMIT}
    """,
    "Method": f"""
        MATCH (p:Paper)-[:USES]->(m:Method)
        RETURN m.name AS name, COUNT(*) AS count
        ORDER BY count DESC LIMIT {LIMIT}
    """,
    "Architecture": f"""
        MATCH (p:Paper)-[:USES]->(a:Architecture)
        RETURN a.name AS name, COUNT(*) AS count
        ORDER BY count DESC LIMIT {LIMIT}
    """,
    "PretrainedModel": f"""
        MATCH (p:Paper)-[:USES]->(pm:PretrainedModel)
        RETURN pm.name AS name, COUNT(*) AS count
        ORDER BY count DESC LIMIT {LIMIT}
    """,
    "Task": f"""
        MATCH (p:Paper)-[:WORKS_ON]->(t:Task)
        RETURN t.name AS name, COUNT(*) AS count
        ORDER BY count DESC LIMIT {LIMIT}
    """,
    "Dataset": f"""
        MATCH (p:Paper)-[:WORKS_ON]->(d:Dataset)
        RETURN d.name AS name, COUNT(*) AS count
        ORDER BY count DESC LIMIT {LIMIT}
    """,
    "Metric": f"""
        MATCH (p:Paper)-[:REPORTS]->(:Result)-[:ON]->(m:Metric)
        RETURN m.name AS name, COUNT(*) AS count
        ORDER BY count DESC LIMIT {LIMIT}
    """
}

def fetch_top_entities(tx, query):
    return tx.run(query).data()

# Run queries and display results
with driver.session() as session:
    for category, query in CATEGORIES.items():
        results = session.execute_read(fetch_top_entities, query)

        table = Table(title=f"Top {LIMIT} {category}s")
        table.add_column(category)
        table.add_column("Count", justify="right")

        for row in results:
            name = row["name"] or "<unnamed>"
            table.add_row(name, str(row["count"]))

        console.print(table)
