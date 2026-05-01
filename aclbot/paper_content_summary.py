# Copyright 2026 National Institute of Advanced Industrial Science and Technology (AIST)
# Licensed under the Apache License, Version 2.0

import os
from neo4j import GraphDatabase
from rich.table import Table
from rich.console import Console
import matplotlib.pyplot as plt

# Neo4j connection
NEO4J_URI = os.getenv('NEO4J_URI', 'bolt://localhost:7687')
NEO4J_USER = os.getenv('NEO4J_USERNAME', 'neo4j')
NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD')

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
console = Console()

def fetch_paper_stats(tx):
    query = """
    MATCH (p:Paper)
    OPTIONAL MATCH (p)-[:CONTAINS]->(pas:Passage)
    WITH p.year AS year,
         COUNT(DISTINCT p) AS total_papers,
         COUNT(DISTINCT CASE WHEN p.abstract IS NOT NULL AND size(p.abstract) > 10 THEN p END) AS with_abstract,
         COUNT(DISTINCT CASE WHEN pas IS NOT NULL THEN p END) AS with_passages
    RETURN year, total_papers, with_abstract, with_passages
    ORDER BY year
    """
    return tx.run(query).data()

def print_table(records):
    table = Table(title="📊 Paper Content by Year")

    table.add_column("Year", justify="right")
    table.add_column("Total Papers", justify="right")
    table.add_column("With Abstract", justify="right")
    table.add_column("With Passages", justify="right")

    for row in records:
        table.add_row(
            str(row["year"]),
            str(row["total_papers"]),
            str(row["with_abstract"]),
            str(row["with_passages"])
        )

    console.print(table)

def plot(records):
    years = [row["year"] for row in records if row["year"] is not None]
    total = [row["total_papers"] for row in records if row["year"] is not None]
    abstracts = [row["with_abstract"] for row in records if row["year"] is not None]
    passages = [row["with_passages"] for row in records if row["year"] is not None]

    plt.figure(figsize=(10, 5))
    plt.plot(years, total, label="Total Papers", marker='o')
    plt.plot(years, abstracts, label="With Abstract", marker='x')
    plt.plot(years, passages, label="With Passages", marker='s')
    plt.xlabel("Year")
    plt.ylabel("Paper Count")
    plt.title("ACL Papers: Abstracts & Passages Over Time")
    plt.legend()
    plt.grid(True, axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

# Main
with driver.session() as session:
    records = session.execute_read(fetch_paper_stats)

print_table(records)
plot(records)
