# Copyright 2026 National Institute of Advanced Industrial Science and Technology (AIST)
# Licensed under the Apache License, Version 2.0

from typing import Annotated, Any
from graph_schema import Paper, Volume, Passage, Author, Event
from semantic_kernel.functions import kernel_function
from paper_service import PaperSearchService


class PaperPlugin:
    def __init__(self, paper_search_service: PaperSearchService):
        self.paper_search_service = paper_search_service
        self._get_kg_instructions_called = False

    @kernel_function
    async def get_paper_by_id(
            self,
            paper_id: str
    ) -> Annotated[Paper, "A paper from the ACL anthology"]:
        """Gets a full paper from the ACL anthology with the given id."""
        return await self.paper_search_service.get_paper_by_id(paper_id)

    @kernel_function
    async def semantic_paper_search(
            self,
            query: str
    ) -> Annotated[list[Paper], "The papers that contain the found passages"]:
        """Do a full text search among all passages in the database. Returns the
        top 3 passages with the corresponding paper metadata."""
        return await self.paper_search_service.semantic_search(query)

    @kernel_function
    async def event_search(
            self,
            query: str
    ) -> Annotated[list[Event], "A list of events found for the query"]:
        """When a user asks for papers from a particular event, first find the
        event id with this function. Then find papers for an event with a query
        similar to:
        `MATCH (p:Paper)-[:PUBLISHED_IN]->(:Volume)-[:BELONGS_TO]->(e:Event {id: 'event_id'})"""
        return await self.paper_search_service.event_search(query)

    @kernel_function
    async def run_cypher_query(
            self,
            cypher_query: str
    ) -> Annotated[Any, "The result of the cypher query"]:
        """Given a cypher query, execute it over the knowledge graph and return the result."""
        if not self._get_kg_instructions_called:
            instructions = await self.get_kg_instructions()
            self._get_kg_instructions_called = True
            return (
                'You did not call get_kg_instructions() yet, so you cannot pass '
                'a self-generated query to the graph. Read these instructions '
                'and then rewrite your query and call run_cypher_query again.\n'
                'Instructions\n'
                f'{instructions}'
            )

        print("Running Cypher query:", cypher_query)
        return await self.paper_search_service.run_cypher_query(cypher_query)

    @kernel_function
    async def get_system_message(
            self
    ) -> Annotated[str, "Basic information about available information. Has been sent in the beginning of the interaction"]:
        return self.paper_search_service.get_system_message()

    @kernel_function
    async def get_available_entities(self) -> str:
        return (
            "To get a list of available areas, contribution types, or entities, "
            "you need to query the graph."
        )

    @kernel_function
    async def get_kg_instructions(
            self
    ) -> Annotated[str, "Information on Nodes and Edges and instructions for writing Cypher queries"]:
        self._get_kg_instructions_called = True
        return await self.paper_search_service.get_kg_instructions()

    def reset(self):
        self._get_kg_instructions_called = False


# Optional: Only for local test/debugging
if __name__ == '__main__':
    import os
    import asyncio

    NEO4J_URI = os.getenv('NEO4J_URI', 'bolt://localhost:7687')
    NEO4J_USER = os.getenv('NEO4J_USERNAME', 'neo4j')
    NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD')

    paper_search_service = PaperSearchService(
        uri=NEO4J_URI,
        user=NEO4J_USER,
        pwd=NEO4J_PASSWORD
    )
    paper_search_plugin = PaperPlugin(paper_search_service)

    # Example tests
    async def test():
        ID = '2022.acl-long.1'
        paper = await paper_search_plugin.get_paper_by_id(ID)
        print(paper)

        QUERY = 'input-adaptive techniques compress the model from the depth perspective'
        result = await paper_search_plugin.semantic_paper_search(QUERY)
        print(result)

    asyncio.run(test())
