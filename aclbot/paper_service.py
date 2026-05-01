# Copyright 2026 National Institute of Advanced Industrial Science and Technology (AIST)
# Licensed under the Apache License, Version 2.0

import re

from neo4j import GraphDatabase
from neo4j_graphrag.embeddings import SentenceTransformerEmbeddings
from neo4j_graphrag.retrievers import VectorCypherRetriever
from neo4j import Record
from neo4j_graphrag.types import RetrieverResultItem
import torch
from graph_schema import Paper, Volume, Passage, Author, Event


class PaperSearchService:
    def __init__(
            self,
            uri: str,
            user: str,
            pwd: str
    ):
        driver = GraphDatabase.driver(uri, auth=(user, pwd))
        self._driver = driver
        self._embedder = SentenceTransformerEmbeddings(
            model='all-MiniLM-L6-v2',device='cuda' if torch.cuda.is_available() else 'cpu'
        )



        # Create LLM object. Used to generate the CYPHER queries
        # self._llm = OpenAILLM(model_name="gpt-4o", model_params={"temperature": 0})

    async def get_paper_by_id(self, paper_id: str) -> Paper | str:
        """
        Search the KG for a paper by ID and return it with passages, authors
        and venue.
        :param paper_id: String ID
        :return: The paper (if found) with passages, authors and venue.
        """

        GET_PAPER_BY_ID_QUERY = """
        MATCH (p:Paper {id: $paper_id})
        RETURN p as paper
        """

        paper_only_records, _, _ = self._driver.execute_query(
            GET_PAPER_BY_ID_QUERY,
            paper_id=paper_id
        )

        if len(paper_only_records) == 0:
            return f'No papers found for id {paper_id}. Are you sure it is correct?'

        paper_node = paper_only_records[0].get('paper')

        GET_PASSAGES_BY_ID_QUERY = """
        MATCH (p:Paper {id: $paper_id})-[:CONTAINS]->(pas:Passage)
        WITH p, collect(pas) as passages
        RETURN p, passages
        """

        paper_and_passages_records, _, _ = self._driver.execute_query(
            GET_PASSAGES_BY_ID_QUERY,
            paper_id=paper_id
        )

        if not paper_and_passages_records:
            passage_list = None
        else:
            passage_list = paper_and_passages_records[0].get('passages')

        paper = await self._get_paper(
            paper_node,
            passage_list,
            format='full_text'
        )

        return paper

    async def semantic_search(self, query: str) -> list[Paper]:
        """
        Embed the query and search among all passages in the KG. Return the
        papers that contain the resulting passages with abstracts and the resulting
        passages only.
        :param text:
        :return:
        """

        PASSAGE_TO_PAPER_TRAVERSAL_QUERY = """
        MATCH (p:Paper)-[:CONTAINS]->(node)
        RETURN p as paper, node as passage
        """

        def semantic_search_record_formatter(
                record: Record
        ) -> RetrieverResultItem:
            result_dict = {}
            result_dict['paper'] = record.get('paper')
            result_dict['passage'] = record.get('passage')
            return RetrieverResultItem(content=result_dict)

        # Set up retriever
        retriever = VectorCypherRetriever(
            driver=self._driver,
            index_name='passage_embedding',
            embedder=self._embedder,
            retrieval_query=PASSAGE_TO_PAPER_TRAVERSAL_QUERY,
            result_formatter=semantic_search_record_formatter
        )

        # Execute query
        retriever_result = retriever.search(
            query_text=query,
            top_k=3
        )

        # Aggregate search results: If the same paper occurs multiple times,
        # aggregate its passages into a single list per paper
        paper_nodes_with_result_passages = {}
        rank_order = []
        for item in retriever_result.items:
            paper_node = item.content['paper']
            paper_id = paper_node['id']
            passage_node = item.content['passage']
            if paper_id not in paper_nodes_with_result_passages:
                paper_nodes_with_result_passages[paper_id] = {
                    'paper_node': paper_node,
                    'passages': [passage_node]
                }
                rank_order.append(paper_id)
            else:
                paper_nodes_with_result_passages[paper_id]['passages'].append(passage_node)

        # Construct Paper objects
        results = []
        for entry in paper_nodes_with_result_passages:
            paper_node = paper_nodes_with_result_passages[entry]['paper_node']
            passage_list = paper_nodes_with_result_passages[entry]['passages']
            paper = await self._get_paper(
                paper_node,
                passage_list=passage_list,
                format='full_text'
            )
            results.append(paper)

        return results
    # TODO think about shortening authors when there are many?

    async def event_search(
            self,
            query: str
    ):
        # Find events
        event_query = """
        MATCH (e:Event) WHERE e.name CONTAINS $query RETURN COLLECT(e) as events
        """

        event_query_results = self._driver.execute_query(
            event_query,
            query=query
        )

        events = [
            Event(
                id=event_node.get('id'),
                name=event_node.get('name'),
                year=event_node.get('year')
            ) for event_node in event_query_results.records[0].get('events')
        ]

        existing_event_ids = [event['id'] for event in events]

        volume_query = """
        MATCH(v:Volume)-[:BELONGS_TO]->(e:Event) WHERE v.name CONTAINS $query RETURN COLLECT(e) as events
        """

        # noinspection PyTypeChecker
        volume_query_results = self._driver.execute_query(
            volume_query,
            query=query
        )

        for event_node in volume_query_results.records[0].get('events'):
            if event_node.get('id') not in existing_event_ids:
                events.append(Event(
                    id=event_node.get('id'),
                    name=event_node.get('name'),
                    year=event_node.get('year')
                ))

        return events


    async def run_cypher_query(
            self,
            cypher_query
    ):
        if self._is_write_operation(cypher_query):
            return "Writing operations are not allowed."
        records, _, _ = self._driver.execute_query(cypher_query)

        if len(records) == 0:
            return (
                'Your query did not produce any results. Please check your query. '
                'If it is not correct, please call this function again with the '
                'correct query. '
            )

        return records

    @staticmethod
    def get_system_message() -> str:
        SYSTEM_MESSAGE = """
        You are ACLBot, a helpful assistant that can answer questions about 
        papers in the ACL anthology. 
        You have access to a knowledge graph of papers from the ACL anthology 
        running on a Neo4j instance. The knowledge graph contains paper full texts,
        information on the publication metadata, the type of research done, 
        research entities (e.g. methods or datasets) and results reported in the 
        papers.
        If the user asks for data, always include a Cypher query in your reply.
        Output the query inside a code block using triple backticks and the `cypher` tag.
        . Do not say you can't access a database. You DO have access to the knowledge graph.
        
        ## Graph Architecture
        
        There are four types of nodes in the graph: 
        1. Nodes that represent the publication content and metadata (Paper, Volume, Event, Author, Passage)
        2. Nodes that represent the type of research done in a paper (Area, ContributionType)
        3. Nodes that represent research entities (Method, Architecture, PretrainedModel, Task, Dataset, Metric) 
        4. Result nodes
        
        Papers are connected to the Volume they appear in (e.g. NAACL 2022 main 
        papers), their Authors and the Passages they contain. Volumes are
        connected to the event they belong to (e.g. NAACL 2022 main papers belong 
        to NAACL 2022). The type of research performed in a paper is represented
        by , papers are connected to the research Area 
        they are in (can be only one) and the types of contributions they make 
        (can be multiple). 
        
        Papers are also connected to the research entities that they work on or 
        are used. The relations between papers and Methods, Architectures and 
        PretrainedModels also specify if they are used as baselines and / or in 
        the proposed model. 
        
        Finally, papers are connected to Results. Each Result is connected to 
        one Task, Dataset and Metric. The Result specifies the value of the score
        that was obtained on the given Metric.

        Example response:

        """

        return SYSTEM_MESSAGE

    @staticmethod
    async def get_kg_instructions() -> str:
        KG_INSTRUCTIONS = """        
        ## Nodes
        
        ### Publication Metadata Nodes
        
        - Paper: {id: STRING, title: STRING, year: int, abstract: STRING}
        - Volume: {id: STRING, name: STRING, year: int, type: STRING}, e.g. {id: "2022.naacl-main", name: "Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies"}
        - Event: {id: STRING, name: STRING, year: int}, e.g. {id: "naacl-2022", name: "NAACL 2022"} 
        - Author: {id: STRING, name: str}
        - Passage: {id: STRING, content: STRING, type: STRING, embedding: LIST[FLOAT]}, common available types: "document-title", "section-title", "paragraph"
        
        ### Type of Research
        
        - Area: {name: STRING}, e.g. {name: "Machine Translation"}. Fixed set of 20 available Areas.
        - ContributionType: {name: STRING} e.g. {name: "NLP engineering experiment"}. Fixed set of 13 available ContributionTypes.
        
        ### Research Entity Nodes
        
        - Method {name: STRING}, e.g. {name: "Expectation-Maximization"}
        - Architecture {name: STRING}, e.g. {name: "Transformer}
        - PretrainedModel {name: STRING}, e.g. {name: "BERT"}
        - Task {name: STRING}, e.g. {name: "Question Answering"}
        - Dataset {name: STRING}, e.g. {name: "Natural Questions"}
        - Metric {name: STRING}, e.g. {name: "F1-Score"}
        
        ### Result Node
        
        - Result {id: STRING, value: FLOAT}
        
        ## Relations
        
        ### Publication Process Relations
        
        - (:Paper)-[:AUTHORED_BY]->(:Author)
        - (:Paper)-[:PUBLISHED_IN]->(:Volume)
        - (:Volume)-[:BELONGS_TO]->(:Event)
        - (:Paper)-[:CONTAINS]->(:Passage)
        - (:Passage)-[:IS_PARENT_OF]->(:Passage) // e.g. (section_title:Passage {type: "section-title"})-[:IS_PARENT_OF]->(paragraph:Passage {type: "paragraph"})
        - (:Passage)-[:IS_FOLLOWED_BY]->(:Passage) // e.g. (paragraph_3:Passage)-[:IS_FOLLOWED_BY]->(paragraph_4:Passage)
        
        ### Type of Research Relations
        
        - (:Paper)-[:HAS_AREA)->(:Area), only one per paper.
        - (:Paper)-[:HAS_CONTRIBUTIONTYPE]->(:ContributionType), can be multiple
        
        ### Research Relations
        
        - (:Paper)-[:USES {as_proposed: BOOL, as_baseline: BOOL}->{:Method/Architecture/PretrainedModel}
        - (:Paper)-[:WORKS_ON]->{:Task/Dataset/Metric}
        
        ### Result Relations
        
        - (:Paper)-[:REPORTS]->(:Result)
        - (:Result)-[:ON]->(:Task/Dataset/Metric)
        
        ## Allowed Cypher Queries
        You can make Cypher queries on the graph, observing the following rules:
        - Any query that writes to the graph is not allowed. 
        
        ## Function Calling Guidance

        Ggenerate Cypher queries that answer the user's question.


        You must return the query as part of the response like this:

        {
        "cypher_query": "MATCH (...) RETURN ..."
        }
        
        For example:

        {
        "cypher_query": "MATCH (p:Paper)-[:HAS_RESULT]->(r:Result)-[:USES_METRIC]->(m:Metric {name: 'Accuracy'}) WHERE p.year = 2022 RETURN COUNT(DISTINCT p) AS count"
        }


        
        """
        return KG_INSTRUCTIONS
        
#         Please ensure all Cypher queries returned are single valid statements. If multiple results are needed, use UNION or UNION ALL to combine them. Do not include multiple separate MATCH ... RETURN blocks



    ############################################################################
    # The functions below are not accessible to the PaperSearchPlugin directly
    ############################################################################

    @staticmethod
    def get_initial_assistant_message():
        INITIAL_ASSISTANT_MESSAGE = (
            'Hi there, I am ACLBot, your assistant for literature research '
            'in the ACL Anthology. What would you like to know?'
        )
        return INITIAL_ASSISTANT_MESSAGE

    async def _get_paper(
            self,
            paper_node,
            passage_list=None,
            format='abstract_only' # 'title_only', 'abstract_only', 'full_text'
    ) -> Paper:
        """
        Construct a Paper object from Cypher query output. Depending on the
        length requirements, the passages and / or the abstract are empty.
        :param paper_node:
        :param author_list:
        :param passage_list:
        :param venue_node:
        :param format:
        :return:
        """

        paper = Paper(
            id=paper_node.get('id'),
            title=paper_node.get('title'),
            year=paper_node.get('year'),
            abstract=None,
            passages=None
        )

        if format == 'title_only':
            return paper

        paper['abstract'] = paper_node.get('abstract')

        if format == 'abstract_only':
            return paper

        if passage_list is None:
            passages = [
                Passage(
                    id='dummy_id_1',
                    content=f'No passages available for {paper["id"]}',
                    type='dummy-type'
                )
            ]
        else:
            passages = [
                Passage(
                    id=p.get('id'),
                    content=p.get('content'),
                    type=p.get('type')
                )
                for p in passage_list
            ]
        passages = self._sort_passages_for_single_paper(passages)

        paper['passages'] = passages

        return paper

    @staticmethod
    def _sort_passages_for_single_paper(
            passages: list[Passage]
    ) -> list[Passage]:
        """
        Sort the given passages according to their sequential order in the paper.
        It is assumed that the passages come from a single paper.
        :param passages: The list of passages to sort
        :return: The sorted list of passages
        """
        # TODO test
        passages = sorted(
            passages,
            key=lambda passage: int(passage['id'].split('_')[-1]))
        return passages

    @staticmethod
    def _is_write_operation(query: str) -> bool:
        """
        Check if a Cypher query performs writing operations.

        :param query: The Cypher query string.
        :return: True if the query performs writing operations, False otherwise.
        """
        write_keywords = [
            r'\bCREATE\b', r'\bMERGE\b', r'\bSET\b', r'\bDELETE\b', r'\bREMOVE\b', r'\bDETACH DELETE\b'
        ]

        # Combine all keywords into a single regex pattern
        pattern = re.compile('|'.join(write_keywords), re.IGNORECASE)

        # Search for any of the write keywords in the query
        return bool(pattern.search(query))
