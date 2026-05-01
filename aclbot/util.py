# Copyright 2026 National Institute of Advanced Industrial Science and Technology (AIST)
# Licensed under the Apache License, Version 2.0

from datetime import datetime
from pathlib import Path

import pandas as pd
from neo4j import Driver, EagerResult

from intertext_graph import Node, IntertextDocument


################################################################################
# OpenAI Cost Tracker
################################################################################


class OpenAICostTracker:
    def __init__(
            self,
            model_name: str,
            cost_file_path: Path = Path('data/openai_cost.csv')
    ):
        if model_name == 'gpt-4o':
            self.cost_per_prompt_token = 2.50 / 1000000
            self.cost_per_completion_token = 10.00 / 1000000

        elif model_name == 'gpt-4o-mini':
            self.cost_per_prompt_token = 0.15 / 1000000
            self.cost_per_completion_token = 0.60 / 1000000

        self._n_prompt_tokens = 0
        self._n_completion_tokens = 0

        self.cost_file_path = cost_file_path

    def track_tokens(
            self,
            n_prompt_tokens: int,
            n_completion_tokens: int
    ):
        self._n_prompt_tokens += n_prompt_tokens
        self._n_completion_tokens += n_completion_tokens

    def compute_current_cost(self):
        return (
            self._n_prompt_tokens * self.cost_per_prompt_token
            + self._n_completion_tokens * self.cost_per_completion_token
        )

    def write_out_cost(self):
        """Open cost file, read accumulated cost and add new row to the file
        with new accumulated cost"""
        current_cost = self.compute_current_cost()

        cost_table = pd.read_csv(self.cost_file_path)
        new_line = cost_table.iloc[-1].copy()

        date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        new_line['date'] = date
        new_line['cost'] = current_cost

        new_line['accumulated_cost'] = new_line['accumulated_cost'] + current_cost

        # Add new line to table and write out
        cost_table = pd.concat([cost_table, new_line.to_frame().T], ignore_index=True)
        cost_table.to_csv(self.cost_file_path, index=False)

    def reset_token_counts(self):
        self._n_prompt_tokens = 0
        self._n_completion_tokens = 0


################################################################################
# ITG Helper functions
################################################################################


def get_section_content(
        section_title_node: Node,
        paper: IntertextDocument
):
    # Get the section content
    # Find all children of the section heading
    section_nodes = []
    for n in paper.unroll_subtree(section_title_node):
        section_nodes.append(n)

    section_content = '\n'.join(
        n.content for n in section_nodes
    )
    return section_content


################################################################################
# Graph helper functions
################################################################################

def format_property(
        property_name: str,
        property_value: str | int | float | bool
) -> str:
    """
    This function formats a property name and value to use it in a cypher query.
     It returns a string as follows:
    '{property_name}: {property_value}'
    """
    if type(property_value) == bool:
        # convert bool to str
        property_value = str(property_value).lower()
    if type(property_value) == str:
        if not property_value.startswith('"'):
            # add quotes
            property_value = f'"{property_value}"'
    return f'{property_name}: {property_value}'


async def check_for_existing_node(
        node_type: str,
        node_property_name: str,
        node_property_value: str,
        driver: Driver,
        delete_existing: bool
) -> EagerResult:
    """
    This function checks if a node of the given type with property node_property_name
    and value node_property_value exists in the graph. If delete_existing is set to True,
    delete the existing node.
    It returns a cypher result object with the found nodes as records.
    """

    # Cypher query to find existing node
    check_node_query = """
    MATCH (n:{node_type} {{ {node_property_name}: $node_property_value }}) 
    RETURN n
    """

    # format query
    check_node_query = check_node_query.format(
        node_type=node_type,
        node_property_name=node_property_name
    )

    # Execute query
    # noinspection PyTypeChecker
    result = driver.execute_query(
        check_node_query,
        node_property_value=node_property_value
    )

    if delete_existing and len(result.records) > 0:
        # delete existing node
        await delete_node(
            node_type=node_type,
            node_property_name=node_property_name,
            node_property_value=node_property_value,
            driver=driver
        )

    return result


async def delete_node(
        node_type: str,
        node_property_name: str,
        node_property_value: str,
        driver: Driver
):
    """
    Deletes a node of the given type with property node_property_name and value
    node_property_value from the graph.
    """
    # Cypher query to delete existing node
    delete_node_query = """
    MATCH (n:{node_type} {{ {node_properties} }}) 
    DELETE n
    """

    node_properties = format_property(
        node_property_name,
        node_property_value
    )

    # format query
    delete_node_query = delete_node_query.format(
        node_type=node_type,
        node_properties=node_properties
    )

    # Execute query
    # noinspection PyTypeChecker
    driver.execute_query(delete_node_query)

    return


async def add_node(
        node_type: str,
        node_properties: dict,
        driver: Driver
):
    """
    Adds a node of type node_type with the given node properties to the graph.
    """
    # Cypher query to add node
    add_node_query = """
    MERGE (n:{node_type} {{ {node_properties} }})
    """

    # format node properties
    node_properties_as_strings = []
    for k, v in node_properties.items():
        node_properties_as_strings.append(
            format_property(k, v)
        )
    node_properties = ', '.join(node_properties_as_strings)

    # format query
    add_node_query = add_node_query.format(
        node_type=node_type,
        node_properties=node_properties,
    )
    # Execute query
    # noinspection PyTypeChecker
    driver.execute_query(add_node_query)

    return


async def delete_relations(
        src_node_type: str,
        src_node_property_name: str,
        src_node_property_value: str,
        tgt_node_type: str | None,
        relation_type: str | None,
        driver: Driver,
) -> None:
    """
    Deletes relations that are outgoing from the node of type src_node_type with
    property src_node_property_name and value src_node_property_value.
    If tgt_node_type is specified, only relations with the given target node type
    are deleted.
    If relation_type is specified, only relations with the given relation type
    are deleted.
    If relation_type and tgt_node_type are not specified, all relations are deleted.
    """

    src_node_properties = format_property(
        src_node_property_name,
        src_node_property_value
    )

    def _delete_relations_query_known_rtype_known_tgt_node_type():
        # Cypher query to delete existing relations
        delete_relation_query = """
            MATCH (s:{src_node_type} {{ {src_node_properties} }})-[r:{relation_type}]->(t:{tgt_node_type}) 
            DELETE r
            """
        # format detelion query
        delete_relation_query = delete_relation_query.format(
            src_node_type=src_node_type,
            src_node_properties=src_node_properties,
            tgt_node_type=tgt_node_type,
            relation_type=relation_type
        )
        return delete_relation_query

    def _delete_relations_query_known_rtype_unknown_tgt_node_type():
        # Cypher query to delete existing relations
        delete_relation_query = """
            MATCH (s:{src_node_type} {{ {src_node_properties} }})-[r:{relation_type}]->(t) 
            DELETE r
            """
        # format deletion query
        delete_relation_query = delete_relation_query.format(
            src_node_type=src_node_type,
            src_node_properties=src_node_properties,
            relation_type=relation_type
        )
        return delete_relation_query

    def _delete_relations_query_unknown_rtype_unknown_tgt_node_type():
        # Cypher query to delete existing relations
        delete_relation_query = """
            MATCH (s:{src_node_type} {{ {src_node_properties} }})-[r]->(t) 
            DELETE r
            """
        # format detelion query
        delete_relation_query = delete_relation_query.format(
            src_node_type=src_node_type,
            src_node_properties=src_node_properties,
        )

        return delete_relation_query

    if tgt_node_type is None:
        if relation_type is None:
            delete_relation_query = _delete_relations_query_unknown_rtype_unknown_tgt_node_type()
        else:
            delete_relation_query = _delete_relations_query_known_rtype_unknown_tgt_node_type()
    else:
        if relation_type is None:
            raise ValueError(
                'relation_type must be specified if tgt_node_type is specified'
            )
        else:
            delete_relation_query = _delete_relations_query_known_rtype_known_tgt_node_type()

    # Execute deletion query
    # noinspection PyTypeChecker
    driver.execute_query(delete_relation_query)

    return


async def check_for_existing_relations(
        src_node_type: str,
        src_node_property_name: str,
        src_node_property_value: str,
        tgt_node_type: str,
        relation_type: str,
        driver: Driver,
        delete_existing: bool
) -> EagerResult:
    """
    Check if there is an existing relation of type relation_type between the
    src_node_type with the src_node_property_name and src_node_property_value and
    any node of type tgt_node_type.
    If delete_existing is set to True, delete the existing relations.
    Returns the found relations.
    """
    # Cypher query to find existing relations
    check_relation_query = """
    MATCH (s:{src_node_type} {{ {src_node_properties} }})-[r:{relation_type}]->(t:{tgt_node_type}) 
    RETURN t AS tgt
    """

    src_node_properties = format_property(
        src_node_property_name,
        src_node_property_value
    )

    # format query
    check_relation_query = check_relation_query.format(
        src_node_type=src_node_type,
        src_node_properties=src_node_properties,
        tgt_node_type=tgt_node_type,
        relation_type=relation_type
    )

    # Execute query
    # noinspection PyTypeChecker
    result = driver.execute_query(check_relation_query)

    if delete_existing and len(result.records) > 0:
        await delete_relations(
            src_node_type=src_node_type,
            src_node_property_name=src_node_property_name,
            src_node_property_value=src_node_property_value,
            tgt_node_type=tgt_node_type,
            relation_type=relation_type,
            driver=driver,
        )

    return result


async def add_relation_without_property_to_graph(
        src_node_type: str,
        src_node_property_name: str,
        src_node_property_value: str,
        tgt_node_type: str,
        tgt_node_property_name: str,
        tgt_node_property_value: str,
        relation_type: str,
        driver: Driver,
) -> None:
    """
    Adds a relation between two nodes of type src_node_type with property
    src_node_property_name and src_node_property_value and type tgt_node_type with
    property tgt_node_property_name and tgt_node_property_value of type relation_type
    to the graph.
    """
    # Cypher query to create relation
    add_relation_query = """
    MATCH (s:{src_node_type} {{ {src_node_properties} }})
    MATCH (t:{tgt_node_type} {{ {tgt_node_properties} }})
    MERGE (s)-[r:{relation_type}]->(t)
    """

    src_node_properties = format_property(
        src_node_property_name,
        src_node_property_value
    )
    tgt_node_properties = format_property(
        tgt_node_property_name,
        tgt_node_property_value
    )

    # format query
    add_relation_query = add_relation_query.format(
        src_node_type=src_node_type,
        src_node_properties=src_node_properties,
        tgt_node_type=tgt_node_type,
        tgt_node_properties=tgt_node_properties,
        relation_type=relation_type
    )

    # Execute query
    # noinspection PyTypeChecker
    driver.execute_query(add_relation_query)

    return


async def add_relation_with_properties_to_graph(
        src_node_type: str,
        src_node_property_name: str,
        src_node_property_value: str,
        tgt_node_type: str,
        tgt_node_property_name: str,
        tgt_node_property_value: str,
        relation_type: str,
        relation_properties: dict,
        driver: Driver,
) -> None:
    """
    Adds a relation between two nodes of type src_node_type with property
    src_node_property_name and src_node_property_value and type tgt_node_type with
    property tgt_node_property_name and tgt_node_property_value of type relation_type
    to the graph.
    The relation is created with the given relation properties.
    """
    # Cypher query
    add_relation_query = """
    MATCH (s:{src_node_type} {{ {src_node_properties} }})
    MATCH (t:{tgt_node_type} {{ {tgt_node_properties} }})
    MERGE (s)-[r:{relation_type} {{ {relation_properties} }}]->(t)
    """

    src_node_properties = format_property(
        src_node_property_name,
        src_node_property_value
    )
    tgt_node_properties = format_property(
        tgt_node_property_name,
        tgt_node_property_value
    )

    relation_properties_as_strings = []
    for k, v in relation_properties.items():
        relation_properties_as_strings.append(
            format_property(k, v)
        )

    relation_properties = ', '.join(relation_properties_as_strings)

    # format query
    add_relation_query = add_relation_query.format(
        src_node_type=src_node_type,
        src_node_properties=src_node_properties,
        tgt_node_type=tgt_node_type,
        tgt_node_properties=tgt_node_properties,
        relation_type=relation_type,
        relation_properties=relation_properties
    )

    # Execute query
    # noinspection PyTypeChecker
    driver.execute_query(add_relation_query)

    return
