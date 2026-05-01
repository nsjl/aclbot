WITH $meta AS meta, $abstract as abstract

// Create paper node
MERGE (paper:Paper {id: meta.id})
ON CREATE SET
  paper.title = meta.title,
  paper.year = meta.year,
  paper.abstract = abstract

// Venue
// TODO: IDs for venues
MERGE (venue:Venue {name: meta.venue})
ON CREATE SET
  venue.short_name = meta.venue_short,
  venue.year = meta.year
MERGE (paper)-[pa:PUBLISHED_AT]->(venue)

// Authors
// TODO: IDs for Authors (For now we assume that all authors with the same name
//  are the same person)
WITH meta, paper
FOREACH (author IN meta.authors |
  MERGE (a:Author {name: author})
  MERGE (paper)-[ab:AUTHORED_BY]->(a)
)

// Passages: First create all passages and make connections to paper
// TODO Maybe create nodes for different section types?
WITH $nodes AS nodes, paper
FOREACH (node_ IN nodes |
  MERGE (pas:Passage {id: node_.ix})
  ON CREATE SET
    pas.content = node_.content,
    pas.type = node_.ntype,
    pas.embedding = node_.embedding
  MERGE (paper)-[co:CONTAINS]->(pas)
)

// Create parent edges between passages
WITH $parent_edges AS parent_edges
FOREACH (edge IN parent_edges |
  MERGE (src:Passage {id:edge.src})
  MERGE (tgt:Passage {id:edge.tgt})
  MERGE (src)-[ifb:IS_PARENT_OF]->(tgt)
)

// Create next edges between passages
WITH $next_edges as next_edges
FOREACH (edge IN next_edges |
  MERGE (src:Passage {id:edge.src})
  MERGE (tgt:Passage {id:edge.tgt})
  MERGE (src)-[ifb:IS_FOLLOWED_BY]->(tgt)
)