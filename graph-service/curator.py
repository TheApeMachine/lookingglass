import os
import time
import logging
import json
import asyncio
import schedule
from typing import List, Dict, Optional, Any
from pydantic import BaseModel, ValidationError
from agents import Agent, Runner, OpenAIChatCompletionsModel, set_tracing_disabled
from openai import AsyncOpenAI
from neo4j import GraphDatabase

# --- Logging & Config ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("curator")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "lm-studio")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "http://localhost:1234/v1")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "openai/gpt-oss-20b")

# Disable tracing since we're using a local model without OpenAI API key
set_tracing_disabled(True)

openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)
logger.info(f"Using OpenAI-compatible endpoint: {OPENAI_BASE_URL} with model: {OPENAI_MODEL}")

NEO4J_HOST = os.getenv("NEO4J_HOST", "neo4j")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")
NEO4J_URI = f"bolt://{NEO4J_HOST}:7687"

# --- Curator agent schema ---
class ConsolidationOpportunity(BaseModel):
    type: str  # "duplicate_relationship", "similar_nodes", "relationship_synonym", etc.
    description: str
    entities: List[str]  # Entity IDs or names involved
    relationships: Optional[List[Dict[str, Any]]] = None  # Relationship details if applicable (flexible to handle synonyms)
    action: str  # What action to take: "merge", "consolidate", "remove", etc.
    confidence: float  # 0.0 to 1.0

class CuratorAnalysis(BaseModel):
    opportunities: List[ConsolidationOpportunity]
    summary: str

curator_agent = Agent(
    name="Graph Curator",
    instructions=("""
        You are analyzing a knowledge graph to find consolidation opportunities. The data provided includes:
        - duplicate_relationships: Relationships that appear multiple times between the same nodes
        - relationship_synonyms: Same node pairs connected by different relationship types that likely mean the same thing
        - orphaned_nodes: Nodes with no connections
        
        IMPORTANT: You MUST return opportunities for ALL items in duplicate_relationships and relationship_synonyms arrays.
        These are pre-identified issues - your job is to format them correctly, not to filter them.
        
        For duplicate_relationships: Each entry represents multiple relationships of the same type between the same nodes.
        Create an opportunity with:
        - type: "duplicate_relationship"
        - entities: [from_id, to_id] from the duplicate_relationships entry
        - relationships: [{"type": rel_type, "from": from_id, "to": to_id}]
        - action: "consolidate"
        - confidence: 0.95 (these are confirmed duplicates)
        
        For relationship_synonyms: Each entry shows the same node pair connected by multiple relationship types.
        Create an opportunity with:
        - type: "relationship_synonym"
        - entities: [from_id, to_id] from the relationship_synonyms entry
        - relationships: [{"from": from_id, "to": to_id, "types": rel_types array, "canonical": choose the most common/preferred type}]
        - action: "consolidate"
        - confidence: 0.85-0.95 (based on how similar the types are)
        
        Return exactly a JSON object:
        {
          "opportunities": [
            {
              "type": "duplicate_relationship",
              "description": "Multiple relationships of type X between nodes A and B",
              "entities": ["entity_id_1", "entity_id_2"],
              "relationships": [{"type": "relationship_type", "from": "entity_id_1", "to": "entity_id_2"}],
              "action": "consolidate",
              "confidence": 0.95
            }
          ],
          "summary": "Found N duplicate relationships and M relationship synonyms"
        }
        
        Process ALL entries in duplicate_relationships and relationship_synonyms. No extra fields, comments, or markdown formatting.
    """),
    model=OpenAIChatCompletionsModel(
        model=OPENAI_MODEL,
        openai_client=openai_client
    ),
)

# --- Helper functions ---
def get_graph_statistics(driver) -> Dict[str, Any]:
    """Get graph-wide statistics and targeted samples for analysis."""
    with driver.session() as sess:
        # Get overall statistics
        total_nodes_query = "MATCH (n) RETURN count(n) AS count"
        total_nodes = sess.run(total_nodes_query).single()["count"]
        
        total_rels_query = "MATCH ()-[r]->() RETURN count(r) AS count"
        total_rels = sess.run(total_rels_query).single()["count"]
        
        # Get label distribution
        label_stats_query = """
        MATCH (n)
        WITH labels(n) AS labels, count(*) AS count
        RETURN labels, count
        ORDER BY count DESC
        LIMIT 30
        """
        label_counts = [dict(record) for record in sess.run(label_stats_query)]
        
        # Get relationship type distribution
        rel_stats_query = """
        MATCH ()-[r]->()
        WITH type(r) AS rel_type, count(*) AS count
        RETURN rel_type, count
        ORDER BY count DESC
        LIMIT 30
        """
        rel_type_counts = [dict(record) for record in sess.run(rel_stats_query)]
        
        # Get nodes with most relationships (likely to have duplicates)
        high_degree_nodes_query = """
        MATCH (n)
        WHERE n.name IS NOT NULL
        WITH n, COUNT { (n)--() } AS degree
        ORDER BY degree DESC
        LIMIT 50
        RETURN n.id AS id, n.name AS name, labels(n) AS labels, 
               properties(n) AS properties, degree
        """
        high_degree_nodes = [dict(record) for record in sess.run(high_degree_nodes_query)]
        
        # Get relationship patterns that are likely to have duplicates
        # Focus on relationships between high-degree nodes
        high_degree_rels_query = """
        MATCH (a)-[r]->(b)
        WHERE a.name IS NOT NULL AND b.name IS NOT NULL
        WITH a, b, type(r) AS rel_type, collect(r) AS rels, count(*) AS count,
             COUNT { (a)--() } AS a_degree, COUNT { (b)--() } AS b_degree
        WHERE count > 1 OR (a_degree > 5 AND b_degree > 5)
        RETURN a.id AS from_id, a.name AS from_name,
               rel_type,
               b.id AS to_id, b.name AS to_name,
               count, [rel IN rels | properties(rel)] AS rel_properties
        ORDER BY count DESC
        LIMIT 100
        """
        high_degree_relationships = [dict(record) for record in sess.run(high_degree_rels_query)]
        
        return {
            "total_nodes": total_nodes,
            "total_relationships": total_rels,
            "label_counts": label_counts,
            "rel_type_counts": rel_type_counts,
            "high_degree_nodes": high_degree_nodes,
            "high_degree_relationships": high_degree_relationships
        }

def find_duplicate_relationships(driver) -> List[Dict[str, Any]]:
    """Find duplicate relationships between the same nodes across the entire graph."""
    with driver.session() as sess:
        query = """
        MATCH (a)-[r]->(b)
        WHERE a.id IS NOT NULL AND b.id IS NOT NULL
        WITH a.id AS from_id, b.id AS to_id, a.name AS from_name, b.name AS to_name,
             type(r) AS rel_type, collect(r) AS rels, count(*) AS count
        WHERE count > 1
        RETURN from_id, to_id, rel_type, count, 
               [rel IN rels | elementId(rel)] AS rel_ids,
               from_name, to_name
        ORDER BY count DESC
        """
        duplicates = [dict(record) for record in sess.run(query)]
        return duplicates

def find_orphaned_nodes(driver) -> List[Dict[str, Any]]:
    """Find nodes with no relationships (orphaned nodes)."""
    with driver.session() as sess:
        query = """
        MATCH (n)
        WHERE NOT (n)--()
          AND n.name IS NOT NULL
        RETURN n.id AS id, n.name AS name, labels(n) AS labels,
               properties(n) AS properties
        LIMIT 100
        """
        orphaned = [dict(record) for record in sess.run(query)]
        return orphaned

def find_relationship_synonyms(driver) -> List[Dict[str, Any]]:
    """Find relationship types that might be synonyms (same node pairs with different relationship types)."""
    with driver.session() as sess:
        # Find cases where the same node pairs have multiple different relationship types
        # that might be synonyms (e.g., "partner_of" and "partners_with" between same nodes)
        query = """
        MATCH (a)-[r]->(b)
        WHERE a.id IS NOT NULL AND b.id IS NOT NULL
        WITH a.id AS from_id, b.id AS to_id, 
             collect(DISTINCT type(r)) AS rel_types
        WHERE size(rel_types) > 1
        RETURN from_id, to_id, rel_types, size(rel_types) AS type_count
        ORDER BY type_count DESC
        LIMIT 100
        """
        synonyms = [dict(record) for record in sess.run(query)]
        return synonyms

async def analyze_graph(driver) -> Optional[CuratorAnalysis]:
    """Use the curator agent to analyze the graph and find consolidation opportunities."""
    try:
        # Get graph-wide statistics and targeted samples
        graph_stats = get_graph_statistics(driver)
        
        # Get specific patterns across the entire graph
        duplicate_rels = find_duplicate_relationships(driver)
        orphaned_nodes = find_orphaned_nodes(driver)
        relationship_synonyms = find_relationship_synonyms(driver)
        
        logger.info(f"Found {len(duplicate_rels)} duplicate relationships, {len(orphaned_nodes)} orphaned nodes, {len(relationship_synonyms)} potential relationship synonyms")
        
        # Prepare context for the agent with comprehensive data
        context = {
            "graph_statistics": {
                "total_nodes": graph_stats["total_nodes"],
                "total_relationships": graph_stats["total_relationships"],
                "label_distribution": graph_stats["label_counts"],
                "relationship_type_distribution": graph_stats["rel_type_counts"]
            },
            "duplicate_relationships": duplicate_rels[:100],  # Top 100 duplicates
            "orphaned_nodes": orphaned_nodes[:50],  # Top 50 orphaned nodes
            "relationship_synonyms": relationship_synonyms[:50],  # Potential synonym patterns
            "high_degree_nodes_sample": graph_stats["high_degree_nodes"][:30],  # Sample of high-degree nodes
            "high_degree_relationships_sample": graph_stats["high_degree_relationships"][:50],  # Sample of high-degree rels
        }
        
        # Log some examples for debugging
        if duplicate_rels:
            logger.info(f"Example duplicate relationship: {duplicate_rels[0]}")
        if relationship_synonyms:
            logger.info(f"Example relationship synonym: {relationship_synonyms[0]}")
        
        context_str = json.dumps(context, indent=2, default=str)
        
        # Run agent analysis
        result = await Runner.run(curator_agent, context_str)
        output = result.final_output
        
        # Log the raw output for debugging
        if isinstance(output, str):
            logger.debug(f"Agent raw output (first 1000 chars): {output[:1000]}")
        
        # Parse response
        if isinstance(output, str):
            try:
                # Try to extract JSON from markdown code blocks if present
                json_str = output.strip()
                if json_str.startswith("```"):
                    lines = json_str.split("\n")
                    json_str = "\n".join(lines[1:-1]) if len(lines) > 2 else json_str
                elif json_str.startswith("```json"):
                    lines = json_str.split("\n")
                    json_str = "\n".join(lines[1:-1]) if len(lines) > 2 else json_str
                
                data = json.loads(json_str)
                return CuratorAnalysis(**data)
            except (json.JSONDecodeError, ValidationError) as e:
                logger.error(f"Failed to parse curator analysis: {e}\nResponse: {output[:500]}")
                return None
        
        if isinstance(output, CuratorAnalysis):
            return output
        
        logger.warning(f"Unexpected output type from curator agent: {type(output)}")
        return None
        
    except Exception as e:
        logger.error(f"Error analyzing graph: {e}", exc_info=True)
        return None

def consolidate_duplicate_relationships(driver, from_id: str, to_id: str, rel_type: str, rel_ids: List[str]):
    """Consolidate duplicate relationships by keeping one and removing others."""
    if len(rel_ids) <= 1:
        return
    
    # Keep the first relationship, remove the rest
    keep_id = rel_ids[0]
    remove_ids = rel_ids[1:]
    
    with driver.session() as sess:
        # Remove duplicate relationships
        for rel_id in remove_ids:
            query = """
            MATCH ()-[r]->()
            WHERE elementId(r) = $rel_id
            DELETE r
            """
            sess.run(query, rel_id=rel_id)
        
        logger.info(f"Consolidated {len(remove_ids)} duplicate {rel_type} relationships between {from_id} and {to_id}")

def merge_similar_nodes(driver, node1_id: str, node2_id: str):
    """Merge two similar nodes into one."""
    with driver.session() as sess:
        # Get node details
        query1 = """
        MATCH (n {id: $id})
        RETURN labels(n) AS labels, properties(n) AS props
        """
        node1 = sess.run(query1, id=node1_id).single()
        node2 = sess.run(query1, id=node2_id).single()
        
        if not node1 or not node2:
            logger.warning(f"Could not find nodes {node1_id} or {node2_id} for merging")
            return
        
        # Get all relationships from node2 and recreate on node1
        get_rels = """
        MATCH (n2 {id: $node2_id})-[r]->(other)
        RETURN type(r) AS rel_type, properties(r) AS rel_props, other.id AS other_id, 'outgoing' AS direction
        UNION
        MATCH (other)<-[r]-(n2 {id: $node2_id})
        RETURN type(r) AS rel_type, properties(r) AS rel_props, other.id AS other_id, 'incoming' AS direction
        """
        rels = [dict(record) for record in sess.run(get_rels, node2_id=node2_id)]
        
        # Recreate relationships on node1 (only if they don't already exist)
        for rel in rels:
            rel_type = rel["rel_type"]
            rel_props = rel["rel_props"]
            other_id = rel["other_id"]
            direction = rel["direction"]
            
            # Check if relationship already exists
            if direction == "outgoing":
                check = """
                MATCH (n1 {id: $node1_id})-[r]->(other {id: $other_id})
                WHERE type(r) = $rel_type
                RETURN count(r) AS exists
                """
            else:
                check = """
                MATCH (other {id: $other_id})-[r]->(n1 {id: $node1_id})
                WHERE type(r) = $rel_type
                RETURN count(r) AS exists
                """
            
            exists = sess.run(check, node1_id=node1_id, other_id=other_id, rel_type=rel_type).single()["exists"]
            
            if exists == 0:
                # Create relationship with dynamic type
                if direction == "outgoing":
                    create_rel = f"""
                    MATCH (n1 {{id: $node1_id}}), (other {{id: $other_id}})
                    CREATE (n1)-[r:`{rel_type}`]->(other)
                    SET r = $rel_props
                    """
                else:
                    create_rel = f"""
                    MATCH (n1 {{id: $node1_id}}), (other {{id: $other_id}})
                    CREATE (other)-[r:`{rel_type}`]->(n1)
                    SET r = $rel_props
                    """
                sess.run(create_rel, node1_id=node1_id, other_id=other_id, rel_props=rel_props)
        
        # Delete old relationships from node2
        delete_rels = """
        MATCH (n2 {id: $node2_id})-[r]->()
        DELETE r
        MATCH ()-[r]->(n2 {id: $node2_id})
        DELETE r
        """
        sess.run(delete_rels, node2_id=node2_id)
        
        # Merge labels - combine all labels from both nodes
        all_labels = set(node1["labels"] + node2["labels"])
        if all_labels:
            label_str = ":".join(all_labels)
            set_labels = f"""
            MATCH (n1 {{id: $node1_id}})
            SET n1:{label_str}
            """
            sess.run(set_labels, node1_id=node1_id)
        
        # Merge properties - add missing properties from node2 to node1
        node2_props = node2["props"]
        for key, value in node2_props.items():
            if key != "id" and value is not None:
                # Check if property exists on node1
                check_prop = """
                MATCH (n1 {id: $node1_id})
                RETURN n1[$key] AS value
                """
                result = sess.run(check_prop, node1_id=node1_id, key=key).single()
                if result and result["value"] is None:
                    # Property doesn't exist, add it
                    set_prop = f"""
                    MATCH (n1 {{id: $node1_id}})
                    SET n1.{key} = $value
                    """
                    sess.run(set_prop, node1_id=node1_id, value=value)
        
        # Delete node2
        delete_node = """
        MATCH (n2 {id: $node2_id})
        DELETE n2
        """
        sess.run(delete_node, node2_id=node2_id)
        
        logger.info(f"Merged node {node2_id} into {node1_id}")

def apply_consolidations(driver, analysis: CuratorAnalysis):
    """Apply the consolidation opportunities found by the agent."""
    applied = 0
    skipped = 0
    
    for opp in analysis.opportunities:
        # Only apply high-confidence opportunities
        if opp.confidence < 0.7:
            logger.debug(f"Skipping low-confidence opportunity: {opp.description} (confidence: {opp.confidence})")
            skipped += 1
            continue
        
        try:
            if opp.type == "duplicate_relationship" and opp.relationships:
                # Consolidate duplicate relationships
                rel = opp.relationships[0]
                from_id = rel.get("from") or opp.entities[0]
                to_id = rel.get("to") or opp.entities[1]
                rel_type = rel.get("type", "")
                
                # Find the actual duplicate relationships
                with driver.session() as sess:
                    query = """
                    MATCH (a {id: $from_id})-[r]->(b {id: $to_id})
                    WHERE type(r) = $rel_type
                    RETURN collect(elementId(r)) AS rel_ids
                    """
                    result = sess.run(query, from_id=from_id, to_id=to_id, rel_type=rel_type).single()
                    if result and len(result["rel_ids"]) > 1:
                        consolidate_duplicate_relationships(driver, from_id, to_id, rel_type, result["rel_ids"])
                        applied += 1
            
            elif opp.type == "relationship_synonym" and opp.relationships:
                # Consolidate relationship synonyms - merge different relationship types
                # that mean the same thing into a single canonical type
                for rel_info in opp.relationships:
                    if "types" in rel_info and "canonical" in rel_info:
                        from_id = rel_info.get("from") or opp.entities[0]
                        to_id = rel_info.get("to") or opp.entities[1]
                        canonical_type = rel_info["canonical"]
                        synonym_types = rel_info["types"]
                        
                        # Sanitize relationship types for Neo4j
                        safe_canonical = "".join(c if c.isalnum() or c == "_" else "_" for c in canonical_type)
                        
                        # Rename all synonym types to the canonical type
                        with driver.session() as sess:
                            for syn_type in synonym_types:
                                if syn_type != canonical_type:
                                    safe_syn_type = "".join(c if c.isalnum() or c == "_" else "_" for c in syn_type)
                                    query = f"""
                                    MATCH (a {{id: $from_id}})-[r:`{safe_syn_type}`]->(b {{id: $to_id}})
                                    WITH a, b, r, properties(r) AS rel_props
                                    WHERE NOT EXISTS((a)-[:`{safe_canonical}`]->(b))
                                    CREATE (a)-[r2:`{safe_canonical}`]->(b)
                                    SET r2 = rel_props
                                    DELETE r
                                    """
                                    try:
                                        sess.run(query, from_id=from_id, to_id=to_id)
                                        logger.info(f"Consolidated relationship type '{syn_type}' to '{canonical_type}' between {from_id} and {to_id}")
                                    except Exception as e:
                                        logger.warning(f"Error consolidating relationship synonym: {e}")
                        applied += 1
                    else:
                        logger.warning(f"Relationship synonym missing 'types' or 'canonical' field: {rel_info}")
                        skipped += 1
            
            elif opp.type == "orphaned_node" and len(opp.entities) >= 1:
                # Optionally remove orphaned nodes (low-quality extractions)
                # For now, just log them - might want to keep for future connections
                logger.info(f"Found orphaned node: {opp.entities[0]} - {opp.description}")
                skipped += 1
            
            else:
                logger.info(f"Opportunity type '{opp.type}' not yet implemented: {opp.description}")
                skipped += 1
                
        except Exception as e:
            logger.error(f"Error applying consolidation: {opp.description} - {e}", exc_info=True)
            skipped += 1
    
    logger.info(f"Applied {applied} consolidations, skipped {skipped}")

def run_curation():
    """Main curation function that analyzes and cleans up the graph."""
    logger.info("Starting graph curation run…")
    driver = None
    
    # Establish connection to Neo4j
    for attempt in range(12):
        try:
            driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
            driver.verify_connectivity()
            logger.info("Successfully connected to Neo4j for curation.")
            break
        except Exception as e:
            logger.warning(f"Curator: Neo4j not available yet, retrying... (attempt {attempt+1}/12)")
            time.sleep(5)
            driver = None
    
    if not driver:
        logger.error("Unable to connect to Neo4j after multiple attempts. Skipping this curation run.")
        return
    
    try:
        # Analyze graph
        analysis = asyncio.run(analyze_graph(driver))
        
        if analysis:
            logger.info(f"Curator analysis complete: {analysis.summary}")
            logger.info(f"Found {len(analysis.opportunities)} consolidation opportunities")
            
            # Log details of opportunities found
            for i, opp in enumerate(analysis.opportunities[:10]):  # Log first 10
                logger.info(f"Opportunity {i+1}: {opp.type} - {opp.description} (confidence: {opp.confidence:.2f})")
            
            # Apply consolidations
            apply_consolidations(driver, analysis)
        else:
            logger.warning("Curator analysis returned no results")
            # Try to get some raw data to see what we're working with
            try:
                duplicate_rels = find_duplicate_relationships(driver)
                if duplicate_rels:
                    logger.info(f"Found {len(duplicate_rels)} duplicate relationships but agent didn't process them")
                    logger.info(f"First duplicate: {duplicate_rels[0]}")
            except Exception as e:
                logger.debug(f"Error checking duplicates: {e}")
            
    except Exception as e:
        logger.error(f"An error occurred during curation run: {e}", exc_info=True)
    finally:
        if driver:
            driver.close()

def main():
    # Schedule to run every 6 hours
    schedule.every(6).hours.do(run_curation)
    logger.info("Curator started – scheduled to run every 6 hours")
    # Immediate first run
    run_curation()
    
    while True:
        schedule.run_pending()
        time.sleep(60)

if __name__ == "__main__":
    main()

