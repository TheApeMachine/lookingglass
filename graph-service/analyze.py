import os
import time
import logging
import schedule
from neo4j import GraphDatabase
from graphdatascience import GraphDataScience

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("analytics")

NEO4J_HOST = os.getenv("NEO4J_HOST", "neo4j")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")
NEO4J_URI = f"bolt://{NEO4J_HOST}:7687"

GDS_GRAPH_NAME = "org_graph"


def run_analytics():
    logger.info("Starting graph analytics run…")
    driver = None
    
    # First, establish a resilient connection to Neo4j
    for attempt in range(12):
        try:
            driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
            driver.verify_connectivity()
            logger.info("Successfully connected to Neo4j for analytics.")
            break
        except Exception as e:
            logger.warning(f"Analytics worker: Neo4j not available yet, retrying... (attempt {attempt+1}/12)")
            time.sleep(5)
            driver = None # Reset driver on failure
    
    if not driver:
        logger.error("Unable to connect to Neo4j after multiple attempts. Skipping this analytics run.")
        return

    try:
        # Now, connect to GDS using the existing driver
        gds = GraphDataScience(driver)
        
        with driver.session() as session:
            # Abort run early if there are no organization nodes yet
            org_count_res = session.run("MATCH (o:Organization) RETURN count(o) AS cnt").single()
            if org_count_res["cnt"] == 0:
                logger.info("No Organization nodes present yet – skipping this analytics run.")
                return

            # Project organization-level graph (org nodes + SOFT_RELATED / PARTNER_OF edges)
            projection_cypher = {
                "nodeQuery": "MATCH (o:Organization) RETURN id(o) AS id",
                "relationshipQuery": """
                    MATCH (o1:Organization)-[r]-(o2:Organization)
                    WHERE type(r) IN ['SOFT_RELATED','PARTNER_OF']
                    RETURN id(o1) AS source, id(o2) AS target, coalesce(r.shared_persons,1) AS weight
                """,
            }
            # Drop old projection if exists
            if gds.graph.exists(GDS_GRAPH_NAME)['exists']:
                gds.graph.drop(GDS_GRAPH_NAME)
            
            # Project via Cypher
            gds.graph.project.cypher(
                GDS_GRAPH_NAME,
                projection_cypher["nodeQuery"],
                projection_cypher["relationshipQuery"],
            )

            # Retrieve a Graph object handle for algorithm calls
            graph = gds.graph.get(GDS_GRAPH_NAME)

            # Community detection (Louvain)
            gds.louvain.mutate(graph, mutateProperty="community")

            # PageRank
            gds.pageRank.mutate(graph, mutateProperty="pagerank")

            # Write back scores to Neo4j node properties
            gds.graph.writeNodeProperties(graph, ["community", "pagerank"])

            # Drop in-memory projection
            gds.graph.drop(graph)

        logger.info("Graph analytics run finished.")
    except Exception as e:
        logger.error(f"An error occurred during analytics run: {e}")
    finally:
        if driver:
            driver.close()


def main():
    # Schedule to run once daily at 02:00
    schedule.every().day.at("02:00").do(run_analytics)
    logger.info("Analytics worker started – scheduled daily 02:00 run")
    # Immediate first run
    run_analytics()

    while True:
        schedule.run_pending()
        time.sleep(60)


if __name__ == "__main__":
    main() 