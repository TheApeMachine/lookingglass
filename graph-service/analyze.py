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
    gds = None

    for attempt in range(12):  # retry for ~1 minute
        try:
            gds = GraphDataScience(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
            logger.info("Successfully connected to Graph Data Science library.")
            break
        except Exception:
            logger.warning(f"GDS not available yet (attempt {attempt + 1}/12). Retrying...")
            time.sleep(5)
    
    if not gds:
        logger.error("Unable to connect to GDS after multiple attempts. Skipping this analytics run.")
        return

    try:
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
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
            # Drop old projection if exists (handle bool, DataFrame, or Series return types without ambiguous truth-value errors)
            exists_resp = gds.graph.exists(GDS_GRAPH_NAME)
            graph_exists = False
            if isinstance(exists_resp, bool):
                graph_exists = exists_resp
            else:
                # For DataFrame/Series-like results, try to access an "exists" field safely
                try:
                    if hasattr(exists_resp, "empty") and not exists_resp.empty:
                        if "exists" in exists_resp:
                            graph_exists = bool(getattr(exists_resp["exists"], "iloc", lambda i=None: exists_resp["exists"]) [0])
                except Exception:
                    pass  # fall back to False if structure unexpected
            if graph_exists:
                gds.graph.drop(GDS_GRAPH_NAME)
            # Project via Cypher (GDS 3.x+)
            project_result = gds.graph.project.cypher(
                GDS_GRAPH_NAME,
                projection_cypher["nodeQuery"],
                projection_cypher["relationshipQuery"],
            )
            # project_result is a GraphCreateResult (mapping-like)
            node_ct = project_result.get("nodeCount") if isinstance(project_result, dict) else getattr(project_result, "nodeCount", "?")
            rel_ct = project_result.get("relationshipCount") if isinstance(project_result, dict) else getattr(project_result, "relationshipCount", "?")
            logger.info(f"Projected graph with {node_ct} nodes, {rel_ct} relationships")

            # Retrieve a Graph object handle for algorithm calls
            graph = gds.graph.get(GDS_GRAPH_NAME)

            # Community detection (Louvain)
            communities = gds.louvain.mutate(graph, relationshipWeightProperty="weight", mutateProperty="community")

            # PageRank
            pr = gds.pageRank.mutate(graph, relationshipWeightProperty="weight", mutateProperty="pagerank")

            # Write back scores to Neo4j node properties
            gds.graph.writeNodeProperties(graph, ["community", "pagerank"])

            # Drop in-memory projection
            gds.graph.drop(graph)

        logger.info("Graph analytics run finished.")
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