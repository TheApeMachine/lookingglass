import os
import asyncio
import logging
import json
import hashlib
import re
from typing import List, Dict, Optional
from pydantic import BaseModel, ValidationError
from agents import Agent, Runner, OpenAIChatCompletionsModel
from openai import AsyncOpenAI
from minio import Minio
from neo4j import GraphDatabase
from html_to_markdown import convert_to_markdown

# --- Logging & Config ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("kg-worker")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "lm-studio")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "http://localhost:1234/v1")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "openai/gpt-oss-20b")

openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)
logger.info(f"Using OpenAI-compatible endpoint: {OPENAI_BASE_URL} with model: {OPENAI_MODEL}")

MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "localhost:9000")
MINIO_USER = os.getenv("MINIO_USER", "miniouser")
MINIO_PASSWORD = os.getenv("MINIO_PASSWORD", "miniopassword")
MINIO_BUCKET = os.getenv("MINIO_BUCKET", "scraped")
MINIO_PREFIX = os.getenv("MINIO_PREFIX", "pages/")  # e.g., your HTML files stored there

neo_driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI", "bolt://localhost:7687"),
    auth=(os.getenv("NEO4J_USER","neo4j"), os.getenv("NEO4J_PASSWORD","password"))
)

minio_client = Minio(
    MINIO_ENDPOINT,
    access_key=MINIO_USER,
    secret_key=MINIO_PASSWORD,
    secure=False
)

# --- KG extraction schema & agent ---
# Model for what the agent returns (without IDs)
class EntityInput(BaseModel):
    name: str
    type: str
    metadata: Optional[Dict[str, str]] = None  # Additional metadata about the entity

class RelationInput(BaseModel):
    origin: str  # Entity name (not ID)
    destination: str  # Entity name (not ID)
    type: str

class KGExtractionInput(BaseModel):
    entities: List[EntityInput]
    relations: List[RelationInput]

# Internal model with generated IDs
class Entity(BaseModel):
    id: str
    name: str
    type: str
    metadata: Optional[Dict[str, str]] = None

class Relation(BaseModel):
    origin: str  # Entity ID
    destination: str  # Entity ID
    type: str

class KGExtractionResult(BaseModel):
    entities: List[Entity]
    relations: List[Relation]

kg_agent = Agent(
    name="KG Extractor",
    instructions=("""
        Extract high-quality named entities and their relationships from the given text.
        
        Focus ONLY on:
        - Specific, concrete named entities (e.g. people, organizations, companies, products, locations, events, etc.)
        - Avoid generic descriptors, common nouns, or vague concepts
        - Only extract entities that are explicitly mentioned and have clear identity
        - Extract meaningful relationships between entities (e.g., "works_for", "located_in", "partner_of", "founder_of", "acquired_by", etc.)
        
        IMPORTANT - Entity Name Rules:
        - Use ONLY canonical/base names - remove all modifiers, honorifics, and titles
        - Do NOT include prefixes like "Dr.", "Mr.", "Mrs.", "Ms.", "Prof.", "Sir", "Lord", etc. in the name
        - Do NOT include suffixes like "Jr.", "Sr.", "III", "PhD", "MD", etc. in the name
        - Do NOT include company suffixes like "Inc.", "LLC", "Ltd.", "Corp.", etc. in the name
        - Use the core, canonical name only (e.g., "John Smith" not "Dr. John Smith Jr.")
        - If honorifics, titles, or other modifiers are relevant, include them in the metadata field instead
        
        For each entity, if there is additional metadata available, include it in the metadata field.
        
        Return exactly a JSON object matching this schema:
        {
          "entities": [
            {
              "name": "Canonical Entity Name (no modifiers)",
              "type": "EntityType (e.g., Person, Organization, Location, Product)",
              "metadata": {"key": "value"}
            }
          ],
          "relations": [
            {
              "origin": "entity_name",
              "destination": "entity_name",
              "type": "relationship_type"
            }
          ]
        }
        
        For relations, use the exact entity names from the entities array. No extra fields, comments, or markdown formatting.
    """),
    model=OpenAIChatCompletionsModel(
        model=OPENAI_MODEL,
        openai_client=openai_client
    ),
)

# --- Helpers ---
def normalize_name(n: str) -> str:
    """Normalize entity name for consistent ID generation (case-insensitive, trimmed)."""
    # Trim and convert to lowercase for comparison
    return n.strip().lower()

def generate_entity_id(name: str) -> str:
    """Generate a deterministic ID for an entity based on normalized name only."""
    normalized_name = normalize_name(name)
    # Use hash to create a fixed-length ID
    return hashlib.sha256(normalized_name.encode('utf-8')).hexdigest()[:32]

def process_kg_input(kg_input: KGExtractionInput) -> KGExtractionResult:
    """Convert agent output (with names) to internal format (with IDs)."""
    # Create mapping from entity name to ID
    name_to_id = {}
    entities = []
    seen_ids = set()  # Track IDs to avoid duplicates within a single extraction
    
    for ent_input in kg_input.entities:
        # Trim the name and generate ID based only on normalized name
        trimmed_name = ent_input.name.strip()
        entity_id = generate_entity_id(trimmed_name)
        
        # Only add if we haven't seen this ID in this extraction
        if entity_id not in seen_ids:
            seen_ids.add(entity_id)
            name_to_id[ent_input.name] = entity_id
            name_to_id[trimmed_name] = entity_id  # Also map trimmed version
            entities.append(Entity(
                id=entity_id,
                name=trimmed_name,  # Store trimmed name
                type=ent_input.type.strip(),  # Use model's type, trimmed
                metadata=ent_input.metadata
            ))
        else:
            # Entity already seen in this extraction, just add name mapping
            name_to_id[ent_input.name] = entity_id
            name_to_id[trimmed_name] = entity_id
    
    # Convert relations to use IDs instead of names
    relations = []
    for rel_input in kg_input.relations:
        # Try both original and trimmed names (case-insensitive matching)
        origin_name = rel_input.origin.strip()
        dest_name = rel_input.destination.strip()
        
        # Try to find ID by normalized name if direct lookup fails
        origin_id = name_to_id.get(rel_input.origin) or name_to_id.get(origin_name)
        if not origin_id:
            # Try case-insensitive lookup
            origin_normalized = normalize_name(origin_name)
            for name, eid in name_to_id.items():
                if normalize_name(name) == origin_normalized:
                    origin_id = eid
                    break
        
        dest_id = name_to_id.get(rel_input.destination) or name_to_id.get(dest_name)
        if not dest_id:
            # Try case-insensitive lookup
            dest_normalized = normalize_name(dest_name)
            for name, eid in name_to_id.items():
                if normalize_name(name) == dest_normalized:
                    dest_id = eid
                    break
        
        if origin_id and dest_id:
            relations.append(Relation(
                origin=origin_id,
                destination=dest_id,
                type=rel_input.type.strip()
            ))
        else:
            logger.warning(f"Could not find entity IDs for relation: {rel_input.origin} -> {rel_input.destination}")
    
    return KGExtractionResult(entities=entities, relations=relations)

def ingest_to_neo(kg: KGExtractionResult, source_url: str):
    with neo_driver.session() as sess:
        # First, ensure all entities exist with their properties
        # MERGE on ID (based on normalized name) to avoid duplicates
        for ent in kg.entities:
            # Ensure name is trimmed
            trimmed_name = ent.name.strip()
            label = ent.type.strip()  # Trim type as well
            
            # Use MERGE to create or update node, ensuring uniqueness by id (name-based)
            # Store name, source_url, type, and any metadata
            set_clauses = ["n.name = $name", "n.type = $type", "n.source_url = $source_url"]
            params = {"id": ent.id, "name": trimmed_name, "type": label, "source_url": source_url}
            
            # Add metadata fields if present
            if ent.metadata:
                for key, value in ent.metadata.items():
                    # Sanitize key for Neo4j property name (alphanumeric and underscore only)
                    safe_key = "".join(c if c.isalnum() or c == "_" else "_" for c in key)
                    set_clauses.append(f"n.{safe_key} = ${safe_key}")
                    params[safe_key] = value
            
            # MERGE on ID first (name-based), then add label and set properties
            # If node exists with different label, add the new label too
            # Sanitize label for Neo4j (alphanumeric and underscore only)
            safe_label = "".join(c if c.isalnum() or c == "_" else "_" for c in label)
            query = f"""
            MERGE (n {{id: $id}})
            SET n:{safe_label}
            SET {', '.join(set_clauses)}
            """
            sess.run(query, **params)
        
        # Then create relationships (using separate queries to avoid cartesian product)
        for rel in kg.relations:
            # Sanitize relationship type for Neo4j (alphanumeric and underscore only)
            safe_rel_type = "".join(c if c.isalnum() or c == "_" else "_" for c in rel.type.strip())
            # Match origin node first, then destination in the same query pattern
            # This avoids cartesian product by connecting the patterns
            query = f"""
            MATCH (a {{id: $origin}})
            WITH a
            MATCH (b {{id: $destination}})
            MERGE (a)-[r:`{safe_rel_type}`]->(b)
            """
            sess.run(query, origin=rel.origin, destination=rel.destination)
    
    logger.info(f"Ingested {len(kg.entities)} entities and {len(kg.relations)} relations into Neo4j.")

async def process_text(text: str, source_url: str = "") -> Optional[KGExtractionResult]:
    # Run extraction using Runner
    result = await Runner.run(kg_agent, text)
    output = result.final_output
    
    # Parse JSON string response
    if isinstance(output, str):
        try:
            # Try to extract JSON from markdown code blocks if present
            json_str = output.strip()
            if json_str.startswith("```"):
                # Extract JSON from code block
                lines = json_str.split("\n")
                json_str = "\n".join(lines[1:-1]) if len(lines) > 2 else json_str
            elif json_str.startswith("```json"):
                lines = json_str.split("\n")
                json_str = "\n".join(lines[1:-1]) if len(lines) > 2 else json_str
            
            # Parse JSON
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON from agent response: {e}\nResponse: {output[:500]}")
            return None
        
        # Validate with Pydantic (only reached if JSON parsing succeeded)
        try:
            # Ensure metadata field exists for entities that don't have it
            if "entities" in data:
                for entity in data["entities"]:
                    if "metadata" not in entity:
                        entity["metadata"] = None
            # Parse as input format (with names, not IDs)
            kg_input = KGExtractionInput(**data)
            # Convert to result format (with generated IDs)
            return process_kg_input(kg_input)
        except ValidationError as e:
            logger.error(f"Failed to validate KG extraction result: {e}\nParsed data: {data}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error creating KG extraction result: {e}\nData: {data}")
            return None
    
    logger.warning(f"Unexpected output type from agent: {type(output)}")
    return None

def process_html(html_bytes: bytes) -> str:
    html = html_bytes.decode("utf-8", errors="ignore")
    return convert_to_markdown(html)

# --- Main worker loop with MinIO listener ---
def run_listener():
    if not minio_client.bucket_exists(MINIO_BUCKET):
        logger.error("Bucket does not exist: %s", MINIO_BUCKET)
        return

    logger.info("Listening for new objects in bucket %s (prefix %s)...", MINIO_BUCKET, MINIO_PREFIX)

    for event in minio_client.listen_bucket_notification(
        MINIO_BUCKET,
        prefix=MINIO_PREFIX,
        suffix=".html",
        events=["s3:ObjectCreated:*"]
    ):
        for rec in event.get("Records", []):
            obj_name = rec["s3"]["object"]["key"]
            logger.info("Detected new object: %s", obj_name)
            try:
                # Get object and its metadata
                obj = minio_client.stat_object(MINIO_BUCKET, obj_name)
                source_url = ""
                
                # Extract source URL from metadata
                if obj.metadata:
                    # MinIO metadata keys are lowercase with hyphens
                    source_url = obj.metadata.get("x-amz-meta-source-url", "")
                    if not source_url:
                        # Try alternative metadata key formats
                        source_url = obj.metadata.get("source-url", "")
                
                with minio_client.get_object(MINIO_BUCKET, obj_name) as resp:
                    html_bytes = resp.read()
                
                md = process_html(html_bytes)
                kg = asyncio.run(process_text(md, source_url=source_url))
                if kg:
                    ingest_to_neo(kg, source_url=source_url)
                else:
                    logger.warning("KG extraction returned None for %s", obj_name)
            except Exception as e:
                logger.error("Error processing %s: %s", obj_name, e, exc_info=True)

if __name__ == "__main__":
    run_listener()
