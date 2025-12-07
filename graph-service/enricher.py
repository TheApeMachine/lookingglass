import os
import asyncio
import logging
import json
import time
import schedule
import aiohttp
import urllib.parse
import re
import html as html_stdlib
from typing import List, Dict, Optional, Any
from pydantic import BaseModel, ValidationError
from agents import Agent, Runner, OpenAIChatCompletionsModel, set_tracing_disabled
from openai import AsyncOpenAI
from neo4j import GraphDatabase
from playwright.async_api import async_playwright, Browser, Page, TimeoutError as PlaywrightTimeoutError
from html_to_markdown import convert_to_markdown
from lxml import html

# --- Logging & Config ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("enricher")

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

BRAVE_API_KEY = os.getenv("BRAVE_API_KEY", "<your-brave-api-key>")
BRAVE_SEARCH_URL = "https://api.search.brave.com/res/v1/web/search"

NORTHDATA_BASE_URL = "https://www.northdata.com"

# --- Enrichment agent schema ---
class EntityEnrichment(BaseModel):
    name: str
    type: str
    metadata: Optional[Dict[str, Any]] = None  # Accept any type, will be converted to string

class RelationEnrichment(BaseModel):
    origin: str  # Entity name
    destination: str  # Entity name
    type: str

class EnrichmentResult(BaseModel):
    entities: List[EntityEnrichment]
    relations: List[RelationEnrichment]
    metadata_updates: Optional[Dict[str, Any]] = None  # Accept any type, will be converted to string
    confidence: Optional[float] = None  # Confidence score 0.0-1.0 that this page is about the target entity

class SearchResultSelection(BaseModel):
    """Model for agent to select which search results to visit."""
    selected_urls: List[str]  # URLs that look interesting/relevant
    reasoning: str  # Why these URLs were selected

result_selection_agent = Agent(
    name="Search Result Selector",
    instructions=("""
        You are analyzing search results for an entity to determine which pages would be most valuable to visit for enrichment.
        
        Entity: {entity_name} (type: {entity_type})
        
        Your task: Review the search results and select the URLs that are most likely to contain:
        - Additional information about the entity
        - Related entities and relationships
        - Useful metadata (location, industry, dates, etc.)
        
        Avoid:
        - Generic directory listings
        - Social media profiles (unless they're official/verified)
        - Pages that are likely duplicates of what we already have
        
        Return exactly a JSON object:
        {{
          "selected_urls": ["url1", "url2", "url3"],
          "reasoning": "Brief explanation of why these URLs were selected"
        }}
        
        Select 3-5 most promising URLs. No extra fields, comments, or markdown formatting.
    """),
    model=OpenAIChatCompletionsModel(
        model=OPENAI_MODEL,
        openai_client=openai_client
    ),
)

enrichment_agent = Agent(
    name="Graph Enricher",
    instructions=("""
        Analyze the provided web page content and extract additional information about the entity.
        
        You will receive context about the target entity including:
        - Name and type
        - Known metadata (location, dates, etc.)
        - Existing relationships to other entities
        
        Use this context to verify that the page is about the correct entity, especially for common names.
        
        CRITICAL: Before extracting information, verify that this page is actually about the target entity.
        - Compare the page content with the known metadata and relationships
        - For common names (e.g., "John Smith"), match against known locations, organizations, dates, roles, etc.
        - If the known metadata/relationships don't match what's on the page, this is likely a different entity
        - If the page mentions multiple people with the same name, use the context to identify which one matches
        - If you're not confident (confidence < 0.7), return empty results rather than risk incorrect data
        
        Your task:
        1. Assess confidence (0.0-1.0) that this page is about the target entity
        2. Only if confidence >= 0.7, extract:
           - Additional entities mentioned on the page that relate to the target entity
           - New relationships between the target entity and other entities
           - Additional metadata about the target entity (e.g., location, industry, founding date, employees, etc.)
        
        Focus on:
        - High-quality, factual information
        - Specific named entities (people, organizations, locations, products, etc.)
        - Clear, meaningful relationships
        - Useful metadata that adds value
        
        Use ONLY canonical names (no modifiers like "Dr.", "Inc.", etc.) - put those in metadata if relevant.
        
        Return exactly a JSON object matching this schema:
        {{
          "confidence": 0.85,
          "entities": [
            {{
              "name": "Canonical Entity Name",
              "type": "EntityType",
              "metadata": {{"key": "value"}}
            }}
          ],
          "relations": [
            {{
              "origin": "target_entity_name",
              "destination": "other_entity_name",
              "type": "relationship_type"
            }}
          ],
          "metadata_updates": {{
            "key": "value"
          }}
        }}
        
        The "origin" in relations should match the target entity name. 
        If confidence < 0.7, return empty arrays for entities and relations.
        No extra fields, comments, or markdown formatting.
    """),
    model=OpenAIChatCompletionsModel(
        model=OPENAI_MODEL,
        openai_client=openai_client
    ),
)

# --- Helper functions ---
def normalize_name(n: str) -> str:
    """Normalize entity name for consistent ID generation (case-insensitive, trimmed)."""
    return n.strip().lower()

def generate_entity_id(name: str) -> str:
    """Generate a deterministic ID for an entity based on normalized name only."""
    import hashlib
    normalized_name = normalize_name(name)
    return hashlib.sha256(normalized_name.encode('utf-8')).hexdigest()[:32]

async def fetch_page_content(url: str, browser: Browser, timeout: int = 30000) -> Optional[str]:
    """Fetch page content using Playwright."""
    try:
        context = await browser.new_context(
            viewport={'width': 1920, 'height': 1080},
            user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            locale='en-US',
            timezone_id='America/New_York',
        )
        
        page = await context.new_page()
        
        # Anti-detection measures
        await page.add_init_script("""
            delete navigator.webdriver;
            Object.defineProperty(navigator, 'plugins', {
                get: function() { return [1, 2, 3, 4, 5]; },
            });
        """)
        
        try:
            await page.goto(url, wait_until="networkidle", timeout=timeout)
            await page.wait_for_timeout(2000)  # Wait for dynamic content
            
            # Get page content
            html = await page.content()
            return html
        except PlaywrightTimeoutError:
            logger.warning(f"Timeout loading {url}")
            return None
        finally:
            await page.close()
            await context.close()
            
    except Exception as e:
        logger.error(f"Error fetching {url}: {e}")
        return None


def extract_bar_charts(raw_html: str) -> str:
    """Extract numeric bar chart data embedded in data-data attributes (e.g., Northdata)."""
    snippets: List[str] = []
    for match in re.finditer(r'data-data=(?P<quote>["\'])(?P<data>.+?)(?P=quote)', raw_html, flags=re.DOTALL):
        try:
            payload_raw = match.group("data")
            payload_json = html_stdlib.unescape(payload_raw)
            data_obj = json.loads(payload_json)
        except Exception:
            continue

        items = data_obj.get("item", []) if isinstance(data_obj, dict) else []
        for item in items:
            item_title = item.get("title") or item.get("item") or "chart"
            data_block = item.get("data", {})
            rows = data_block.get("data") if isinstance(data_block, dict) else None
            if not rows or not isinstance(rows, list):
                continue
            snippets.append(f"Chart: {item_title}")
            for row in rows:
                year = row.get("year")
                formatted = row.get("formattedValue") or row.get("value0")
                if year is None or formatted is None:
                    continue
                snippets.append(f"- {year}: {formatted}")

    return "\n".join(snippets)

def build_search_query(entity_name: str, entity_type: str) -> str:
    """Build a high-quality search query using Brave Search operators."""
    # Use exact phrase matching for the entity name to avoid partial matches
    # This ensures we get results where the full name appears, not just first/last name
    name_quoted = f'"{entity_name}"'
    
    # Require the entity type to appear in the page (using + operator)
    # This helps filter out irrelevant matches
    type_required = f"+{entity_type}"
    
    # Combine: exact name phrase match + required type
    # The quotes ensure the full name appears together, preventing matches on just part of the name
    query = f'{name_quoted} {type_required}'
    
    return query

def is_enrichment_eligible_entity_type(entity_type: str) -> bool:
    """Check if entity type is eligible for enrichment."""
    eligible_types = [
        "company", "organization", "person", "emailaddress", "address", "institute", "event"
    ]
    # Case-insensitive matching
    entity_type_lower = entity_type.lower().strip()
    return any(eligible.lower() == entity_type_lower for eligible in eligible_types)

def is_northdata_eligible_entity_type(entity_type: str) -> bool:
    """Check if entity type is eligible for Northdata lookup (subset of enrichment-eligible types)."""
    northdata_types = ["company", "organization", "person"]
    entity_type_lower = entity_type.lower().strip()
    return any(eligible.lower() == entity_type_lower for eligible in northdata_types)

async def search_northdata(entity_name: str, browser: Browser) -> List[Dict[str, Any]]:
    """Search Northdata for company/organization/person and extract result URLs."""
    if not is_northdata_eligible_entity_type(entity_type):
        return []
    
    try:
        # Build search URL
        query_param = urllib.parse.quote(entity_name)
        search_url = f"{NORTHDATA_BASE_URL}/?query={query_param}"
        
        logger.info(f"Searching Northdata for: {entity_name}")
        
        # Fetch search page
        html_content = await fetch_page_content(search_url, browser, timeout=30000)
        if not html_content:
            logger.warning(f"Failed to fetch Northdata search page for {entity_name}")
            return []
        
        # Parse HTML to extract event elements
        tree = html.fromstring(html_content)
        
        # Find all event divs with data-uri attributes
        event_elements = tree.xpath('//div[@class="event" and @data-uri]')
        
        results = []
        for event in event_elements:
            try:
                # Extract data attributes
                data_uri = event.get('data-uri', '')
                data_id = event.get('data-id', '')
                data_score = event.get('data-score', '0')
                
                # Try to extract data-details JSON if available
                data_details_str = event.get('data-details', '{}')
                try:
                    data_details = json.loads(data_details_str.replace('&quot;', '"'))
                except:
                    data_details = {}
                
                # Extract title from the link
                title_elem = event.xpath('.//a[@class="title"]')
                title = title_elem[0].text_content().strip() if title_elem else ""
                
                # Extract extra text (like registration number)
                extra_elem = event.xpath('.//div[@class="extra text"]')
                extra_text = extra_elem[0].text_content().strip() if extra_elem else ""
                
                # Build full URL
                if data_uri:
                    if data_uri.startswith('/'):
                        full_url = f"{NORTHDATA_BASE_URL}{data_uri}"
                    else:
                        full_url = f"{NORTHDATA_BASE_URL}/{data_uri}"
                    
                    results.append({
                        "url": full_url,
                        "title": title,
                        "description": extra_text,
                        "score": float(data_score) if data_score else 0.0,
                        "northdata_id": data_id,
                        "northdata_details": data_details
                    })
            except Exception as e:
                logger.warning(f"Error parsing Northdata event element: {e}")
                continue
        
        # Sort by score (highest first)
        results.sort(key=lambda x: x.get("score", 0), reverse=True)
        
        logger.info(f"Found {len(results)} Northdata results for '{entity_name}'")
        return results
        
    except Exception as e:
        logger.error(f"Error searching Northdata: {e}", exc_info=True)
        return []

async def search_brave(entity_name: str, entity_type: str, count: int = 10) -> List[Dict[str, Any]]:
    """Search Brave Search API with optimized query operators for high-quality results."""
    try:
        # Build optimized query using search operators
        query = build_search_query(entity_name, entity_type)
        logger.debug(f"Search query: {query}")
        
        async with aiohttp.ClientSession() as session:
            params = {
                "q": query,
                "count": count,
                "safesearch": "moderate",  # Filter explicit content
                "spellcheck": 1,  # Auto-correct typos
                "text_decorations": 0,  # Cleaner snippets
            }
            headers = {
                "Accept": "application/json",
                "Accept-Encoding": "gzip",
                "X-Subscription-Token": BRAVE_API_KEY,
            }
            
            async with session.get(BRAVE_SEARCH_URL, params=params, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    # Extract web results
                    results = []
                    if "web" in data and "results" in data["web"]:
                        for result in data["web"]["results"]:
                            results.append({
                                "url": result.get("url", ""),
                                "title": result.get("title", ""),
                                "description": result.get("description", ""),
                            })
                    logger.info(f"Found {len(results)} search results for '{entity_name}'")
                    return results
                else:
                    error_text = await response.text()
                    logger.error(f"Brave API error: {response.status} - {error_text}")
                    return []
    except Exception as e:
        logger.error(f"Error searching Brave: {e}", exc_info=True)
        return []

async def select_interesting_results(entity_name: str, entity_type: str, search_results: List[Dict[str, Any]]) -> List[str]:
    """Use agent to select which search results to visit."""
    if not search_results:
        return []
    
    # Format search results for agent
    results_text = "\n".join([
        f"{i+1}. {r['title']}\n   URL: {r['url']}\n   {r.get('description', '')}"
        for i, r in enumerate(search_results)
    ])
    
    context = f"""
Search results for {entity_name} ({entity_type}):

{results_text}
"""
    
    # Format agent instructions
    formatted_instructions = result_selection_agent.instructions.replace("{entity_name}", entity_name).replace("{entity_type}", entity_type)
    
    temp_agent = Agent(
        name="Search Result Selector",
        instructions=formatted_instructions,
        model=result_selection_agent.model
    )
    
    try:
        result = await Runner.run(temp_agent, context)
        output = result.final_output
        
        if isinstance(output, str):
            try:
                json_str = output.strip()
                if json_str.startswith("```"):
                    lines = json_str.split("\n")
                    json_str = "\n".join(lines[1:-1]) if len(lines) > 2 else json_str
                elif json_str.startswith("```json"):
                    lines = json_str.split("\n")
                    json_str = "\n".join(lines[1:-1]) if len(lines) > 2 else json_str
                
                data = json.loads(json_str)
                selected = SearchResultSelection(**data)
                logger.info(f"Selected {len(selected.selected_urls)} URLs: {selected.reasoning}")
                return selected.selected_urls
            except (json.JSONDecodeError, ValidationError) as e:
                logger.error(f"Failed to parse result selection: {e}\nResponse: {output[:500]}")
                # Fallback: return first 3 results
                return [r["url"] for r in search_results[:3]]
        
        if isinstance(output, SearchResultSelection):
            return output.selected_urls
        
        # Fallback: return first 3 results
        return [r["url"] for r in search_results[:3]]
        
    except Exception as e:
        logger.error(f"Error selecting results: {e}")
        # Fallback: return first 3 results
        return [r["url"] for r in search_results[:3]]

async def enrich_entity_page(entity_id: str, entity_name: str, entity_type: str, page_url: str, browser: Browser, entity_metadata: Optional[Dict[str, Any]] = None, entity_relationships: Optional[List[Dict[str, str]]] = None) -> Optional[EnrichmentResult]:
    """Use the enrichment agent to extract information from a web page."""
    try:
        # Fetch page content
        html = await fetch_page_content(page_url, browser)
        if not html:
            return None
        
        # Convert to markdown and append extracted chart data (e.g., Northdata bar charts)
        chart_snippets = extract_bar_charts(html)
        markdown = convert_to_markdown(html)
        if chart_snippets:
            markdown = f"{markdown}\n\nExtracted Charts:\n{chart_snippets}"
        
        # Build entity context information
        entity_context_parts = [f"Name: {entity_name}", f"Type: {entity_type}"]
        
        # Add metadata if available
        if entity_metadata:
            # Filter out internal fields and format nicely
            relevant_metadata = {k: v for k, v in entity_metadata.items() 
                               if k not in ['id', 'name', 'type', 'source_url', 'enriched_at'] and v is not None}
            if relevant_metadata:
                metadata_str = ", ".join([f"{k}: {v}" for k, v in relevant_metadata.items()])
                entity_context_parts.append(f"Known metadata: {metadata_str}")
        
        # Add relationships if available
        if entity_relationships:
            rel_strs = []
            for rel in entity_relationships[:10]:  # Limit to first 10 relationships
                rel_type = rel.get("type", "related_to")
                related_name = rel.get("related_name", "Unknown")
                related_type = rel.get("related_type", "Entity")
                rel_strs.append(f"{rel_type} -> {related_name} ({related_type})")
            if rel_strs:
                entity_context_parts.append(f"Known relationships: {'; '.join(rel_strs)}")
        
        entity_context = "\n".join(entity_context_parts)
        
        # Prepare context for agent
        context = f"""
Entity to enrich:
{entity_context}

Page URL: {page_url}

Page content:
{markdown[:50000]}  # Limit to avoid token limits
"""
        
        # Format agent instructions with entity details
        formatted_instructions = enrichment_agent.instructions.replace("{entity_name}", entity_name).replace("{entity_type}", entity_type).replace("{page_url}", page_url)
        
        # Create a temporary agent with formatted instructions
        temp_agent = Agent(
            name="Graph Enricher",
            instructions=formatted_instructions,
            model=enrichment_agent.model
        )
        
        # Run agent analysis
        result = await Runner.run(temp_agent, context)
        output = result.final_output
        
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
                # Ensure metadata_updates exists
                if "metadata_updates" not in data:
                    data["metadata_updates"] = None
                # Ensure confidence exists, default to 1.0 if not provided
                if "confidence" not in data:
                    data["confidence"] = 1.0
                # Filter out low-confidence enrichments (< 0.7)
                confidence = data.get("confidence", 1.0)
                if confidence < 0.7:
                    logger.warning(f"Low confidence ({confidence:.2f}) enrichment for {entity_name} from {page_url}, skipping")
                    return None
                return EnrichmentResult(**data)
            except (json.JSONDecodeError, ValidationError) as e:
                logger.error(f"Failed to parse enrichment result: {e}\nResponse: {output[:500]}")
                return None
        
        if isinstance(output, EnrichmentResult):
            return output
        
        logger.warning(f"Unexpected output type from enrichment agent: {type(output)}")
        return None
        
    except Exception as e:
        logger.error(f"Error enriching entity {entity_name} from {page_url}: {e}", exc_info=True)
        return None

def ingest_enrichment(driver, entity_id: str, enrichment: EnrichmentResult, source_url: Optional[str] = None):
    """Ingest enrichment results into Neo4j."""
    with driver.session() as sess:
        # Update metadata on the original entity
        if enrichment.metadata_updates:
            set_clauses = []
            params = {"id": entity_id}
            for key, value in enrichment.metadata_updates.items():
                safe_key = "".join(c if c.isalnum() or c == "_" else "_" for c in key)
                set_clauses.append(f"n.{safe_key} = ${safe_key}")
                # Convert value to string if it's not already
                if isinstance(value, (list, dict)):
                    params[safe_key] = json.dumps(value)
                elif value is not None:
                    params[safe_key] = str(value)
                else:
                    params[safe_key] = None
            
            if set_clauses:
                query = f"""
                MATCH (n {{id: $id}})
                SET {', '.join(set_clauses)}
                """
                sess.run(query, **params)
        
        # Add new entities
        for ent in enrichment.entities:
            trimmed_name = ent.name.strip()
            entity_id_new = generate_entity_id(trimmed_name)
            label = ent.type.strip()
            safe_label = "".join(c if c.isalnum() or c == "_" else "_" for c in label)
            
            set_clauses = ["n.name = $name", "n.type = $type"]
            params = {"id": entity_id_new, "name": trimmed_name, "type": label}
            
            # Extract source_url from metadata if present, otherwise use parameter
            source_url_value = None
            if ent.metadata and "source_url" in ent.metadata:
                source_url_value = ent.metadata.get("source_url")
            elif source_url:
                source_url_value = source_url
            
            # Add source_url property (like page_worker.py does)
            if source_url_value:
                set_clauses.append("n.source_url = $source_url")
                params["source_url"] = source_url_value
            else:
                # Set empty string if no source_url available
                set_clauses.append("n.source_url = $source_url")
                params["source_url"] = ""
            
            if ent.metadata:
                for key, value in ent.metadata.items():
                    # Skip source_url as it's already handled above as a property
                    if key == "source_url":
                        continue
                    safe_key = "".join(c if c.isalnum() or c == "_" else "_" for c in key)
                    set_clauses.append(f"n.{safe_key} = ${safe_key}")
                    # Convert value to string if it's not already
                    if isinstance(value, (list, dict)):
                        params[safe_key] = json.dumps(value)
                    elif value is not None:
                        params[safe_key] = str(value)
                    else:
                        params[safe_key] = None
            
            query = f"""
            MERGE (n {{id: $id}})
            SET n:{safe_label}
            SET {', '.join(set_clauses)}
            """
            sess.run(query, **params)
        
        # Add new relationships
        for rel in enrichment.relations:
            # Find entity IDs by name (normalized)
            origin_normalized = normalize_name(rel.origin)
            dest_normalized = normalize_name(rel.destination)
            
            # Get IDs
            get_ids_query = """
            MATCH (n)
            WHERE toLower(trim(n.name)) = $name
            RETURN n.id AS id
            LIMIT 1
            """
            origin_result = sess.run(get_ids_query, name=origin_normalized).single()
            dest_result = sess.run(get_ids_query, name=dest_normalized).single()
            
            if origin_result and dest_result:
                origin_id = origin_result["id"]
                dest_id = dest_result["id"]
                safe_rel_type = "".join(c if c.isalnum() or c == "_" else "_" for c in rel.type.strip())
                
                # Check if relationship already exists
                check_query = f"""
                MATCH (a {{id: $origin_id}})-[r]->(b {{id: $dest_id}})
                WHERE type(r) = $rel_type
                RETURN count(r) AS exists
                """
                exists = sess.run(check_query, origin_id=origin_id, dest_id=dest_id, rel_type=safe_rel_type).single()["exists"]
                
                if exists == 0:
                    # Create relationship - Neo4j will optimize this internally
                    # The Cartesian product warning is harmless for small datasets with indexed IDs
                    create_rel_query = f"""
                    MATCH (a {{id: $origin_id}})
                    MATCH (b {{id: $dest_id}})
                    CREATE (a)-[r:`{safe_rel_type}`]->(b)
                    """
                    sess.run(create_rel_query, origin_id=origin_id, dest_id=dest_id)
        
        logger.info(f"Enriched entity {entity_id}: added {len(enrichment.entities)} entities, {len(enrichment.relations)} relations")

def get_nodes_to_enrich(driver, limit: int = 10) -> List[Dict[str, Any]]:
    """Get nodes that could benefit from enrichment, including their relationships."""
    with driver.session() as sess:
        # Prioritize nodes with source_url but limited metadata/relationships
        # Only include enrichment-eligible entity types
        query = """
        MATCH (n)
        WHERE n.source_url IS NOT NULL
          AND n.name IS NOT NULL
          AND (n.enriched_at IS NULL OR n.enriched_at < timestamp() - 86400000)
          AND ANY(label IN labels(n) WHERE label IN ['Company', 'Organization', 'Person', 'EmailAddress', 'Address', 'Institute', 'Event'])
        WITH n, COUNT { (n)--() } AS rel_count
        ORDER BY rel_count ASC, n.name ASC
        LIMIT $limit
        RETURN n.id AS id, n.name AS name, labels(n) AS labels,
               n.source_url AS source_url, properties(n) AS properties
        """
        nodes = []
        for record in sess.run(query, limit=limit):
            node_data = dict(record)
            entity_id = node_data["id"]
            
            # Get related nodes and relationships
            rel_query = """
            MATCH (n {id: $id})-[r]-(related)
            RETURN type(r) AS rel_type, 
                   labels(related) AS related_labels,
                   related.name AS related_name,
                   related.id AS related_id
            LIMIT 20
            """
            relationships = []
            for rel_record in sess.run(rel_query, id=entity_id):
                rel_data = dict(rel_record)
                relationships.append({
                    "type": rel_data.get("rel_type", ""),
                    "related_name": rel_data.get("related_name", ""),
                    "related_type": rel_data.get("related_labels", [""])[0] if rel_data.get("related_labels") else "Entity"
                })
            
            node_data["relationships"] = relationships
            nodes.append(node_data)
        
        return nodes

async def enrich_batch(driver, browser: Browser, batch_size: int = 5):
    """Enrich a batch of nodes."""
    nodes = get_nodes_to_enrich(driver, limit=batch_size)
    
    if not nodes:
        logger.info("No nodes found that need enrichment")
        return
    
    logger.info(f"Enriching {len(nodes)} nodes...")
    
    for node in nodes:
        entity_id = node["id"]
        entity_name = node["name"]
        entity_type = node["labels"][0] if node["labels"] else "Entity"
        
        # Check if entity type is eligible for enrichment
        if not is_enrichment_eligible_entity_type(entity_type):
            logger.debug(f"Skipping {entity_name} ({entity_type}) - not an enrichment-eligible type")
            continue
        
        logger.info(f"Searching for enrichment opportunities for {entity_name} ({entity_type})")
        
        try:
            # Check if this entity type is eligible for Northdata lookup
            use_northdata = is_northdata_eligible_entity_type(entity_type)
            
            # Collect URLs from multiple sources
            all_urls_to_visit = []
            
            # 1. Northdata lookup (for companies/organizations/people)
            if use_northdata:
                logger.info(f"Performing Northdata lookup for {entity_name} ({entity_type})")
                northdata_results = await search_northdata(entity_name, browser)
                
                if northdata_results:
                    # Select top 2-3 Northdata results (highest scored)
                    top_northdata = northdata_results[:3]
                    for result in top_northdata:
                        all_urls_to_visit.append({
                            "url": result["url"],
                            "source": "northdata",
                            "title": result.get("title", ""),
                            "description": result.get("description", "")
                        })
                    logger.info(f"Added {len(top_northdata)} Northdata URLs for {entity_name}")
                
                # Rate limiting after Northdata search
                await asyncio.sleep(2)
            
            # 2. Brave Search (for all entity types)
            search_results = await search_brave(entity_name, entity_type, count=10)
            
            if search_results:
                logger.info(f"Found {len(search_results)} Brave search results for {entity_name}")
                
                # Select interesting results to visit
                selected_urls = await select_interesting_results(entity_name, entity_type, search_results)
                
                for url in selected_urls:
                    all_urls_to_visit.append({
                        "url": url,
                        "source": "brave",
                        "title": "",
                        "description": ""
                    })
            
            if not all_urls_to_visit:
                logger.warning(f"No URLs found to visit for {entity_name}")
                continue
            
            logger.info(f"Will visit {len(all_urls_to_visit)} URLs for {entity_name}")
            
            # Enrich from each selected URL
            all_entities = []
            all_relations = []
            all_metadata = {}
            enrichment_urls = []  # Track URLs for each enrichment
            
            # Get entity metadata and relationships for context
            entity_metadata = node.get("properties", {})
            entity_relationships = node.get("relationships", [])
            
            for url_info in all_urls_to_visit:
                url = url_info["url"]
                source = url_info.get("source", "unknown")
                logger.info(f"Enriching {entity_name} from {url} (source: {source})")
                
                enrichment = await enrich_entity_page(
                    entity_id, entity_name, entity_type, url, browser,
                    entity_metadata=entity_metadata,
                    entity_relationships=entity_relationships
                )
                
                if enrichment:
                    confidence = enrichment.confidence or 1.0
                    logger.info(f"Enrichment confidence: {confidence:.2f} for {entity_name} from {url}")
                    # Store entities with their source URL in metadata
                    for ent in enrichment.entities:
                        # Add source_url to entity metadata if not already present
                        if not ent.metadata:
                            ent.metadata = {}
                        if "source_url" not in ent.metadata:
                            ent.metadata["source_url"] = url
                        all_entities.append(ent)
                    all_relations.extend(enrichment.relations)
                    if enrichment.metadata_updates:
                        all_metadata.update(enrichment.metadata_updates)
                    enrichment_urls.append(url)
                
                # Rate limiting - wait between page visits
                await asyncio.sleep(2)
            
            # Combine all enrichments and ingest
            if all_entities or all_relations or all_metadata:
                combined_enrichment = EnrichmentResult(
                    entities=all_entities,
                    relations=all_relations,
                    metadata_updates=all_metadata if all_metadata else None,
                    confidence=None  # Combined enrichment doesn't need confidence
                )
                # Use first enrichment URL as source (or combine if needed)
                primary_source_url = enrichment_urls[0] if enrichment_urls else None
                ingest_enrichment(driver, entity_id, combined_enrichment, source_url=primary_source_url)
                
                # Mark as enriched
                with driver.session() as sess:
                    mark_query = """
                    MATCH (n {id: $id})
                    SET n.enriched_at = timestamp()
                    """
                    sess.run(mark_query, id=entity_id)
                
                logger.info(f"Successfully enriched {entity_name} with {len(all_entities)} entities, {len(all_relations)} relations")
            else:
                logger.warning(f"No enrichment found for {entity_name} from selected URLs")
                
        except Exception as e:
            logger.error(f"Error enriching {entity_name}: {e}", exc_info=True)
        
        # Rate limiting - wait between entities
        await asyncio.sleep(3)

async def run_enrichment():
    """Main enrichment function."""
    logger.info("Starting graph enrichment run…")
    driver = None
    
    # Establish connection to Neo4j
    for attempt in range(12):
        try:
            driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
            driver.verify_connectivity()
            logger.info("Successfully connected to Neo4j for enrichment.")
            break
        except Exception as e:
            logger.warning(f"Enricher: Neo4j not available yet, retrying... (attempt {attempt+1}/12)")
            await asyncio.sleep(5)
            driver = None
    
    if not driver:
        logger.error("Unable to connect to Neo4j after multiple attempts. Skipping this enrichment run.")
        return
    
    # Launch browser
    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=True,
            args=[
                '--no-sandbox',
                '--disable-setuid-sandbox',
                '--disable-dev-shm-usage',
                '--disable-blink-features=AutomationControlled',
            ]
        )
        
        try:
            # Enrich a batch of nodes
            await enrich_batch(driver, browser, batch_size=10)
        finally:
            await browser.close()
            if driver:
                driver.close()

def main():
    # Schedule to run every 4 hours
    schedule.every(4).hours.do(lambda: asyncio.run(run_enrichment()))
    logger.info("Enricher started – scheduled to run every 4 hours")
    # Immediate first run
    asyncio.run(run_enrichment())
    
    while True:
        schedule.run_pending()
        time.sleep(60)

if __name__ == "__main__":
    main()

