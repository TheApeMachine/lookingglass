import os
import logging
import time
import itertools
import warnings
from minio import Minio
from gliner import GLiNER
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from neo4j import GraphDatabase
from html_to_markdown import convert_to_markdown
import re
alphabets= "([A-Za-z])"
prefixes = "(Mr|St|Mrs|Ms|Dr)[.]"
suffixes = "(Inc|Ltd|Jr|Sr|Co)"
starters = "(Mr|Mrs|Ms|Dr|Prof|Capt|Cpt|Lt|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
websites = "[.](com|net|org|io|gov|edu|me)"
digits = "([0-9])"
multiple_dots = r'\.{2,}'


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("page-worker")
warnings.filterwarnings("ignore", message="Sentence of length .* has been truncated")

# Environment variables
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "minio:9000")
MINIO_USER = os.getenv("MINIO_USER", "miniouser")
MINIO_PASSWORD = os.getenv("MINIO_PASSWORD", "miniopassword")
MINIO_BUCKET = os.getenv("MINIO_BUCKET", "scraped")

NEO4J_HOST = os.getenv("NEO4J_HOST", "neo4j")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")
NEO4J_URI = f"bolt://{NEO4J_HOST}:7687"

# Clients
minio_client = Minio(MINIO_ENDPOINT, access_key=MINIO_USER, secret_key=MINIO_PASSWORD, secure=False)
# Neo4j driver (retry because the DB container may not be ready immediately)
neo_driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
for attempt in range(12):  # retry for ~1 minute
    try:
        neo_driver.verify_connectivity()
        logger.info("Connected to Neo4j")
        break
    except Exception as e:
        logger.warning(f"Neo4j not available yet (attempt {attempt + 1}/12): {e}")
        time.sleep(5)
else:
    logger.error("Unable to connect to Neo4j after multiple attempts.")
    raise

# Models (lazy)
_ner_model = None
_re_model = None
_re_tokenizer = None

def get_ner_model():
    global _ner_model
    if _ner_model is None:
        _ner_model = GLiNER.from_pretrained("numind/NuNER_Zero")
    return _ner_model

def split_into_sentences(text: str) -> list[str]:
    """
    Split the text into sentences.

    If the text contains substrings "<prd>" or "<stop>", they would lead 
    to incorrect splitting because they are used as markers for splitting.

    :param text: text to be split into sentences
    :type text: str

    :return: list of sentences
    :rtype: list[str]
    """
    text = " " + text + "  "
    text = text.replace("\n"," ")
    text = re.sub(prefixes,"\\1<prd>",text)
    text = re.sub(websites,"<prd>\\1",text)
    text = re.sub(digits + "[.]" + digits,"\\1<prd>\\2",text)
    text = re.sub(multiple_dots, lambda match: "<prd>" * len(match.group(0)) + "<stop>", text)
    if "Ph.D" in text: text = text.replace("Ph.D.","Ph<prd>D<prd>")
    text = re.sub("\s" + alphabets + "[.] "," \\1<prd> ",text)
    text = re.sub(acronyms+" "+starters,"\\1<stop> \\2",text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>\\3<prd>",text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>",text)
    text = re.sub(" "+suffixes+"[.] "+starters," \\1<stop> \\2",text)
    text = re.sub(" "+suffixes+"[.]"," \\1<prd>",text)
    text = re.sub(" " + alphabets + "[.]"," \\1<prd>",text)
    if "”" in text: text = text.replace(".”","”.")
    if "\"" in text: text = text.replace(".\"","\".")
    if "!" in text: text = text.replace("!\"","\"!")
    if "?" in text: text = text.replace("?\"","\"?")
    text = text.replace(".",".<stop>")
    text = text.replace("?","?<stop>")
    text = text.replace("!","!<stop>")
    text = text.replace("<prd>",".")
    sentences = text.split("<stop>")
    sentences = [s.strip() for s in sentences]
    if sentences and not sentences[-1]: sentences = sentences[:-1]
    return sentences

def get_re_model():
    global _re_model, _re_tokenizer
    if _re_model is None:
        model_name = "Babelscape/rebel-large"
        _re_tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        _re_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return _re_model, _re_tokenizer

def _parse_rebel_span(text: str):
    triples = []
    current = {"subj": "", "rel": "", "obj": "", "cur": None}
    for tok in text.split():
        if tok == "<triplet>":
            if current["subj"]:
                triples.append((current["subj"].strip(), current["rel"].strip(), current["obj"].strip()))
            current = {"subj": "", "rel": "", "obj": "", "cur": "subj"}
        elif tok == "<subj>":
            current["cur"] = "subj"
        elif tok == "<obj>":
            current["cur"] = "obj"
        elif tok == "<rel>":
            current["cur"] = "rel"
        else:
            if current["cur"]:
                current[current["cur"]] += " " + tok
    if current["subj"]:
        triples.append((current["subj"].strip(), current["rel"].strip(), current["obj"].strip()))
    return triples

def extract_relations(text: str, max_len: int = 512):
    model, tok = get_re_model()
    triples_all = []
    for i in range(0, len(text), max_len):
        chunk = text[i:i+max_len]
        inputs = tok(chunk, return_tensors="pt", truncation=True, max_length=512)
        outputs = model.generate(**inputs, max_length=512, num_beams=3)
        decoded = tok.batch_decode(outputs, skip_special_tokens=False)[0]
        triples_all.extend(_parse_rebel_span(decoded))
    return triples_all

REL_MAP = {
    "employee": "EMPLOYEE_OF",
    "works_for": "EMPLOYEE_OF",
    "employment": "EMPLOYEE_OF",
    "founder": "FOUNDER_OF",
    "founded_by": "FOUNDER_OF",
    "founder_of": "FOUNDER_OF",
    "partner": "PARTNER_OF",
    "partner_of": "PARTNER_OF",
    "affiliate": "PARTNER_OF",
}

# Accept the original text so we can extract spans without relying on a global variable
def merge_entities(entities, text: str):
    if not entities:
        return []
    merged = []
    current = entities[0]
    for next_entity in entities[1:]:
        if next_entity['label'] == current['label'] and (next_entity['start'] == current['end'] + 1 or next_entity['start'] == current['end']):
            current['text'] = text[current['start']: next_entity['end']].strip()
            current['end'] = next_entity['end']
        else:
            merged.append(current)
            current = next_entity
    # Append the last entity
    merged.append(current)
    return merged

def chunk_by_tokens(sentences, tokenizer, max_tok=384):
    """
    Generator to chunk sentences by token count.
    """
    buf, cur_len = [], 0
    for s in sentences:
        tokens = tokenizer.encode(s, add_special_tokens=False)
        if cur_len + len(tokens) > max_tok and buf:
            yield " ".join(buf)
            buf, cur_len = [], 0
        buf.append(s)
        cur_len += len(tokens)
    if buf:
        yield " ".join(buf)


def process_page_object(object_name: str):
    try:
        with minio_client.get_object(MINIO_BUCKET, object_name) as response:
            html_bytes = response.read()
        text = html_bytes.decode("utf-8", errors="ignore")
        text = convert_to_markdown(text)

        ner = get_ner_model()
        ner_tokenizer = ner.tokenizer
        labels = ["person", "organization", "location", "project", "initiative", "job"]
        labels = [l.lower() for l in labels]

        sentences = split_into_sentences(text)

        found_entities = []

        for chunk_text in chunk_by_tokens(sentences, ner_tokenizer, max_tok=384):
            try:
                entities = ner.predict_entities(chunk_text, labels, flat_ner=True, threshold=0.3)
                # Merge adjacent token-level entities to form complete multi-word entities
                entities = merge_entities(entities, chunk_text)

                for entity in entities:
                    found_entities.append(entity)
            except Exception as e:
                logger.warning(f"NER failed on chunk: {e}")

        triples = extract_relations(text)

        with neo_driver.session() as sess:
            # Batch-create entities
            entity_map = {
                "person": "Person",
                "organization": "Organization",
                "job": "Job",
                "project": "Project",
                "location": "Location"
            }
            for label, node_label in entity_map.items():
                entities_to_create = list({e["text"] for e in found_entities if e["label"] == label})
                if entities_to_create:
                    sess.run(f"UNWIND $entities AS e MERGE (n:{node_label}:Entity {{name: e}})", entities=entities_to_create)

            # Allow only triples whose subject AND object were detected by NuNER.
            valid_entities = {e["text"] for e in found_entities}

            organization_entities = {e["text"] for e in found_entities if e["label"] == "organization"}
            
            pairs = list(itertools.combinations(organization_entities, 2))
            if pairs:
                sess.run(
                    """
                    UNWIND $pairs AS pair
                    MERGE (o1:Organization {name: pair[0]})
                    MERGE (o2:Organization {name: pair[1]})
                    MERGE (o1)-[r:SOFT_RELATED]-(o2)
                    ON CREATE SET r.shared_count = 1
                    ON MATCH  SET r.shared_count = coalesce(r.shared_count, 0) + 1
                    """,
                    pairs=pairs,
                )

            # Batch-create relationships
            valid_triples_by_type = {}
            for subj, rel, obj in triples:
                if subj in valid_entities and obj in valid_entities:
                    edge = REL_MAP.get(rel.lower())
                    if edge:
                        if edge not in valid_triples_by_type:
                            valid_triples_by_type[edge] = []
                        valid_triples_by_type[edge].append({'subj': subj, 'obj': obj})
            
            for edge_type, batch in valid_triples_by_type.items():
                sess.run(
                    f"""
                    UNWIND $batch as pair
                    MERGE (a:Entity {{name: pair.subj}})
                    MERGE (b:Entity {{name: pair.obj}})
                    MERGE (a)-[:`{edge_type}`]->(b)
                    """,
                    batch=batch
                )

    except Exception as e:
        logger.error(f"Error processing {object_name}: {e}", exc_info=True)


def main():
    # Ensure bucket exists
    if not minio_client.bucket_exists(MINIO_BUCKET):
        logger.error(f"Bucket {MINIO_BUCKET} does not exist")
        return

    logger.info("Page worker listening for new HTML objects…")
    while True:
        try:
            # Listen for new HTML uploads under pages/ prefix
            events = minio_client.listen_bucket_notification(
                MINIO_BUCKET,
                prefix="pages/",
                suffix=".html",
                events=["s3:ObjectCreated:*"],
            )
            for notification in events:
                for record in notification.get("Records", []):
                    obj_name = record["s3"]["object"]["key"]
                    logger.info(f"New HTML object: {obj_name}")
                    process_page_object(obj_name)
        except Exception as e:
            logger.error(f"Listener error: {e}", exc_info=True)
            time.sleep(5)

if __name__ == "__main__":
    main() 