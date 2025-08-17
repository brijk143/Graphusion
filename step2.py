#!/usr/bin/env python3
import os
import sys
import json
import logging
from collections import Counter
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from tqdm import tqdm

# Load environment variables from .env file (if present)
load_dotenv()

# Read and validate API key at the very start
google_api_key = os.getenv("GOOGLE_API_KEY")
if not google_api_key:
    sys.exit("ERROR: Please set the GOOGLE_API_KEY environment variable (or add it to a .env file).")

# Configure logging early
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s"
)

# === Hardcoded paths & parameters ===
# Define the base output directory
OUTPUT_BASE_DIR = "output"

INPUT_CONCEPTS_PATH  = os.path.join(OUTPUT_BASE_DIR, "seed_entities.txt") # Query concepts
INPUT_ABSTRACTS_PATH = "abstract.txt"                              # Text content to analyze
# Construct the full path for the output file within the output directory
OUTPUT_FILE          = os.path.join(OUTPUT_BASE_DIR, "candidate_triples.jsonl")
PROMPT_PATH          = os.path.join("prompts", "prompts_step2.txt") # Fix for SyntaxWarning and cross-platform
MODEL_NAME           = "gemini-2.0-flash"
MAX_INPUT_CHAR       = 10000  # max characters of content per API call

# Hardcoded relation definitions - 7 predefined types
RELATION_DEFS = {
    "Compare": {
        "label": "Compare",
        "description": "Represents a relationship between two or more entities where a comparison is being made. For example, \"A is larger than B\" or \"X is more efficient than Y.\" (Non-directional)"
    },
    "Part-of": {
        "label": "Part-of",
        "description": "Denotes a relationship where one entity is a constituent or component of another. For instance, \"Wheel is a part of a Car.\" (Directional)"
    },
    "Conjunction": {
        "label": "Conjunction",
        "description": "Indicates a logical or semantic relationship where two or more entities are connected to form a group or composite idea. For example, \"Salt and Pepper.\" (Non-directional)"
    },
    "Evaluate-for": {
        "label": "Evaluate-for",
        "description": "Represents an evaluative relationship where one entity is assessed in the context of another. For example, \"A tool is evaluated for its effectiveness.\" (Directional)"
    },
    "Is-a-Prerequisite-of": {
        "label": "Is-a-Prerequisite-of",
        "description": "This dual-purpose relationship implies that one entity is either a characteristic of another or a required precursor for another. For instance, \"The ability to code is a prerequisite of software development.\" (Directional)"
    },
    "Used-for": {
        "label": "Used-for",
        "description": "Denotes a functional relationship where one entity is utilized in accomplishing or facilitating the other. For example, \"A hammer is used for driving nails.\" (Directional)"
    },
    "Hyponym-Of": {
        "label": "Hyponym-Of",
        "description": "Establishes a hierarchical relationship where one entity is a more specific version or subtype of another. For instance, \"A Sedan is a hyponym of a Car.\" (Directional)"
    }
}

def get_message_type(msg):
    """Helper function to get the message type as a string"""
    if isinstance(msg, SystemMessage):
        return "SYSTEM"
    elif isinstance(msg, HumanMessage):
        return "USER"
    elif isinstance(msg, AIMessage):
        return "ASSISTANT"
    else:
        return "UNKNOWN"

def load_abstracts():
    """Load abstracts/content from file"""
    if not os.path.exists(INPUT_ABSTRACTS_PATH):
        logging.error(f"Abstracts file not found: {INPUT_ABSTRACTS_PATH}")
        logging.info("Please create an abstracts file with the content to analyze")
        return None
    
    with open(INPUT_ABSTRACTS_PATH, 'r', encoding='utf-8') as f:
        content = f.read().strip()
    
    if not content:
        logging.error("Abstracts file is empty")
        return None
    
    logging.info(f"Loaded {len(content)} characters from abstracts file")
    return content

def chunk_content(content, max_chars):
    """Split content into chunks that fit within the character limit"""
    chunks = []
    words = content.split()
    current_chunk = []
    current_length = 0
    
    for word in words:
        word_length = len(word) + 1  # +1 for space
        if current_length + word_length > max_chars and current_chunk:
            chunks.append(' '.join(current_chunk))
            current_chunk = [word]
            current_length = word_length
        else:
            current_chunk.append(word)
            current_length += word_length
    
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks

def parse_triplet_output(text):
    """Parse the triplet output format: (concept, relation, concept)(concept, relation, concept)"""
    triplets = []
    if not text or text.lower().strip() == "none":
        return triplets
    
    # Try to parse as JSON first (in case model returns JSON)
    try:
        json_data = json.loads(text)
        if isinstance(json_data, list):
            for item in json_data:
                if isinstance(item, dict) and 's' in item and 'p' in item and 'o' in item:
                    triplets.append({
                        's': item['s'],
                        'p': item['p'],
                        'o': item['o']
                    })
        return triplets
    except json.JSONDecodeError:
        pass
    
    # Parse the expected format: (concept, relation, concept)(concept, relation, concept)
    import re
    pattern = r'\(([^,]+),\s*([^,]+),\s*([^)]+)\)'
    matches = re.findall(pattern, text)
    
    for match in matches:
        head, relation, tail = [item.strip() for item in match]
        triplets.append({
            's': head,
            'p': relation,
            'o': tail
        })
    
    return triplets

def extract_candidate_triples():
    logging.info("Starting Step 2: Candidate Triple Extraction with Gemini API")

    # Initialize the Gemini model client
    model = ChatGoogleGenerativeAI(
        model=MODEL_NAME,
        temperature=0.1,
        google_api_key=google_api_key
    )

    # Load query concepts (seed entities)
    # Check if the input concepts file exists
    if not os.path.exists(INPUT_CONCEPTS_PATH):
        logging.error(f"Input concepts file not found: {INPUT_CONCEPTS_PATH}")
        logging.info("Please ensure step1.py has run successfully to generate this file.")
        return

    with open(INPUT_CONCEPTS_PATH, 'r', encoding='utf-8') as f:
        query_concepts = [line.strip().strip('"') for line in f if line.strip()]

    # Load abstracts/content
    all_content = load_abstracts()
    if not all_content:
        return
    
    # Load the prompt template
    if not os.path.exists(PROMPT_PATH):
        logging.error(f"Prompt file not found: {PROMPT_PATH}")
        logging.info("Please ensure 'prompts/prompts_step2.txt' exists.")
        return

    prompt_txt = open(PROMPT_PATH, 'r', encoding='utf-8').read()
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", "You are a knowledge graph builder using Google Gemini."),
        ("user", prompt_txt)
    ])

    # Ensure output directory exists before writing files
    # This will create 'output/' if it doesn't exist
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    extracted_counts = []
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as outp:
        for query_concept in tqdm(query_concepts, desc="Processing query concepts"):
            # Chunk the content to fit within limits
            content_chunks = chunk_content(all_content, MAX_INPUT_CHAR)
            
            for i, content_chunk in enumerate(content_chunks):
                relation_defs_text = "\n".join(
                    f"{r}: {RELATION_DEFS[r]['description']}" for r in RELATION_DEFS
                )

                # Fill in the prompt with the current query concept and content chunk
                invocation = prompt_template.invoke({
                    "query_concept": query_concept,            # The specific concept we're querying for
                    "content": content_chunk,                  # The text content to analyze
                    "relation_definitions": relation_defs_text
                })

                # DEBUG: print out exactly what goes to Gemini (only for first concept, first chunk)
                if query_concept == query_concepts[0] and i == 0:
                    print("\n=== Filled Prompt (First Query Concept, First Chunk) ===")
                    for msg in invocation.to_messages():
                        msg_type = get_message_type(msg)
                        print(f"{msg_type}: {msg.content[:500]}{'...' if len(msg.content) > 500 else ''}\n")

                # Call Gemini
                try:
                    response = model.invoke(invocation)
                    text = getattr(response, "content", "").strip()
                except Exception as e:
                    logging.error(f"Error calling Gemini for concept '{query_concept}', chunk {i}: {e}")
                    continue

                if not text or text.lower() == "none":
                    continue

                # Parse the triplet output
                triplets = parse_triplet_output(text)
                
                if not triplets:
                    logging.warning(f"No valid triplets parsed for '{query_concept}', chunk {i}: {text}")
                    continue

                # Write each extracted triple
                for triplet in triplets:
                    rel = triplet.get('p')
                    if rel not in RELATION_DEFS:
                        logging.warning(f"Unknown relation '{rel}' - skipping triplet")
                        continue
                    
                    # Add metadata
                    triplet['query_concept'] = query_concept
                    triplet['content_chunk'] = i
                    
                    outp.write(json.dumps(triplet) + "\n")
                    extracted_counts.append(rel)

    logging.info(f"Extracted {len(extracted_counts)} triples to {OUTPUT_FILE}")
    logging.info(f"Triple counts by relation: {Counter(extracted_counts)}")


if __name__ == "__main__":
    extract_candidate_triples()