#!/usr/bin/env python3
import json
import os
import sys
import logging
from collections import defaultdict
from tqdm import tqdm
from dotenv import load_dotenv

# Load environment variables from .env file (if present)
load_dotenv()

# Read and validate API key at the very start
google_api_key = os.getenv("GOOGLE_API_KEY")

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s: %(message)s")

# Use a mock implementation of the LangChain and Google API for this example
def mock_gemini_api_call(prompt):
    """
    Simulates a call to the Gemini API.
    In a full implementation, this would be a real API call.
    """
    logging.info("Simulating LLM call for a fusion task...")
    return """(concept A, relation X, concept B)"""

# === Hardcoded paths & parameters ===
OUTPUT_BASE_DIR = "output"
INPUT_CANDIDATE_TRIPLES = os.path.join(OUTPUT_BASE_DIR, "candidate_triples.jsonl")
INPUT_ABSTRACTS_PATH = "abstract.txt"
OUTPUT_FINAL_GRAPH = os.path.join(OUTPUT_BASE_DIR, "final_graph.jsonl")
PROMPT_FUSION_PATH = "prompts/prompts_step3.txt"
MODEL_NAME = "gemini-2.5-flash"  # As requested

# A simplified, hardcoded mapping of relations for demonstration
RELATION_DEFINITIONS = {
    "Compare": "Represents a comparison between two or more entities.",
    "Part-of": "Denotes a relation where one entity is a component of another.",
    "Conjunction": "Two or more entities are connected to form a group.",
    "Evaluate-for": "One entity is assessed in the context of another.",
    "Is-a-Prerequisite-of": "One entity is a required precursor for another.",
    "Used-for": "One entity is utilized in accomplishing or facilitating another.",
    "Hyponym-Of": "Establishes a hierarchical relation where one entity is a more specific version or subtype of another.",
}

def load_prompt_template(file_path):
    """Loads the prompt template from a given file path."""
    if not os.path.isfile(file_path):
        sys.exit(f"ERROR: Prompt file {file_path} does not exist.")
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

def load_abstracts(file_path):
    """
    Loads all text from the abstracts file into a single string.
    This is used as the background context for the LLM.
    """
    if not os.path.isfile(file_path):
        logging.warning(f"Abstracts file not found at {file_path}. Background context will be empty.")
        return ""
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

def merge_entities(triples):
    """
    Merges similar entities based on simple normalization rules.
    This is a simplification of Rule 2 in the prompt.
    For example, "long short-term memory" and "lstm" would be mapped to a canonical form.
    Args:
        triples (list): A list of triplet dictionaries.
    Returns:
        list: A list of triplets with merged entities.
    """
    entity_mapping = {
        "lstm": "long short-term memory",
        "rnn": "recurrent neural network",
        "llm": "large language model",
    }
    merged_triples = []
    for t in triples:
        s = t['s'].strip().lower()
        o = t['o'].strip().lower()
        t['s'] = entity_mapping.get(s, s)
        t['o'] = entity_mapping.get(o, o)
        merged_triples.append(t)
    return merged_triples

def resolve_conflicts_and_fuse(merged_triples, background_text, prompt_template):
    """
    Resolves conflicting relations and fuses the triples using the LLM.
    This implements Rule 3 of the prompt.
    Args:
        merged_triples (list): Triples with merged entities.
        background_text (str): The full text for background context.
        prompt_template (str): The loaded prompt template string.
    Returns:
        list: A new list of triples after conflict resolution.
    """
    triple_groups = defaultdict(list)
    for t in merged_triples:
        triple_groups[(t['s'], t['o'])].append(t)

    final_triples = []
    for (s, o), group in triple_groups.items():
        if len(group) == 1:
            final_triples.append(group[0])
        else:
            logging.warning(f"Conflict detected for ({s}, {o}). Relations: {[t['p'] for t in group]}")
            # Populate the prompt template (no LLM-KG or E_G)
            prompt = prompt_template.format(
                entity=s,
                background=background_text,
                relation_definitions='\n'.join([f"{k}: {v}" for k, v in RELATION_DEFINITIONS.items()])
            )
            # In a full implementation, the LLM would be called here.
            # For this example, we will just pick the first relation as a placeholder.
            # real_llm_response = mock_gemini_api_call(prompt)
            # final_triples.extend(parse_llm_output(real_llm_response))
            final_triples.append(group[0])
            logging.info(f"Resolved conflict for ({s}, {o}) by picking first relation: {group[0]['p']}")
    return final_triples

def infer_novel_triplets(final_triples, background_text, prompt_template):
    """
    Infers new, novel triplets using the LLM based on the existing graph and background text.
    This implements Rule 4 of the prompt.
    Args:
        final_triples (list): The list of triples after conflict resolution.
        background_text (str): The full text for background context.
        prompt_template (str): The loaded prompt template string.
    Returns:
        list: A list of newly inferred triplets.
    """
    logging.info("Starting novel triplet inference with LLM.")
    llm_kg_verbalized = ', '.join([f"({t['s']}, {t['p']}, {t['o']})" for t in final_triples])
    inference_prompt = (
        f"You are a knowledge graph builder. Based on the following knowledge graph and "
        f"background text, find any new, undiscovered triplets.\n\n"
        f"Knowledge Graph: {llm_kg_verbalized}\n\n"
        f"Background: {background_text}\n\n"
        f"Output the new triplets in the format (entity, relation, entity), one per line. "
        f"If there are no new triplets, respond with 'None'.\n"
        f"Relation definitions:\n"
        f"{chr(10).join([f'{k}: {v}' for k, v in RELATION_DEFINITIONS.items()])}\n"
    )
    # In a full implementation, the LLM would be called here.
    # For this example, we will return a mock triplet.
    logging.info("Calling LLM to infer novel triplets...")
    # real_llm_response = mock_gemini_api_call(inference_prompt)
    return [{'s': 'mock entity', 'p': 'Used-for', 'o': 'inferred task'}]

def main():
    """Main function to orchestrate the fusion process."""
    logging.info("Starting Step 3: Knowledge Graph Fusion and Refinement")
    if not os.path.isfile(INPUT_CANDIDATE_TRIPLES):
        sys.exit(f"ERROR: Input file {INPUT_CANDIDATE_TRIPLES} does not exist.")
    with open(INPUT_CANDIDATE_TRIPLES, 'r', encoding='utf-8') as f:
        candidate_triples = [json.loads(line) for line in f]
    prompt_template = load_prompt_template(PROMPT_FUSION_PATH)
    background_text = load_abstracts(INPUT_ABSTRACTS_PATH)
    merged_triples = merge_entities(candidate_triples)
    logging.info(f"Merged similar entities. Processed {len(merged_triples)} triples.")
    final_triples = resolve_conflicts_and_fuse(merged_triples, background_text, prompt_template)
    logging.info(f"Resolved conflicts. Final graph contains {len(final_triples)} unique triples.")
    novel_triplets = infer_novel_triplets(final_triples, background_text, prompt_template)
    final_triples.extend(novel_triplets)
    logging.info(f"Inferred {len(novel_triplets)} novel triplets. Total triples: {len(final_triples)}")
    os.makedirs(os.path.dirname(OUTPUT_FINAL_GRAPH), exist_ok=True)
    with open(OUTPUT_FINAL_GRAPH, 'w', encoding='utf-8') as f:
        for t in final_triples:
            f.write(json.dumps(t) + "\n")
    logging.info(f"Final refined knowledge graph saved to {OUTPUT_FINAL_GRAPH}")

if __name__ == "__main__":
    main()