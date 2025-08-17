#!/usr/bin/env python3
from dotenv import load_dotenv
import os
import sys
import json
import random
import logging

# Load environment variables from .env file (if present)
load_dotenv()

# Read and validate API key at the very start
google_api_key = os.getenv("GOOGLE_API_KEY")
if not google_api_key:
    sys.exit("ERROR: Please set the GOOGLE_API_KEY environment variable (or add it to a .env file).")

# Configure logging early
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s: %(message)s")

from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from tqdm import tqdm

def extract_seed_entities(model_name, raw_text_file, output_file, prompt_path, max_input_char, num_samples):
    """
    Extract seed entities from a single text file using Google Gemini API.
    
    Args:
        model_name (str): Gemini model identifier
        raw_text_file (str): Path to the text file containing documents
        output_file (str): Path to write extracted concepts
        prompt_path (str): Path to prompt template file
        max_input_char (int): Maximum characters per input sample
        num_samples (int): Number of text chunks to sample
    """
    logging.info("Starting Step 1: Seed Entity Extraction with Gemini API")
    
    # Initialize the model with API key
    model = ChatGoogleGenerativeAI(
        model=model_name, 
        temperature=0.1,
        google_api_key=google_api_key
    )
    
    # Read the text file
    try:
        with open(raw_text_file, 'r', encoding='utf-8') as rf:
            full_text = rf.read()
    except IOError as e:
        logging.error(f"Could not read text file {raw_text_file}: {e}")
        return
    
    if not full_text.strip():
        logging.error("Text file is empty")
        return
    
    # Split text into chunks (assuming documents are separated by double newlines or similar)
    # You can modify this splitting logic based on your text format
    text_chunks = []
    
    # Method 1: Split by double newlines (common for abstracts)
    if '\n\n' in full_text:
        text_chunks = [chunk.strip() for chunk in full_text.split('\n\n') if chunk.strip()]
    # Method 2: Split by periods and group (if single long text)
    elif len(full_text) > max_input_char * 2:
        sentences = full_text.split('.')
        current_chunk = ""
        for sentence in sentences:
            if len(current_chunk + sentence) < max_input_char:
                current_chunk += sentence + "."
            else:
                if current_chunk:
                    text_chunks.append(current_chunk.strip())
                current_chunk = sentence + "."
        if current_chunk:
            text_chunks.append(current_chunk.strip())
    # Method 3: Use the entire text as one chunk
    else:
        text_chunks = [full_text]
    
    logging.info(f"Split text into {len(text_chunks)} chunks")
    
    # Sample chunks if we have more than requested
    if len(text_chunks) > num_samples:
        samples = random.sample(text_chunks, num_samples)
        logging.info(f"Sampled {num_samples} chunks from {len(text_chunks)} total")
    else:
        samples = text_chunks
        logging.info(f"Using all {len(text_chunks)} chunks")
    
    # Load prompt template
    try:
        with open(prompt_path, 'r', encoding='utf-8') as pf:
            prompt_txt = pf.read()
    except IOError as e:
        logging.error(f"Could not read prompt file {prompt_path}: {e}")
        return
    
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", "You are a knowledge graph builder using Google Gemini."),
        ("user", prompt_txt)
    ])
    
    candidate_concepts = set()
    successful_extractions = 0
    
    for i, text_chunk in enumerate(tqdm(samples, desc="Extracting seed entities")):
        try:
            # Truncate text if needed
            truncated_text = text_chunk[:max_input_char]
            
            # Create the prompt
            invocation = prompt_template.invoke({"abstracts": truncated_text})
            
            # Generate response
            response = model.invoke(invocation)
            
            # Extract concepts from response
            response_text = response.content if hasattr(response, 'content') else str(response)
            
            # Split by comma and clean concepts
            for c in response_text.split(','):
                c_clean = c.strip().lower()
                if c_clean and len(c_clean) > 1:  # Filter out very short concepts
                    candidate_concepts.add(c_clean)
            
            successful_extractions += 1
            
        except Exception as e:
            logging.warning(f"Error processing chunk {i+1}: {e}")
            continue
    
    # Save results
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as wf:
        for concept in sorted(candidate_concepts):
            wf.write(concept + '\n')
    
    logging.info(f"Extracted {len(candidate_concepts)} unique seed entities from {successful_extractions}/{len(samples)} chunks")
    logging.info(f"Results saved to {output_file}")

def main():
    # Hardcoded paths and parameters
    RAW_TEXT_FILE = "abstract.txt"
    # Define an output directory
    OUTPUT_DIR = "output" # or "data/output", etc.
    OUTPUT_FILE_NAME = "seed_entities.txt"
    OUTPUT_FILE = os.path.join(OUTPUT_DIR, OUTPUT_FILE_NAME) # Combine directory and filename
    PROMPT_PATH = "prompts\prompts_step1.txt" # Consider using os.path.join for cross-platform compatibility
    MODEL_NAME = "gemini-2.0-flash"
    MAX_INPUT_CHAR = 10000
    NUM_SAMPLES = 1000
    
    # Validate paths
    if not os.path.isfile(RAW_TEXT_FILE):
        sys.exit(f"ERROR: Text file {RAW_TEXT_FILE} does not exist")
    
    if not os.path.isfile(PROMPT_PATH):
        sys.exit(f"ERROR: Prompt file {PROMPT_PATH} does not exist")

    # The os.makedirs call in extract_seed_entities will now work correctly
    # because os.path.dirname(OUTPUT_FILE) will return "output" (or your chosen directory)
    extract_seed_entities(
        model_name=MODEL_NAME,
        raw_text_file=RAW_TEXT_FILE,
        output_file=OUTPUT_FILE,
        prompt_path=PROMPT_PATH,
        max_input_char=MAX_INPUT_CHAR,
        num_samples=NUM_SAMPLES
    )

if __name__ == "__main__":
    main()   