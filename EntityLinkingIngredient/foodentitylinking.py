import os
import time
import traceback

import numpy as np
import pandas as pd
import openai
from pinecone import Pinecone, ServerlessSpec
import google.generativeai as genai

# IMPORTANT: For handling 429 / ResourceExhausted from Gemini
from google.api_core.exceptions import ResourceExhausted

from tqdm import tqdm

# ============= 1) Configuration & Initialization =============
print("Initializing APIs and setting up configurations...")

# API keys
openai.api_key = os.environ.get("OPENAI_API_KEY")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

# Gemini/PaLM config
genai.configure(api_key=GEMINI_API_KEY)

# For advanced control (optional)
generation_config = genai.GenerationConfig(
    temperature=0.2,
    top_p=0.9,
    top_k=40,
    max_output_tokens=2048
)

# Gemini/PaLM model name
gemini_model = genai.GenerativeModel(
    model_name="gemini-2.0-flash-thinking-exp-01-21",  # or any available to you
    generation_config=generation_config
)

# Pinecone settings
PINECONE_ENV = "us-east-1"  # e.g., "us-west4-gcp"
INDEX_NAME = "usda-ingredient-index"
NAMESPACE = "usda"
BATCH_SIZE = 100

# OpenAI embedding model
EMBEDDING_MODEL = "text-embedding-3-large"  # Example
EMBED_DIMENSIONS = 3072

# Filenames
USDA_CSV_PATH = "/content/Usda_ingredients_id.csv"
INGREDIENT_LIST_PATH = "/content/ingredientList.csv"
SAVE_MAPPING_PATH = "/content/drive/MyDrive/ingredients_mapped.csv"


# ============= 2) Load USDA Data =============
print("Loading USDA data from CSV...")
usda_df = pd.read_csv(USDA_CSV_PATH)
usda_items = [
    {"id": str(row["ingredientId"]), "text": str(row["ingredientName"])}
    for _, row in usda_df.iterrows()
]
print(f"Total USDA items loaded: {len(usda_items)}")


# ============= 3) Embedding Utilities with Retry =============
def embed_texts(texts, model=EMBEDDING_MODEL):
    """
    Embeds a list of texts using OpenAI, with retry logic for rate limits.
    """
    max_retries = 5
    for attempt in range(max_retries):
        try:
            response = openai.Embedding.create(model=model, input=texts)
            return [item["embedding"] for item in response["data"]]
        except openai.error.RateLimitError:
            print("Hit OpenAI rate limit, sleeping for 5 seconds...")
            time.sleep(5)
        except Exception as e:
            print(f"Error during embed_texts: {e}")
            raise
    raise RuntimeError("Exceeded maximum retries for embed_texts.")


def embed_single_text(text, model=EMBEDDING_MODEL):
    """
    Embeds a single text using OpenAI, with retry logic.
    """
    max_retries = 5
    for attempt in range(max_retries):
        try:
            response = openai.Embedding.create(model=model, input=[text])
            return response["data"][0]["embedding"]
        except openai.error.RateLimitError:
            print("Hit OpenAI rate limit, sleeping for 5 seconds...")
            time.sleep(5)
        except Exception as e:
            print(f"Error during embed_single_text: {e}")
            time.sleep(5)
    raise RuntimeError("Exceeded maximum retries for embed_single_text.")


# ============= 4) Pinecone Initialization & Index Check =============
print("Initializing Pinecone and checking/creating index if needed...")
pc = Pinecone(api_key=PINECONE_API_KEY)
index_list = pc.list_indexes().names()

if INDEX_NAME in index_list:
    print(f"Index '{INDEX_NAME}' already exists.")
    skip_embedding = True
else:
    print(f"Index '{INDEX_NAME}' not found. Creating it and embedding USDA data...")
    skip_embedding = False
    pc.create_index(
        name=INDEX_NAME,
        dimension=EMBED_DIMENSIONS,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region=PINECONE_ENV),
    )

index = pc.Index(INDEX_NAME)


# ============= 5) Build & Upsert USDA Embeddings (if needed) =============
if not skip_embedding:
    print("Building embeddings for all USDA items and upserting to Pinecone in batches...")
    usda_vectors = []

    # Embedding in batches
    for i in tqdm(range(0, len(usda_items), BATCH_SIZE), desc="Embedding USDA items"):
        batch = usda_items[i : i + BATCH_SIZE]
        texts = [b["text"] for b in batch]
        embeddings = embed_texts(texts, model=EMBEDDING_MODEL)
        usda_vectors.extend(
            {
                "id": item["id"],
                "values": emb,
                "metadata": {"ingredientName": item["text"]},
            }
            for item, emb in zip(batch, embeddings)
        )

    # Upserting in batches
    for i in tqdm(range(0, len(usda_vectors), BATCH_SIZE), desc="Upserting to Pinecone"):
        segment = usda_vectors[i : i + BATCH_SIZE]
        max_retries = 5
        for attempt in range(max_retries):
            try:
                index.upsert(vectors=segment, namespace=NAMESPACE)
                break
            except Exception as e:
                print(f"Error upserting batch to Pinecone (attempt {attempt+1}): {e}")
                time.sleep(5)
        else:
            raise RuntimeError("Exceeded maximum Pinecone upsert retries.")


# ============= 6) Query Pinecone for top matches (top_k=30) =============
def query_usda(ingredient_text, top_k=45, namespace=NAMESPACE):
    """
    Queries Pinecone to get the top_k=45 matches for the given ingredient_text.
    """
    emb = embed_single_text(ingredient_text, model=EMBEDDING_MODEL)
    max_retries = 5
    for attempt in range(max_retries):
        try:
            result = index.query(
                vector=emb,
                top_k=top_k,       # <<-- 30 results
                namespace=namespace,
                include_metadata=True
            )
            return result["matches"]
        except openai.error.RateLimitError:
            print("Hit rate limit querying Pinecone, sleeping for 5s...")
            time.sleep(5)
        except Exception as e:
            print(f"Error in query_usda (attempt {attempt+1}): {e}")
            time.sleep(5)
    raise RuntimeError("Exceeded maximum retries for query_usda.")


# ============= 7) Gemini Disambiguation =============
def gemini_disambiguation(input_ing, candidates):
    """
    Calls the Gemini LLM to decide on the best USDA candidate or 'no match'.
    """
    prompt = f"""
You are a Food Ingredient Disambiguation Assistant.
You have a user-provided ingredient and some candidate USDA ingredients with similarity scores.

Task:
1. Choose which candidate from the list is the best match for the user-provided ingredient.
2. If there is no suitable candidate, respond with 'no match'.
3. Only provide the name of the best ingredient or 'no match'.

User ingredient: {input_ing}

Candidates:
"""
    for i, c in enumerate(candidates, start=1):
        prompt += f"{i}. {c['metadata']['ingredientName']} (score={c['score']:.3f})\n"

    prompt += "\nAnswer with either one candidate name exactly or 'no match'.\n"

    print("\n--- Prompt to Gemini LLM ---")
    print(prompt.strip())
    print("--- End of Prompt ---")

    chat_session = gemini_model.start_chat(history=[])
    max_retries = 5
    response = None

    for attempt in range(max_retries):
        try:
            response = chat_session.send_message(prompt)
            break
        except ResourceExhausted:
            print("Gemini ResourceExhausted error; sleeping 5s then retrying...")
            time.sleep(5)
        except Exception as e:
            print(f"Error calling Gemini LLM: {e}")
            time.sleep(5)
    else:
        raise RuntimeError("Exceeded max retries for Gemini LLM call.")

    llm_response = (response.text or "").strip()
    print(f"LLM Response: {llm_response}\n")
    return llm_response


# ============= 8) Ingredient Linking Workflow (top_k=30) =============
def link_ingredient(ingredient_text, top_k=30):
    """
    Returns a dict with the final mapping outcome for the given ingredient_text,
    using top_k=30 Pinecone results.
    """
    print(f"\n=== Linking Ingredient: {ingredient_text} ===")
    top_candidates = query_usda(ingredient_text, top_k=top_k, namespace=NAMESPACE)

    if not top_candidates:
        print(f"No Pinecone matches for '{ingredient_text}', returning null.\n")
        return {
            "input": ingredient_text,
            "best_usda_name": None,
            "score": 0.0
        }

    # Print candidate info
    for i, cand in enumerate(top_candidates):
        print(f"  Candidate {i+1}: {cand['metadata']['ingredientName']} (Score={cand['score']:.3f})")

    # Send top candidates to Gemini for LLM disambiguation
    best_name = gemini_disambiguation(ingredient_text, top_candidates)

    # If LLM says "no match" or is empty
    if best_name.lower() in ["no match", "no", "none", ""]:
        print(f"  Decision => no match => returning null for '{ingredient_text}'.\n")
        return {
            "input": ingredient_text,
            "best_usda_name": None,
            "score": 0.0
        }

    # If LLM's chosen name matches exactly one of the top candidates
    for c in top_candidates:
        if c["metadata"]["ingredientName"].lower() == best_name.lower():
            print(f"  Decision => {best_name} (score={c['score']:.3f})\n")
            return {
                "input": ingredient_text,
                "best_usda_name": c["metadata"]["ingredientName"],
                "score": c["score"]
            }

    # If no exact LLM match, default to top Pinecone candidate
    print("  No exact LLM match. Returning top Pinecone candidate.\n")
    top_choice = top_candidates[0]
    return {
        "input": ingredient_text,
        "best_usda_name": top_choice["metadata"]["ingredientName"],
        "score": top_choice["score"]
    }


# ============= 9) Main Execution With Partial Saves =============
def main():
    try:
        print("Reading ingredient list and linking them...")
        ingredient_df = pd.read_csv(INGREDIENT_LIST_PATH)
        ingredients_list = ingredient_df["ingredient"].tolist()

        # Load partial results if file exists
        if os.path.exists(SAVE_MAPPING_PATH):
            existing_results = pd.read_csv(SAVE_MAPPING_PATH)
            mapped_already = set(existing_results["input"].tolist())
            print(f"Found existing partial results with {len(mapped_already)} ingredients. Resuming...")
        else:
            existing_results = pd.DataFrame(columns=["input", "best_usda_name", "score"])
            mapped_already = set()

        final_rows = []

        # Process each ingredient
        for ing in tqdm(ingredients_list, desc="Linking ingredients"):
            if ing in mapped_already:
                # Already processed
                continue

            # Link with top_k=30
            result = link_ingredient(ing, top_k=30)
            final_rows.append(result)

            # Save partial every 20 new results
            if len(final_rows) % 20 == 0:
                partial_df = pd.concat([existing_results, pd.DataFrame(final_rows)], ignore_index=True)
                partial_df.drop_duplicates(subset="input", keep="last", inplace=True)
                partial_df.to_csv(SAVE_MAPPING_PATH, index=False)
                print(f"[Checkpoint] Saved partial results to {SAVE_MAPPING_PATH}")

        # Combine final with partial, remove duplicates
        df_results = pd.concat([existing_results, pd.DataFrame(final_rows)], ignore_index=True)
        df_results.drop_duplicates(subset="input", keep="last", inplace=True)

        print("\nSample of final results:")
        print(df_results.head(20))

        df_results.to_csv(SAVE_MAPPING_PATH, index=False)
        print(f"\nIngredient mapping completed and saved to '{SAVE_MAPPING_PATH}'.")

        # ============= 10) Re-process any NaNs using top_k=30 =============
        # If you want to immediately fill in any leftover NaNs in the same run:
        still_null = df_results[df_results['best_usda_name'].isna()]
        if not still_null.empty:
            print(f"\nRe-processing {len(still_null)} null entries with top_k=30...")
            for idx, row in still_null.iterrows():
                ing = row["input"]
                outcome = link_ingredient(ing, top_k=30)
                df_results.loc[idx, "best_usda_name"] = outcome["best_usda_name"]
                df_results.loc[idx, "score"] = outcome["score"]

            # Save again after re-processing
            df_results.to_csv(SAVE_MAPPING_PATH, index=False)
            print(f"Re-processed null entries saved to '{SAVE_MAPPING_PATH}'.")

    except Exception as e:
        print("An error occurred. Saving partial results and exiting.")
        traceback.print_exc()

        if 'final_rows' in locals():
            partial_df = pd.concat([existing_results, pd.DataFrame(final_rows)], ignore_index=True)
        else:
            partial_df = existing_results

        partial_df.drop_duplicates(subset="input", keep="last", inplace=True)
        partial_df.to_csv(SAVE_MAPPING_PATH, index=False)
        print(f"Partial results saved to {SAVE_MAPPING_PATH} before exiting.")


if __name__ == "__main__":
    main()
