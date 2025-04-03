# -*- coding: utf-8 -*-
"""
Generates evaluation scenarios (lists of criteria) for recipe recommendations,
focusing on readability, maintainability, and efficiency. Scenarios consist of
(value, relation) pairs, with configurable criteria counts per scenario and
optional random sampling for larger sets.
"""

import os
import random
import time
import logging
from itertools import combinations, product
from typing import Any, Dict, List, Optional, Tuple, Set

import pandas as pd

# ==============================================================================
# Configuration Constants
# ==============================================================================

# --- File Paths & Data Source ---
BASE_APP_DIR: str = "/app"  # Base directory (adjust if needed)
INPUT_RECIPE_CSV: str = os.path.join(BASE_APP_DIR, "BalancedRecipe_entity_linking.csv")

# --- Data Processing & Node Cleaning ---
# Placeholder for unknown/missing values where explicitly needed
UNKNOWN_PLACEHOLDER: str = "Unknown"
# Character replacements for creating clean labels consistent across modules
NODE_REPLACEMENT_MAP: Dict[str, str] = {" ": "_", "-": "_", ">": "", "<": "less_than_"}

# --- Scenario Generation Settings ---
# Number of top ingredients to consider for the 'contains' relation options
TOP_N_INGREDIENTS: int = 20
# List of criteria counts for which to generate scenarios (e.g., [1, 2, 3, 4, 5])
COMBINATION_SIZES_TO_GENERATE: List[int] = [1, 2, 3, 4, 5]
# Maximum number of scenarios to keep per size (None means keep all)
# Sampling is applied only if the number of generated scenarios exceeds this limit.
# Useful for managing computational cost with large combination sizes.
SAMPLING_CONFIG: Dict[int, Optional[int]] = {
    1: None,     # Typically generates few, keep all
    2: None,     # Typically generates moderate amount, keep all
    3: 25000,    # Sample up to 25k for size 3
    4: 50000,    # Sample up to 50k for size 4
    5: 50000,    # Sample up to 50k for size 5
}
# Default sample size if a size > 2 is not explicitly listed in SAMPLING_CONFIG
DEFAULT_SAMPLE_SIZE: int = 50000
# Seed for random number generator for reproducible sampling
RANDOM_SEED: int = 42

# --- Logging Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)-8s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
log = logging.getLogger(__name__) # Use a dedicated logger for the module

# ==============================================================================
# Helper Functions (Ensure Consistency Across Modules)
# ==============================================================================

def create_clean_label(label: Any) -> str:
    """
    Cleans and standardizes a label for consistent use across KG components.

    Handles various input types, applies replacements, converts to lowercase,
    and ensures a valid string representation. Crucial for matching entities
    between data processing, graph creation, and evaluation.

    Args:
        label: The input value (string, number, None, etc.).

    Returns:
        A cleaned, lowercased string label, or an empty string if the input
        is essentially empty (None, NaN, empty string after stripping).
    """
    if pd.isna(label):
        return ""
    # Convert non-strings safely
    if not isinstance(label, str):
        label = str(label)

    cleaned = label.strip()
    if not cleaned: # If label is just whitespace
        return ""

    # Apply defined replacements
    for char, replacement in NODE_REPLACEMENT_MAP.items():
        cleaned = cleaned.replace(char, replacement)

    # Convert to lowercase for case-insensitive matching
    cleaned = cleaned.lower()

    # Optional: Add more aggressive cleaning if needed (e.g., remove special chars)
    # import re
    # cleaned = re.sub(r'[^a-z0-9_]', '', cleaned) # Keep only letters, numbers, underscore

    return cleaned

def parse_and_clean_multi_value_string(value: Any, separator: str = ',') -> List[str]:
    """
    Splits a string by a separator and cleans each resulting item.

    Filters out empty or invalid labels after cleaning. Handles non-string
    or missing input gracefully.

    Args:
        value: The input string (or other type) potentially containing multiple values.
        separator: The delimiter separating the values.

    Returns:
        A list of valid, cleaned string labels derived from the input.
    """
    if pd.isna(value) or not isinstance(value, str):
        return []

    cleaned_labels = []
    for item in value.split(separator):
        cleaned_label = create_clean_label(item)
        if cleaned_label: # Only add if cleaning resulted in a non-empty string
            cleaned_labels.append(cleaned_label)
    return cleaned_labels

# ==============================================================================
# Core Scenario Generation Logic
# ==============================================================================

def _extract_top_ingredients(recipe_df: pd.DataFrame, n: int) -> List[str]:
    """Extracts the top N most frequent cleaned ingredients from the DataFrame."""
    ingredients_col = 'best_foodentityname'
    if ingredients_col not in recipe_df.columns:
        log.warning(f"'{ingredients_col}' column not found. Cannot extract ingredients.")
        return []

    try:
        # Efficiently parse, clean, explode, and count
        all_ingredients = recipe_df[ingredients_col].dropna().apply(
            parse_and_clean_multi_value_string
        ).explode()

        if all_ingredients.empty:
            log.warning("No valid ingredients found after parsing and cleaning.")
            return []

        ingredient_counts = all_ingredients.value_counts()
        top_ingredients = ingredient_counts.head(n).index.tolist()
        log.info(f"Extracted top {len(top_ingredients)} ingredients.")
        return top_ingredients
    except Exception as e:
        log.error(f"Error during ingredient extraction: {e}", exc_info=True)
        return []

def define_relation_criteria_options(recipe_df: pd.DataFrame, top_n_ingredients: int) -> Dict[str, List[str]]:
    """
    Defines the universe of possible criteria values for each relation type.

    Combines dynamically extracted top ingredients with statically defined,
    cleaned options for other attributes (diet, meal type, etc.). Ensures
    all values are cleaned consistently.

    Args:
        recipe_df: DataFrame containing recipe data (needs ingredient column etc.).
        top_n_ingredients: Number of top ingredients to extract.

    Returns:
        A dictionary mapping relation names to a list of their possible cleaned
        criteria values. Returns an empty dict if critical errors occur.
    """
    log.info("Defining criteria options for relations...")
    start_time = time.time()
    relation_options: Dict[str, List[str]] = {}

    # --- 1. Dynamic Options: Ingredients ---
    relation_options['contains'] = _extract_top_ingredients(recipe_df, top_n_ingredients)

    # --- 2. Static Options (Must match cleaned graph node labels) ---
    # Define categories and their potential raw values
    static_definitions = {
        "HasProteinLevel": ["low_protein", "medium_protein", "high_protein"],
        "HasCarbLevel": ["low_carb", "medium_carb", "high_carb"],
        "HasFatLevel": ["low_fat", "medium_fat", "high_fat"],
        "HasSaturatedFatLevel": ["low_saturated_fat", "medium_saturated_fat", "high_saturated_fat"], # Verify these exact labels
        "HasFiberLevel": ["low_fiber", "medium_fiber", "high_fiber"],
        "HasSodiumLevel": ["low_sodium", "medium_sodium", "high_sodium"],
        "HasSugarLevel": ["low_sugar", "medium_sugar", "high_sugar"],
        "HasCholesterolLevel": ["low_cholesterol", "medium_cholesterol", "high_cholesterol"],
        "HasCalorieLevel": ["low_calorie", "medium_calorie", "high_calorie"],
        "isForMealType": ["breakfast", "lunch", "dinner", "snack", "dessert", "starter", "brunch", "drink"],
        "hasDietType": ["vegetarian", "vegan", "paleo", "standard", "gluten_free", "dairy_free", UNKNOWN_PLACEHOLDER], # Needs explicit handling if UNKNOWN is a node
        "isFromRegion": ["global", "asia", "north_america", "europe", "middle_east", "latin_america_and_caribbean", "oceania", "africa"],
        "needTimeToCook": ["less_than_60_mins", "less_than_15_mins", "less_than_30_mins", "less_than_6_hours", "less_than_4_hours", "more_than_6_hours"],
    }

    # Clean static options using the *same* cleaning function
    log.info("Cleaning static criteria options...")
    for relation, raw_values in static_definitions.items():
        cleaned_values = [create_clean_label(v) for v in raw_values]
        # Filter out empty strings that might result from cleaning
        valid_cleaned_values = [cv for cv in cleaned_values if cv]
        if valid_cleaned_values:
            relation_options[relation] = valid_cleaned_values
        else:
            log.warning(f"No valid cleaned options remained for relation '{relation}'.")

    # --- 3. Final Filtering ---
    # Remove relations that ended up with no valid options
    final_options = {k: v for k, v in relation_options.items() if v}

    duration = time.time() - start_time
    log.info(f"Defined criteria options for {len(final_options)} relations in {duration:.2f}s.")
    return final_options

def _generate_combinations_for_size(
    relation_options: Dict[str, List[str]],
    size: int
) -> List[List[Tuple[str, str]]]:
    """Internal helper: Generates all unique scenario combinations for a specific size."""
    log.debug(f"Generating all combinations for size {size}...")
    possible_scenarios: List[List[Tuple[str, str]]] = []
    # Get relations that actually have options defined
    valid_relation_items = list(relation_options.items())

    if len(valid_relation_items) < size:
        log.warning(f"Cannot form combinations of size {size}: "
                    f"Only {len(valid_relation_items)} relations have options.")
        return []

    # Use itertools for efficient combination generation
    # 1. Choose 'size' relations
    for relation_combo in combinations(valid_relation_items, size):
        relations_in_combo = [item[0] for item in relation_combo]
        value_lists_for_combo = [item[1] for item in relation_combo]

        # 2. Find all value combinations for the chosen relations
        for value_combo in product(*value_lists_for_combo):
            # Create the scenario: list of (value, relation) tuples
            scenario = list(zip(value_combo, relations_in_combo))
            possible_scenarios.append(scenario)

    log.debug(f"Generated {len(possible_scenarios)} raw combinations for size {size}.")
    return possible_scenarios

def generate_evaluation_scenarios(
    relation_options: Dict[str, List[str]],
    sizes: List[int] = COMBINATION_SIZES_TO_GENERATE,
    sampling_config: Dict[int, Optional[int]] = SAMPLING_CONFIG,
    default_sample_size: int = DEFAULT_SAMPLE_SIZE,
    seed: int = RANDOM_SEED
) -> Dict[int, List[List[Tuple[str, str]]]]:
    """
    Generates and potentially samples evaluation scenarios for specified sizes.

    This is the main public function for scenario generation.

    Args:
        relation_options: Pre-defined mapping of relations to their possible values.
        sizes: List of criteria counts (scenario sizes) to generate.
        sampling_config: Map of size to max number of scenarios (or None for all).
        default_sample_size: Fallback sample limit for sizes > 2 if not in config.
        seed: Random seed for reproducible sampling.

    Returns:
        A dictionary mapping each requested size to its list of generated
        (and potentially sampled) scenarios.
    """
    log.info(f"Generating evaluation scenarios for sizes: {sizes}...")
    random.seed(seed) # Ensure reproducibility
    scenarios_by_size: Dict[int, List[List[Tuple[str, str]]]] = {}
    total_generated = 0
    total_kept = 0

    if not relation_options:
        log.error("Relation options are empty. Cannot generate scenarios.")
        return {}

    for size in sizes:
        if size <= 0:
            log.warning(f"Skipping invalid scenario size: {size}")
            continue

        log.info(f"--- Processing size: {size} ---")
        start_time = time.time()

        # 1. Generate all possibilities for this size
        all_combos = _generate_combinations_for_size(relation_options, size)
        num_generated = len(all_combos)
        total_generated += num_generated
        duration = time.time() - start_time
        log.info(f"Generated {num_generated} potential scenarios in {duration:.2f}s.")

        if not all_combos:
            scenarios_by_size[size] = []
            continue

        # 2. Determine sample size limit for this combination size
        limit = sampling_config.get(size)
        # Apply default sampling for sizes > 2 only if not explicitly set to None/Value
        if limit is None and size > 2 and size not in sampling_config:
            limit = default_sample_size
            log.debug(f"Applying default sample size {limit} for size {size}.")

        # 3. Apply sampling if a limit is set and we exceeded it
        if limit is not None and num_generated > limit:
            log.info(f"Sampling down to {limit} scenarios from {num_generated} (seed={seed})...")
            scenarios_by_size[size] = random.sample(all_combos, k=limit)
            log.info(f"Kept {len(scenarios_by_size[size])} scenarios after sampling.")
        else:
            # Keep all if no limit, or if generated count is within limit
            scenarios_by_size[size] = all_combos
            kept_reason = "sampling not configured" if limit is None else f"within limit ({limit})"
            log.info(f"Keeping all {num_generated} generated scenarios ({kept_reason}).")

        total_kept += len(scenarios_by_size[size])

    log.info("--- Scenario Generation Summary ---")
    log.info(f"Total potential scenarios generated (all sizes): {total_generated}")
    log.info(f"Total scenarios kept after sampling: {total_kept}")
    for s, scenarios in scenarios_by_size.items():
        log.info(f"  Size {s}: {len(scenarios)} scenarios kept.")
    log.info("---------------------------------")

    return scenarios_by_size

# ==============================================================================
# Example Usage (when script is run directly)
# ==============================================================================

if __name__ == "__main__":
    log.info("===== Running Scenario Generation Module Example =====")

    # 1. Load minimal data needed to define options
    try:
        log.info(f"Loading recipe data from: {INPUT_RECIPE_CSV}")
        # Only load columns potentially needed for options if performance is critical
        # required_cols = ['Name', 'best_foodentityname', ...] # List needed cols
        # recipe_df_raw = pd.read_csv(INPUT_RECIPE_CSV, usecols=required_cols)
        recipe_df_raw = pd.read_csv(INPUT_RECIPE_CSV)
        # Deduplication is good practice even just for defining options accurately
        recipe_df = recipe_df_raw.drop_duplicates(subset='Name', keep='first').reset_index(drop=True)
        log.info(f"Loaded {len(recipe_df)} unique recipes for option definition.")
    except FileNotFoundError:
        log.critical(f"CRITICAL: Recipe file not found at '{INPUT_RECIPE_CSV}'. Cannot proceed.")
        exit(1) # Exit with error code
    except Exception as e:
        log.critical(f"CRITICAL: Failed to load recipe data: {e}", exc_info=True)
        exit(1)

    # 2. Define the universe of possible relation values
    available_relation_options = define_relation_criteria_options(recipe_df, TOP_N_INGREDIENTS)

    # 3. Generate scenarios for configured sizes, with sampling
    if available_relation_options:
        generated_scenarios_map = generate_evaluation_scenarios(
            relation_options=available_relation_options,
            # Uses defaults: sizes=[1, 2, 3, 4, 5], sampling_config=SAMPLING_CONFIG, seed=RANDOM_SEED
        )

        # 4. Display a sample of the results
        log.info("===== Example Generated Scenarios (Sample) =====")
        for size, scenarios in generated_scenarios_map.items():
            print(f"\n--- Size {size} Scenarios (Top 5) ---")
            if scenarios:
                for i, scenario in enumerate(scenarios[:5]):
                    print(f"  [{i+1}] {scenario}")
                if len(scenarios) > 5:
                    print("      ...")
            else:
                print("  (No scenarios generated or kept)")
        log.info("==============================================")
    else:
        log.error("Scenario generation skipped because relation options could not be defined.")

    log.info("===== Scenario Generation Module Example Finished =====")