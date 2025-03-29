# -*- coding: utf-8 -*-
"""
Knowledge Graph Embedding Pipeline for Recipe Recommendation and Evaluation (Cleaned).

This script performs the following steps:
1. Loads recipe data from a CSV file.
2. Processes the data to create nodes and relationships for a knowledge graph.
   - Cleans labels for nodes using a specific replacement map.
   - Handles multi-value fields (ingredients, types, etc.).
   - Assigns specific relations based on 'Healthy_Type' keywords.
3. Generates triples (head, relation, tail) representing the graph.
4. Saves the generated triples to a CSV file.
5. Creates a PyKEEN TriplesFactory from the triples.
6. Trains a Knowledge Graph Embedding model (ConvE specified).
   - Uses the *entire* dataset for training, testing, and validation as per
     the original script's setup.
7. Prints standard PyKEEN evaluation metrics.
8. Saves the trained model.
9. Prepares the original recipe DataFrame for evaluation filtering.
10. Defines test scenarios based on combinations of recipe criteria (1 to 8).
    - Dynamically extracts top ingredients for 'contains' relation.
    - Generates all combinations for sizes 1 & 2.
    - Samples scenarios randomly for sizes 3 to 8.
11. Calculates the average number of recipes matching the criteria in the
    sampled scenarios for each level (used as 'k' in Precision@k).
12. Evaluates the trained model's performance for each criteria level:
    - Filters the evaluation DataFrame to find ground truth (relevant) recipes.
    - Predicts recipe heads (`predict_target`) for each criterion in the scenario.
    - Aggregates prediction scores by summing them for each potential recipe head.
    - Ranks predicted recipes by aggregated score.
    - Calculates Precision@k, Recall@N, F1@N, and Accuracy@N, where 'N' is
      the number of ground truth recipes for the scenario and 'k' is the
      pre-calculated average occurrence.
13. Saves detailed evaluation results to separate CSV files for each criteria count.
"""

import os
import random
from itertools import combinations, product
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx
import numpy as np
import pandas as pd
import torch # Often needed for device selection

# PyKEEN Imports
from pykeen.pipeline import pipeline
from pykeen.predict import predict_target
from pykeen.triples import TriplesFactory
# from pykeen.hpo import hpo_pipeline # Not used in the final original script

# --- Configuration Constants ---

# File Paths & Directories
BASE_APP_DIR = "/app" # Adjust if necessary
INPUT_CSV_PATH = os.path.join(BASE_APP_DIR, "BalancedRecipe_entity_linking.csv")
TRIPLES_OUTPUT_PATH = os.path.join(BASE_APP_DIR, "recipes_triples_clean.csv") # Changed filename slightly
RESULTS_DIR = os.path.join(BASE_APP_DIR, "ConvEEvulationResults") # Original results directory name

# Data Processing
UNKNOWN_PLACEHOLDER = "Unknown" # Placeholder for missing optional values like Diet_Types
NODE_REPLACEMENT_MAP = {" ": "_", "-": "_", ">": "", "<": "less_than_"} # For cleaning labels

# Model Training & PyKEEN
MODEL_NAME = "ConvE" # Explicitly using ConvE as per original
EPOCHS = 150
RANDOM_SEED = 42 # For reproducibility in PyKEEN and sampling
# Note: Original script used the *same* TriplesFactory for train/test/validation.
# This is generally NOT recommended for rigorous evaluation but preserved here.
# A typical split would be: TRAIN_SPLIT, VALID_SPLIT, TEST_SPLIT = 0.7, 0.1, 0.2

# Evaluation Settings
TOP_N_INGREDIENTS = 20 # For the 'contains' relation options
MAX_CRITERIA_COMBINATIONS = 8 # Evaluate scenarios from 1 to 8 criteria
# Define sample sizes used in the original script for random scenario selection
SAMPLING_CONFIG = {
    3: 25000,
    4: 50000,
    5: 50000,
    6: 50000,
    7: 50000,
    8: 50000,
}
DEFAULT_SAMPLE_SIZE = 50000 # Fallback if size not in SAMPLING_CONFIG (though 1-8 are covered)

# Relation Mapping (Graph Creation & Evaluation Consistency)
# Base relations used when creating triples
RELATION_MAPPING = {
    'ingredients': 'contains',
    'diet_types': 'hasDietType',
    'meal_type': 'isForMealType',
    'cook_time': 'needTimeToCook',
    'region_countries': 'isFromRegion',
    'healthy_types': 'HasHealthAttribute' # Default/base for health types
}

# Specific relations derived from 'Healthy_Type' values during graph creation
HEALTH_RELATION_KEYWORDS = {
    # Order matters for overlapping terms (e.g., 'saturated fat' vs 'fat')
    'saturated fat': 'HasSaturatedFatLevel',
    'protein': 'HasProteinLevel',
    'carb': 'HasCarbLevel',
    'fat': 'HasFatLevel', # General fat if 'saturated' isn't present
    'calorie': 'HasCalorieLevel',
    'sodium': 'HasSodiumLevel',
    'sugar': 'HasSugarLevel',
    'fiber': 'HasFiberLevel',
    'cholesterol': 'HasCholesterolLevel',
}

# Mapping from relations used in *evaluation scenarios* back to DataFrame columns
# This MUST align with how relations were created and which columns hold the info.
EVALUATION_RELATION_COLUMN_MAP = {
    # Health relations map back to the single 'Healthy_Type' column
    "HasProteinLevel": "Healthy_Type",
    "HasCarbLevel": "Healthy_Type",
    "HasFatLevel": "Healthy_Type",
    "HasSaturatedFatLevel": "Healthy_Type", # Check if needed based on actual node values
    "HasFiberLevel": "Healthy_Type",
    "HasSodiumLevel": "Healthy_Type",
    "HasSugarLevel": "Healthy_Type",
    "HasCholesterolLevel": "Healthy_Type",
    "HasCalorieLevel": "Healthy_Type",
    # Other relations map to their respective columns
    "isForMealType": "meal_type",
    "hasDietType": "Diet_Types",
    "isFromRegion": "CleanedRegion",
    "needTimeToCook": "cook_time",
    # 'contains' relation maps to the linked ingredients column
    "contains": "best_foodentityname",
}

# --- Helper Functions ---

def create_node_label(label: Any) -> str:
    """
    Cleans and standardizes a label string for use as a graph node.

    Replaces specific characters, converts to lowercase, strips whitespace,
    and handles non-string input by converting to string first.

    Args:
        label: The input label (can be str, int, float, etc.).

    Returns:
        A standardized string suitable for use as a node ID. Returns empty
        string if input is None or NaN after string conversion.
    """
    if pd.isna(label):
        return "" # Handle NaN explicitly
    if not isinstance(label, str):
        label = str(label)

    cleaned_label = label.strip().lower()
    # Apply replacements BEFORE potential removal of other chars
    for char, replacement in NODE_REPLACEMENT_MAP.items():
        cleaned_label = cleaned_label.replace(char, replacement)

    # Add more aggressive cleaning if needed, e.g., remove all non-alphanumeric/_
    # import re
    # cleaned_label = re.sub(r'[^a-zA-Z0-9_]', '', cleaned_label)
    return cleaned_label

def parse_multi_value_column(value: Any, separator: str = ',') -> List[str]:
    """
    Parses a string containing multiple values separated by a separator.

    Args:
        value: The input string or other type (will be converted to string).
               Handles NaN/None by returning an empty list.
        separator: The delimiter string (default ',').

    Returns:
        A list of cleaned, non-empty string labels.
    """
    if pd.isna(value):
        return []
    if not isinstance(value, str):
        value = str(value) # Ensure it's a string before splitting

    cleaned_items = []
    for item in value.split(separator):
        cleaned_label = create_node_label(item.strip()) # Clean each part
        if cleaned_label: # Only add non-empty labels
            cleaned_items.append(cleaned_label)
    return cleaned_items

def get_health_relation(element_node_label: str) -> str:
    """
    Determines the specific health-related relation type based on keywords
    found within the cleaned node label. Uses HEALTH_RELATION_KEYWORDS.

    Args:
        element_node_label: The cleaned label of the health attribute node.

    Returns:
        The specific relation name (e.g., 'HasProteinLevel') or the default
        'HasHealthAttribute' if no specific keyword matches.
    """
    # Ensure label is lower for case-insensitive matching
    label_lower = element_node_label.lower()

    # Check keywords based on predefined mapping (order might matter)
    for keyword, relation in HEALTH_RELATION_KEYWORDS.items():
        if keyword in label_lower:
             # Special check for 'fat' to avoid misclassifying 'saturated fat'
             if keyword == 'fat' and 'saturated' in label_lower:
                 continue # Skip general 'fat' if 'saturated' is present
             return relation # Return the first specific match found

    # Fallback to the default relation if no specific keyword is found
    return RELATION_MAPPING['healthy_types']


# --- Graph Creation Logic ---

def process_recipes_for_graph(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """
    Transforms the recipe DataFrame into a dictionary structured for graph creation.
    Uses cleaned labels for all entities.

    Args:
        df: The input DataFrame with recipe data (unique recipes expected).

    Returns:
        A dictionary where keys are cleaned recipe names (nodes) and values
        are dictionaries mapping detail types (like 'ingredients', 'diet_types')
        to lists of cleaned associated entity labels (nodes) or single cleaned labels.
    """
    recipes_graph_dict = {}
    required_columns = ['Name', 'best_foodentityname', 'Healthy_Type', 'meal_type', 'cook_time', 'Diet_Types', 'CleanedRegion']
    # Check if essential columns exist
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
         print(f"Warning: Missing essential columns for graph creation: {missing_cols}")
         # Decide how to handle: raise error, return empty, or proceed with available data?
         # For now, proceed with available data.

    for _, row in df.iterrows():
        recipe_name = create_node_label(row['Name'])
        if not recipe_name: # Skip recipes with invalid names
            continue

        # Use linked entities if available, otherwise empty list
        ingredients = parse_multi_value_column(row.get('best_foodentityname', ''))
        healthy_types = parse_multi_value_column(row.get('Healthy_Type', ''))
        meal_types = parse_multi_value_column(row.get('meal_type', ''))
        # cook_time is expected to be a single value per recipe
        cook_time = create_node_label(row.get('cook_time', ''))
        # Diet_Types might be missing, use placeholder if empty list results
        diet_types = parse_multi_value_column(row.get('Diet_Types', ''))
        if not diet_types:
             # Original script assigned UNKNOWN_PLACEHOLDER if Diet_Types was NaN.
             # Let's replicate that: use placeholder if parsing results in empty list.
             # Note: create_node_label should NOT return UNKNOWN_PLACEHOLDER directly.
             diet_types = [create_node_label(UNKNOWN_PLACEHOLDER)]
        region_countries = parse_multi_value_column(row.get('CleanedRegion', ''))

        recipes_graph_dict[recipe_name] = {
            "ingredients": ingredients,
            "diet_types": diet_types,
            "meal_type": meal_types,
            "cook_time": cook_time, # Stored as single string node label
            "region_countries": region_countries,
            "healthy_types": healthy_types,
        }
    return recipes_graph_dict


def create_graph_and_triples(recipes_dict: Dict[str, Dict[str, Any]]) -> Tuple[nx.Graph, np.ndarray]:
    """
    Builds a NetworkX graph and generates a NumPy array of (head, relation, tail) triples.

    Args:
        recipes_dict: A dictionary representing recipes and their attributes,
                      generated by `process_recipes_for_graph`.

    Returns:
        A tuple containing:
            - G (nx.Graph): The constructed NetworkX graph (undirected).
            - triples_array (np.ndarray): A NumPy array of triples [N, 3].
    """
    G = nx.Graph()
    triples_list = []

    for recipe_node, details in recipes_dict.items():
        # Ensure recipe node exists (should be added only once)
        if not G.has_node(recipe_node):
            G.add_node(recipe_node, type='recipe')

        # Iterate through different types of details (ingredients, diet_types, etc.)
        for detail_key, elements in details.items():
            # Determine the base relation type from mapping
            base_relation = RELATION_MAPPING.get(detail_key, 'hasAttribute')

            # Standardize elements to always be a list for iteration
            items_to_process = elements if isinstance(elements, list) else [elements]

            for element_node in items_to_process:
                # Skip if the element node label is empty after cleaning
                if not element_node:
                    continue

                # Determine the final relation and node type
                relation = base_relation
                node_type = detail_key # Default node type is the category key

                # Special handling for 'healthy_types' to get specific relations/types
                if detail_key == 'healthy_types':
                    relation = get_health_relation(element_node)
                    # Use the *derived specific relation* as the node type for health attributes
                    node_type = relation

                # Add the element node if it doesn't exist, setting its type
                if not G.has_node(element_node):
                    G.add_node(element_node, type=node_type)

                # Add edge to NetworkX graph (if it doesn't exist - graph is undirected)
                if not G.has_edge(recipe_node, element_node):
                    # Store the directed relation name on the edge attribute
                    G.add_edge(recipe_node, element_node, relation=relation)

                # Always add the triple (Head, Relation, Tail) to the list for PyKEEN
                # PyKEEN handles triples, directionality is explicit here.
                triples_list.append((recipe_node, relation, element_node))

    # Convert the list of triples to a NumPy array
    triples_array = np.array(triples_list, dtype=str)
    return G, triples_array


# --- Evaluation Setup Functions ---

def prepare_evaluation_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepares a *copy* of the recipe DataFrame for filtering during evaluation.
    Applies node label cleaning to columns that will be used for matching criteria.
    Crucially, it formats multi-value columns as comma-separated cleaned strings
    to allow `str.contains` matching used in evaluation.

    Args:
        df: The original, cleaned (deduplicated) recipe DataFrame.

    Returns:
        A new DataFrame ready for evaluation filtering.
    """
    print("Preparing DataFrame copy for evaluation filtering...")
    df_eval = df.copy()

    # Clean the 'Name' column for matching predicted heads to ground truth
    df_eval['Name'] = df_eval['Name'].apply(create_node_label)

    # Clean columns used for filtering based on EVALUATION_RELATION_COLUMN_MAP
    # The goal is to have the column contain strings that `str.contains` can match
    # against the cleaned `tail_entity` values from scenarios.
    processed_cols = set(['Name'])
    for relation, col in EVALUATION_RELATION_COLUMN_MAP.items():
        if col in df_eval.columns and col not in processed_cols:
            print(f"  Processing column '{col}' for evaluation...")
            # Check if the column typically holds multiple values (based on original parsing)
            if col in ['best_foodentityname', 'Healthy_Type', 'meal_type', 'Diet_Types', 'CleanedRegion']:
                # Parse into list of cleaned labels, then join back into a single string
                # This ensures that `str.contains(cleaned_tail_entity)` works correctly.
                df_eval[col] = df_eval[col].apply(
                    lambda x: ','.join(parse_multi_value_column(x)) if pd.notna(x) else ''
                )
            elif col == 'cook_time':
                # Assumed single value, just clean it
                df_eval[col] = df_eval[col].apply(create_node_label)
            else:
                # Fallback: Apply basic cleaning if structure unknown
                 print(f"Warning: Unknown structure for column '{col}'. Applying basic node label cleaning.")
                 df_eval[col] = df_eval[col].apply(create_node_label)
            processed_cols.add(col)
        elif col not in df_eval.columns:
            print(f"Warning: Evaluation column '{col}' mapped from relation '{relation}' not found in DataFrame.")

    print("Evaluation DataFrame prepared.")
    return df_eval


def get_relation_options(df_eval: pd.DataFrame, top_n_ingredients: int) -> Dict[str, List[str]]:
    """
    Defines the universe of possible 'tail' entities for each relation type
    used in generating evaluation test scenarios. Extracts top ingredients dynamically.

    Args:
        df_eval: The prepared evaluation DataFrame (needed for extracting ingredients).
        top_n_ingredients: The number of top ingredients to include for 'contains'.

    Returns:
        A dictionary where keys are relation names and values are lists of
        cleaned possible tail entity labels for that relation.
    """
    print(f"Generating relation options (including top {top_n_ingredients} ingredients)...")
    # Dynamic extraction of top N ingredients from the processed column
    top_ingredients = []
    if 'best_foodentityname' in df_eval.columns:
        try:
            # The column should now contain comma-separated cleaned labels
            ingredient_counts = df_eval['best_foodentityname'].dropna().apply(
                lambda x: [ing for ing in x.split(',') if ing] # Split the pre-cleaned string
            ).explode().value_counts()

            # Take the top N, index contains the cleaned ingredient labels
            top_ingredients = ingredient_counts.head(top_n_ingredients).index.tolist()
            if not top_ingredients:
                print("Warning: No ingredients extracted from 'best_foodentityname'. 'contains' relation will have no options.")
            else:
                print(f"  Found {len(top_ingredients)} top ingredients.")
        except Exception as e:
            print(f"Warning: Error extracting top ingredients from 'best_foodentityname': {e}. 'contains' relation may be empty.")
            top_ingredients = []
    else:
        print("Warning: Column 'best_foodentityname' not found. 'contains' relation will have no options.")

    # Define the static options based on knowledge of the data/domain
    # These values MUST match the node labels created during graph generation
    relation_options = {
        # Health Levels - ensure these labels match graph nodes (e.g., low_protein)
        "HasProteinLevel": ["low_protein", "medium_protein", "high_protein"],
        "HasCarbLevel": ["low_carb", "medium_carb", "high_carb"],
        "HasFatLevel": ["low_fat", "medium_fat", "high_fat"],
        "HasSaturatedFatLevel": ["low_saturated_fat", "medium_saturated_fat", "high_saturated_fat"], # Assuming these exist
        "HasFiberLevel": ["low_fiber", "medium_fiber", "high_fiber"],
        "HasSodiumLevel": ["low_sodium", "medium_sodium", "high_sodium"],
        "HasSugarLevel": ["low_sugar", "medium_sugar", "high_sugar"],
        "HasCholesterolLevel": ["low_cholesterol", "medium_cholesterol", "high_cholesterol"],
        "HasCalorieLevel": ["low_calorie", "medium_calorie", "high_calorie"],
        # Other Categories
        "isForMealType": ["breakfast", "lunch", "dinner", "snack", "dessert", "starter", "brunch", "drink"],
        "hasDietType": ["vegetarian", "vegan", "paleo", "standard", "gluten_free", "dairy_free", UNKNOWN_PLACEHOLDER.lower()], # Include placeholder possibility
        "isFromRegion": ["global", "asia", "north_america", "europe", "middle_east", "latin_america_and_caribbean", "oceania", "africa"],
        "needTimeToCook": ["less_than_60_mins", "less_than_15_mins", "less_than_30_mins", "less_than_6_hours", "less_than_4_hours", "more_than_6_hours"],
        # Dynamically added ingredients
        "contains": top_ingredients
    }

    # Clean the *statically defined* option values using the same node cleaning function
    print("  Cleaning static relation option values...")
    for relation, values in relation_options.items():
        if relation != "contains": # 'contains' values are already cleaned
            relation_options[relation] = [create_node_label(v) for v in values if create_node_label(v)] # Clean and remove empty

    # Filter out relations that ended up with no valid options
    final_relation_options = {k: v for k, v in relation_options.items() if v}
    print(f"Final relation options generated for {len(final_relation_options)} relations.")

    return final_relation_options

def generate_test_scenarios(
    relation_options: Dict[str, List[str]],
    combination_size: int
) -> List[List[Tuple[str, str]]]:
    """
    Generates all possible test scenarios (combinations of criteria) for a specific size.

    Each scenario is a list of (tail_entity, relation_type) tuples.

    Args:
        relation_options: Dictionary mapping relation types to their possible tail values.
        combination_size: The number of criteria to combine in each scenario.

    Returns:
        A list of all generated test scenarios for the given combination size.
        Returns an empty list if no scenarios can be generated.
    """
    all_test_cases: List[List[Tuple[str, str]]] = []

    # Get relation types that actually have options
    valid_relations = list(relation_options.items()) # List of (relation_name, [value1, value2,...])

    if len(valid_relations) < combination_size:
        # Cannot form combinations if not enough relations have options
        print(f"Warning: Not enough relations with options ({len(valid_relations)}) to form combinations of size {combination_size}.")
        return []

    # Iterate through all combinations of *relation types* of the given size
    for relation_combination_tuple in combinations(valid_relations, combination_size):
        # relation_combination_tuple is like:
        # ( ('hasDietType', ['vegan', 'vegetarian']), ('isForMealType', ['lunch']) )

        # Extract the lists of values and the relation names for this specific combination
        value_lists = [relation_data[1] for relation_data in relation_combination_tuple]
        relation_names = [relation_data[0] for relation_data in relation_combination_tuple]

        # Generate the Cartesian product of the *values* for the selected relations
        # Example: product(['vegan', 'vegetarian'], ['lunch']) -> ('vegan', 'lunch'), ('vegetarian', 'lunch')
        for specific_values_tuple in product(*value_lists):
            # Create the single test case: list of (value, relation_type) tuples
            test_case = list(zip(specific_values_tuple, relation_names))
            all_test_cases.append(test_case)

    return all_test_cases


def calculate_average_occurrence(
    scenarios: List[List[Tuple[str, str]]],
    df_eval: pd.DataFrame
) -> Tuple[int, List[List[Tuple[str, str]]]]:
    """
    Calculates the average number of recipes matching the criteria across a list of scenarios.
    Also identifies scenarios that result in zero matching recipes.

    Args:
        scenarios: A list of test scenarios to analyze.
        df_eval: The prepared evaluation DataFrame.

    Returns:
        A tuple containing:
        - average_k (int): The floor integer of the average matching recipes per *valid* scenario.
        - zero_match_scenarios (List): A list of scenarios that yielded 0 matches.
    """
    total_matching_recipes = 0
    valid_scenario_count = 0 # Scenarios with > 0 matches
    zero_match_scenarios = []

    if not scenarios:
         print("Warning: No scenarios provided to calculate_average_occurrence.")
         return 0, []

    print(f"Calculating average occurrence (k) for {len(scenarios)} scenarios...")
    progress_step = max(1, len(scenarios) // 10) # Show progress roughly 10 times

    for i, scenario in enumerate(scenarios):
        if i > 0 and i % progress_step == 0:
             print(f"  ...processed {i}/{len(scenarios)} scenarios for average k calculation.")

        # Start with a boolean Series indicating all rows are potential matches
        current_matches = pd.Series([True] * len(df_eval))

        # Sequentially apply filters for each criterion in the scenario
        for tail_value, relation in scenario:
            column_name = EVALUATION_RELATION_COLUMN_MAP.get(relation)

            if column_name and column_name in df_eval.columns:
                # Apply the filter using str.contains on the prepared column.
                # Ensure case-insensitivity and handle potential NaNs that might remain.
                # Only update rows that are still True in current_matches.
                 try:
                     # Ensure tail_value is a string for contains
                      search_term = str(tail_value)
                      current_matches &= df_eval[column_name].fillna('').astype(str).str.contains(search_term, case=False) & current_matches
                 except Exception as e:
                      print(f"Error filtering column '{column_name}' with value '{tail_value}': {e}")
                      # Decide how to handle: invalidate the scenario? For now, continue filtering.
                      # To be safe, we could invalidate the match for this criterion:
                      # current_matches = pd.Series([False] * len(df_eval))
                      # break # Stop processing this scenario if a filter fails catastrophically

            else:
                print(f"Warning: Column for relation '{relation}' ('{column_name}') not found or not mapped. Skipping this filter in scenario {i+1}.")
                # Decide if a missing filter invalidates the scenario for 'k' calculation?
                # For now, we proceed, assuming the scenario is still valid based on other criteria.

        # Count the number of rows where all filters resulted in True
        count = df_eval[current_matches].shape[0]

        # Accumulate counts for averaging
        if count > 0:
            total_matching_recipes += count
            valid_scenario_count += 1
        else:
            zero_match_scenarios.append(scenario)

    # Calculate the average
    average_float = total_matching_recipes / valid_scenario_count if valid_scenario_count > 0 else 0.0
    average_k = int(average_float) # Floor integer value as used in original logic for Top-K

    print(f"Average occurrence calculation complete.")
    print(f"  Total matching recipes across {valid_scenario_count} valid scenarios: {total_matching_recipes}")
    print(f"  Average (float): {average_float:.4f}")
    print(f"  Using k = {average_k} (floor integer) for Precision@k.")
    print(f"  Number of scenarios with 0 matches: {len(zero_match_scenarios)}")

    # Optional: Print some zero-match scenarios for debugging
    if zero_match_scenarios and len(zero_match_scenarios) < len(scenarios): # Avoid printing if ALL failed
        print("\nSample scenarios with 0 results:")
        for z in zero_match_scenarios[:min(5, len(zero_match_scenarios))]: # Print a small sample
            print(f"  - {z}")

    return average_k, zero_match_scenarios


# --- Core Evaluation Function ---

def run_evaluation(
    combination_size: int,
    model_pipeline_result: Any, # Opaque type hint for PyKeen result
    recipes_df_eval: pd.DataFrame,
    relation_options: Dict[str, List[str]],
    output_dir: str,
    random_seed: int,
    sampling_config: Dict[int, int],
    default_sample_size: int
):
    """
    Runs the custom evaluation for a specific number of combined criteria.

    Generates test scenarios, calculates average occurrence (k), predicts recipes
    using the trained model, calculates Precision@k, Recall@N, F1@N, Accuracy@N,
    and saves the results to a CSV file.

    Args:
        combination_size: The number of criteria per scenario (e.g., 1, 2, ...).
        model_pipeline_result: The result object from the PyKEEN pipeline call.
        recipes_df_eval: The DataFrame prepared for evaluation filtering.
        relation_options: Dictionary of possible tail entities for each relation.
        output_dir: Directory where the result CSV will be saved.
        random_seed: Seed for random sampling of scenarios.
        sampling_config: Dictionary mapping combination_size to specific sample counts.
        default_sample_size: Fallback sample size if not in config.
    """
    print(f"\n--- Running Evaluation for Combination Size: {combination_size} ---")
    random.seed(random_seed) # Set seed for reproducible sampling

    # 1. Generate Scenarios
    all_scenarios = generate_test_scenarios(relation_options, combination_size)
    if not all_scenarios:
        print(f"No scenarios could be generated for combination size {combination_size}. Skipping evaluation.")
        return

    print(f"Generated {len(all_scenarios)} total possible scenarios.")

    # 2. Sample Scenarios (if applicable, based on original script's logic)
    sample_size = sampling_config.get(combination_size, default_sample_size if combination_size >=3 else None)

    if sample_size is not None and len(all_scenarios) > sample_size:
        print(f"Sampling {sample_size} scenarios randomly (seed={random_seed}).")
        test_scenarios = random.sample(all_scenarios, k=min(sample_size, len(all_scenarios)))
    else:
        test_scenarios = all_scenarios # Use all if no sampling needed or not enough scenarios

    if not test_scenarios:
        print("No test scenarios selected after potential sampling. Skipping evaluation.")
        return

    print(f"Evaluating using {len(test_scenarios)} scenarios.")

    # 3. Calculate Average Occurrence (k for Precision@k)
    # Pass only the *selected* test_scenarios for k calculation
    average_k, _ = calculate_average_occurrence(test_scenarios, recipes_df_eval)

    # Handle k=0: Precision@0 is ill-defined. Set to 1 for calculations, but results are suspect.
    effective_k = max(1, average_k)
    if average_k == 0:
        print("Warning: Average occurrence (k) is 0. Precision@k calculation will use k=1, but results may not be meaningful.")

    # 4. Prepare for Loop
    test_results = []
    model = model_pipeline_result.model
    # Use the training factory for context (entity/relation IDs) during prediction
    triples_factory = model_pipeline_result.training

    # Get the set of all known *cleaned* recipe names from the evaluation DF
    # Used to filter model predictions to only include actual recipes.
    known_recipe_names: Set[str] = set(recipes_df_eval['Name'].unique())
    if not known_recipe_names:
         print("Critical Warning: No known recipe names found in evaluation DataFrame. Evaluation cannot proceed.")
         return


    print(f"Starting evaluation loop over {len(test_scenarios)} scenarios...")
    progress_step = max(1, len(test_scenarios) // 10)

    # 5. Evaluation Loop per Scenario
    for idx, scenario in enumerate(test_scenarios):
        if idx > 0 and idx % progress_step == 0:
             print(f"  ...evaluated {idx}/{len(test_scenarios)} scenarios for size {combination_size}.")

        # 5a. Find Ground Truth (Relevant Recipes for this scenario)
        is_relevant = pd.Series([True] * len(recipes_df_eval))
        for tail_entity, relation in scenario:
            column_name = EVALUATION_RELATION_COLUMN_MAP.get(relation)
            if column_name and column_name in recipes_df_eval.columns:
                 try:
                      search_term = str(tail_entity)
                      is_relevant &= recipes_df_eval[column_name].fillna('').astype(str).str.contains(search_term, case=False) & is_relevant
                 except Exception as e:
                      # Log error but continue, this scenario might become invalid
                      print(f"Error filtering ground truth for scenario {idx+1}, criterion ('{tail_entity}', '{relation}'): {e}")
                      # Option: invalidate scenario by setting is_relevant to all False
                      # is_relevant = pd.Series([False] * len(recipes_df_eval))
                      # break
            # else: # Warning about missing column already given during calculation of k

        relevant_recipes_df = recipes_df_eval[is_relevant]
        # Use the cleaned 'Name' column; ensure uniqueness
        relevant_recipe_names: Set[str] = set(relevant_recipes_df['Name'].unique())
        expected_match_count = len(relevant_recipe_names) # This is 'N' for Recall@N etc.

        # Skip scenario if ground truth is empty
        if expected_match_count == 0:
            # print(f"Skipping Scenario {idx+1} (criteria: {scenario}) - 0 relevant recipes found.")
            continue

        # 5b. Get Model Predictions
        all_predictions_list = []
        valid_prediction_criteria_count = 0
        for tail_entity, relation in scenario:
            # Check if relation and tail entity are known to the model
            if relation not in triples_factory.relation_to_id:
                 # print(f"Warning: Relation '{relation}' OOV in scenario {idx+1}. Skipping criterion.")
                 continue
            if tail_entity not in triples_factory.entity_to_id:
                # print(f"Warning: Tail entity '{tail_entity}' OOV in scenario {idx+1}. Skipping criterion.")
                continue

            # Predict heads for this specific (relation, tail) pair
            try:
                predicted_heads_df = predict_target(
                    model=model,
                    relation=relation,
                    tail=tail_entity,
                    triples_factory=triples_factory # Important for ID mapping
                ).df # Get the pandas DataFrame output
                all_predictions_list.append(predicted_heads_df)
                valid_prediction_criteria_count += 1
            except Exception as e:
                print(f"Error during prediction for scenario {idx+1}, criterion ('{tail_entity}', '{relation}'): {e}")
                # Continue to next criterion in the scenario

        # Skip if no valid predictions could be generated for *any* criterion
        if not all_predictions_list:
            # print(f"Skipping Scenario {idx+1} (criteria: {scenario}) - No valid predictions generated (all criteria OOV?).")
            continue
        # Also skip if predictions were generated but for fewer criteria than expected (e.g. all failed)
        # if valid_prediction_criteria_count < len(scenario):
            # print(f"Warning: Scenario {idx+1} only got predictions for {valid_prediction_criteria_count}/{len(scenario)} criteria.")
            # Decide whether to proceed with partial predictions or skip. Original script implicitly proceeded.

        # 5c. Aggregate Scores & Rank Predictions
        # Combine predictions from all criteria in the scenario
        all_predictions_df = pd.concat(all_predictions_list, ignore_index=True)

        # Filter predictions to include only known recipe names
        # This ensures we only rank actual recipes from our dataset
        recipe_predictions_df = all_predictions_df[all_predictions_df['head_label'].isin(known_recipe_names)]

        if recipe_predictions_df.empty:
             # print(f"Skipping Scenario {idx+1} (criteria: {scenario}) - No predicted heads match known recipe names.")
             continue

        # Aggregate scores by summing for recipes predicted by multiple criteria
        aggregated_predictions = recipe_predictions_df.groupby('head_label').agg(
            total_score=('score', 'sum')
        ).reset_index()

        # Sort by aggregated score (descending) to get the final ranking
        final_predictions_sorted = aggregated_predictions.sort_values(
            by="total_score", ascending=False
        )

        # 5d. Calculate Metrics
        # Get the set of predicted recipe names at K (for Precision)
        predicted_recipes_at_k: Set[str] = set(final_predictions_sorted["head_label"].head(effective_k))

        # Get the set of predicted recipe names at N (for Recall, Accuracy, F1@N)
        # N is the number of relevant recipes for this scenario
        predicted_recipes_at_n: Set[str] = set(final_predictions_sorted["head_label"].head(expected_match_count))

        # Calculate True Positives (intersection of predicted and relevant)
        true_positives_at_k: Set[str] = predicted_recipes_at_k.intersection(relevant_recipe_names)
        true_positives_at_n: Set[str] = predicted_recipes_at_n.intersection(relevant_recipe_names)

        # Precision @ k (using effective_k to avoid division by zero)
        precision_at_k = len(true_positives_at_k) / len(predicted_recipes_at_k) if predicted_recipes_at_k else 0.0
        # Handle edge case if k=0 was forced to k=1 and prediction set is empty
        if effective_k == 1 and not predicted_recipes_at_k: precision_at_k = 0.0


        # Recall @ N (N = |relevant|)
        recall_at_n = len(true_positives_at_n) / expected_match_count # expected_match_count > 0 checked earlier

        # Precision @ N (for standard F1 calculation)
        precision_at_n = len(true_positives_at_n) / len(predicted_recipes_at_n) if predicted_recipes_at_n else 0.0

        # F1 Score @ N (harmonic mean of Precision@N and Recall@N)
        if (precision_at_n + recall_at_n) > 0:
            f1_at_n = 2 * (precision_at_n * recall_at_n) / (precision_at_n + recall_at_n)
        else:
            f1_at_n = 0.0

        # Accuracy @ N (proportion of relevant items found in top N predictions)
        # Note: This is equivalent to Recall@N in this setup where N = |relevant|
        accuracy_at_n = recall_at_n

        # 5e. Store Results for this Scenario
        result_dict = {
            "Scenario_Index": idx + 1,
            "Criteria": scenario, # Store the actual criteria tuple list
            "Relevant_Count (N)": expected_match_count,
            "Avg_Occurrence_K (used)": effective_k, # Record the k used for P@k
            "Predicted_Count_at_K": len(predicted_recipes_at_k),
            "TP_Count_at_K": len(true_positives_at_k),
            "Precision_at_K": precision_at_k,
            "Predicted_Count_at_N": len(predicted_recipes_at_n),
            "TP_Count_at_N": len(true_positives_at_n),
            "Recall_at_N": recall_at_n,
            "Precision_at_N": precision_at_n, # Store P@N for clarity
            "F1_at_N": f1_at_n,
            "Accuracy_at_N": accuracy_at_n, # Equivalent to Recall@N here
        }
        test_results.append(result_dict)

    # 6. Save Results for this Combination Size
    if test_results:
        results_df = pd.DataFrame(test_results)
        # Ensure Criteria column doesn't cause issues during saving (convert to string)
        results_df['Criteria'] = results_df['Criteria'].astype(str)

        output_filename = os.path.join(output_dir, f"{combination_size}_criteria_{MODEL_NAME}.csv")
        try:
            results_df.to_csv(output_filename, index=False)
            print(f"\nEvaluation results for size {combination_size} saved to {output_filename}")
            # Display summary statistics for this level
            print(f"  Summary Metrics (Size {combination_size}):")
            print(f"    Mean Precision@{effective_k}: {results_df['Precision_at_K'].mean():.4f}")
            print(f"    Mean Recall@N: {results_df['Recall_at_N'].mean():.4f}")
            print(f"    Mean F1@N: {results_df['F1_at_N'].mean():.4f}")
            print(f"    Mean Accuracy@N: {results_df['Accuracy_at_N'].mean():.4f}")
        except Exception as e:
            print(f"\nError saving results for size {combination_size} to {output_filename}: {e}")

    else:
        print(f"\nNo valid evaluation results were generated for combination size {combination_size}.")

    print(f"--- Finished Evaluation for Combination Size: {combination_size} ---")


# --- Main Execution Function ---

def main():
    """Main function orchestrating the entire pipeline."""
    print("--- Recipe KG Embedding & Evaluation Pipeline ---")
    print(f"Using PyTorch device: {'cuda' if torch.cuda.is_available() else 'cpu'}")

    # --- 1. Load and Basic Preparation ---
    print(f"\n[Step 1] Loading data from: {INPUT_CSV_PATH}")
    try:
        recipes_df_raw = pd.read_csv(INPUT_CSV_PATH)
    except FileNotFoundError:
        print(f"FATAL Error: Input CSV file not found at {INPUT_CSV_PATH}")
        return
    except Exception as e:
        print(f"FATAL Error: Failed to load CSV {INPUT_CSV_PATH}: {e}")
        return

    print(f"Initial rows loaded: {len(recipes_df_raw)}")
    # Drop duplicates based on 'Name' - crucial for unique recipe nodes
    recipes_df = recipes_df_raw.drop_duplicates(subset='Name', keep='first').reset_index(drop=True)
    print(f"Rows after dropping duplicates by 'Name': {len(recipes_df)}")
    if recipes_df.empty:
        print("FATAL Error: No recipes remaining after deduplication.")
        return

    # --- 2. Process Data for Graph & Create Triples ---
    print("\n[Step 2] Processing recipes and generating triples...")
    recipes_graph_dict = process_recipes_for_graph(recipes_df)
    if not recipes_graph_dict:
         print("FATAL Error: No recipes could be processed for the graph.")
         return

    G, triples_array = create_graph_and_triples(recipes_graph_dict)
    print(f"Generated {len(triples_array)} triples.")
    if len(triples_array) == 0:
         print("FATAL Error: No triples were generated. Cannot proceed.")
         return
    print(f"NetworkX graph created with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")

    # Optional: Save triples (as done in original)
    triples_df = pd.DataFrame(triples_array, columns=['Head', 'Relation', 'Tail']).dropna()
    try:
        triples_df.to_csv(TRIPLES_OUTPUT_PATH, index=False)
        print(f"Triples saved to {TRIPLES_OUTPUT_PATH}")
    except Exception as e:
        print(f"Warning: Failed to save triples to {TRIPLES_OUTPUT_PATH}: {e}")

    # --- 3. Create PyKEEN Triples Factory ---
    print("\n[Step 3] Creating PyKEEN TriplesFactory...")
    try:
        # Using the generated NumPy array directly
        triples_factory = TriplesFactory.from_labeled_triples(triples_array)
        print(f"TriplesFactory created successfully:")
        print(f"  Num Entities: {triples_factory.num_entities}")
        print(f"  Num Relations: {triples_factory.num_relations}")
        print(f"  Num Triples: {triples_factory.num_triples}")
    except Exception as e:
        print(f"FATAL Error: Failed to create TriplesFactory: {e}")
        return

    # --- 4. Train KGE Model ---
    print(f"\n[Step 4] Training KGE model '{MODEL_NAME}'...")
    # Replicating original setup: using the *same* factory for all splits.
    # For real-world use, split properly:
    # training_tf, testing_tf, validation_tf = triples_factory.split([0.8, 0.1, 0.1], random_state=RANDOM_SEED)
    try:
        pipeline_result = pipeline(
            model=MODEL_NAME,
            training=triples_factory, # Use full factory for training
            testing=triples_factory,  # Use full factory for testing
            validation=triples_factory,# Use full factory for validation
            training_kwargs=dict(num_epochs=EPOCHS, batch_size=128), # Added batch size
            evaluation_kwargs=dict(batch_size=128), # Consistent batch size
            stopper='early', # Using early stopping as configured in original
            stopper_kwargs=dict(frequency=5, patience=5, delta=0.001, metric='hits@10'), # Example reasonable stopper params
            random_seed=RANDOM_SEED,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            # Ensure evaluation happens on the validation set for stopper
            evaluator_kwargs=dict(filtered=True), # Use filtered evaluation (realistic)
        )
        print("Model training pipeline finished.")
    except Exception as e:
        print(f"FATAL Error: Model training failed: {e}")
        import traceback
        traceback.print_exc() # Print stack trace for debugging
        return

    # --- 5. Print Standard PyKEEN Metrics & Save Model ---
    print("\n[Step 5] PyKEEN Standard Evaluation Metrics (evaluated on validation_tf=full_factory):")
    try:
        metrics = pipeline_result.metric_results.to_dict()
        for key, value in metrics.items():
            # Ensure value is numeric before formatting
            if isinstance(value, (int, float)):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}") # Print non-numeric values directly
    except Exception as e:
        print(f"Warning: Could not display standard metrics: {e}")

    # Save the trained model and pipeline results
    model_save_dir = os.path.join(BASE_APP_DIR, f"{MODEL_NAME}_pipeline_saved")
    try:
        pipeline_result.save_to_directory(model_save_dir)
        print(f"Trained model and pipeline results saved to {model_save_dir}")
    except Exception as e:
        print(f"Warning: Failed to save trained model to {model_save_dir}: {e}")

    # --- 6. Prepare for Custom Multi-Criteria Evaluation ---
    print("\n[Step 6] Preparing for custom multi-criteria evaluation...")
    # Prepare the DataFrame specifically for evaluation filtering
    recipes_df_eval = prepare_evaluation_dataframe(recipes_df)
    if recipes_df_eval.empty:
         print("FATAL Error: Evaluation DataFrame is empty after preparation.")
         return

    # Generate the universe of relation options for building scenarios
    relation_options = get_relation_options(recipes_df_eval, TOP_N_INGREDIENTS)
    if not relation_options:
         print("FATAL Error: No relation options generated. Cannot create test scenarios.")
         return

    # Ensure results directory exists
    try:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        print(f"Results will be saved in: {RESULTS_DIR}")
    except Exception as e:
        print(f"FATAL Error: Could not create results directory {RESULTS_DIR}: {e}")
        return

    # --- 7. Run Custom Evaluation Loop ---
    print("\n[Step 7] Starting custom evaluation loop...")
    for i in range(1, MAX_CRITERIA_COMBINATIONS + 1):
        run_evaluation(
            combination_size=i,
            model_pipeline_result=pipeline_result,
            recipes_df_eval=recipes_df_eval,
            relation_options=relation_options,
            output_dir=RESULTS_DIR,
            random_seed=RANDOM_SEED,
            sampling_config=SAMPLING_CONFIG, # Pass the specific sampling dict
            default_sample_size=DEFAULT_SAMPLE_SIZE
        )

    print("\n--- Pipeline Finished ---")

# --- Script Entry Point ---
if __name__ == "__main__":
    main()
