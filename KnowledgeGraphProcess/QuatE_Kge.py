# -*- coding: utf-8 -*-
"""
Knowledge Graph Embedding Model Training and Evaluation for Recipe Recommendations.

This script loads recipe data, constructs a knowledge graph, trains KGE models
(RotatE, QuatE), and evaluates their performance on recommendation scenarios
with varying numbers of criteria.
"""

import os
import random
from itertools import combinations, product
import logging
import time # Added for timing sections

import networkx as nx
import numpy as np
import pandas as pd
from pykeen.pipeline import pipeline
from pykeen.predict import predict_target
from pykeen.triples import TriplesFactory

# --- Configuration Constants ---
# File Paths
BASE_DIR = '/app' # Base directory for data and results
INPUT_RECIPE_FILE = os.path.join(BASE_DIR, 'BalancedRecipe_entity_linking.csv')
TRIPLES_FILE = os.path.join(BASE_DIR, 'recipes_triples_cleaned.csv') # Centralized triples file
ROTAT_E_RESULTS_DIR = os.path.join(BASE_DIR, 'RotatEEvulationResults')
QUAT_E_RESULTS_DIR = os.path.join(BASE_DIR, 'QuatEEvulationResults')

# Data Processing
UNKNOWN_PLACEHOLDER = 'Unknown'
TOP_N_INGREDIENTS = 20 # Number of top ingredients to use for 'contains' relation

# Model Training
KGE_MODELS = ['RotatE', 'QuatE'] # Models to train and evaluate
EPOCHS = 150 # Consider making this smaller for quicker testing if needed
EARLY_STOPPING = True
# You might want to specify other PyKEEN hyperparameters here if needed
# E.g., embedding dimension, learning rate. These will use PyKEEN defaults otherwise.
# MODEL_KWARGS = {'RotatE': dict(embedding_dim=200), 'QuatE': dict(embedding_dim=200)}
# TRAINING_KWARGS = dict(num_epochs=EPOCHS, learning_rate=0.01)

# Evaluation
# Criteria sizes to test for each model
CRITERIA_SIZES = {
    'RotatE': [4, 5, 6, 7, 8],
    'QuatE': [1, 2, 3, 4, 5, 6, 7, 8]
}
# Number of random scenarios to generate per criteria size
# Reduced for faster execution/testing, increase for more robust evaluation
NUM_RANDOM_SCENARIOS = {
    1: 100,
    2: 500,
    3: 1000, # Reduced from original large numbers
    4: 2000, # Reduced from original large numbers
    5: 2000, # Reduced from original large numbers
    6: 2000, # Reduced from original large numbers
    7: 2000, # Reduced from original large numbers
    8: 2000, # Reduced from original large numbers
}
DEFAULT_NUM_SCENARIOS = 1000 # Fallback if size not in NUM_RANDOM_SCENARIOS

# Random Seed for reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED) # Seed numpy as well

# Logging Setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler() # Log to console
        # Optionally add logging to a file:
        # logging.FileHandler(os.path.join(BASE_DIR, "recipe_kg_log.txt"))
    ]
)

# --- Helper Functions ---

def create_node_label(label):
    """
    Creates a standardized, valid node label from a string.

    Replaces spaces and hyphens with underscores, removes problematic characters,
    converts to lowercase, and handles non-string inputs.

    Args:
        label: The input label (string or other type).

    Returns:
        A standardized string label, or the string representation of the input
        if it's not a string initially.
    """
    if isinstance(label, str):
        # Added more robust cleaning for potential edge cases
        label = label.replace(" ", "_").replace("-", "_")
        label = ''.join(c for c in label if c.isalnum() or c == '_') # Keep only alphanum and underscore
        # Avoid labels starting/ending with underscore or being just underscore
        label = label.strip('_').lower()
        if not label: # Handle cases where cleaning results in empty string
            return UNKNOWN_PLACEHOLDER.lower() # Or some default
        # Prevent labels potentially conflicting with numeric IDs if any exist
        if label.isdigit():
            label = f"num_{label}"
        return label
    # Convert non-strings, including NaN, safely
    if pd.isna(label):
        return UNKNOWN_PLACEHOLDER.lower()
    return str(label)

def safe_split_and_clean(text, delimiter=','):
    """
    Safely splits a string by a delimiter and cleans each part using create_node_label.

    Handles potential NaN or non-string inputs gracefully. Filters out empty strings
    and placeholders resulting from cleaning.

    Args:
        text: The input text to split.
        delimiter: The delimiter character.

    Returns:
        A list of cleaned, standardized, non-empty, non-placeholder node labels.
    """
    if pd.isna(text) or not isinstance(text, str) or not text.strip():
        return []
    cleaned_labels = []
    for item in text.split(delimiter):
        cleaned = create_node_label(item.strip())
        # Exclude empty strings or placeholders after cleaning
        if cleaned and cleaned != UNKNOWN_PLACEHOLDER.lower():
            cleaned_labels.append(cleaned)
    return cleaned_labels

# --- Data Loading and Preprocessing ---

def load_and_preprocess_recipes(file_path):
    """
    Loads recipe data, cleans it, standardizes relevant columns into lists for KG.

    Args:
        file_path (str): Path to the recipe CSV file.

    Returns:
        pd.DataFrame: Preprocessed recipe DataFrame with original and list columns.
    """
    start_time = time.time()
    logging.info(f"Loading recipes from {file_path}...")
    try:
        df = pd.read_csv(file_path)
        logging.info(f"Initial recipe count: {len(df)}")
        df = df.drop_duplicates(subset='Name', keep='first').reset_index(drop=True)
        logging.info(f"Recipe count after dropping duplicates by name: {len(df)}")

        # Standardize node labels in relevant columns using list format for multi-value attributes
        logging.info("Standardizing labels and creating list columns...")
        df['Name_clean'] = df['Name'].apply(create_node_label) # Keep original Name if needed, use clean one for KG
        df['best_foodentityname_list'] = df['best_foodentityname'].apply(safe_split_and_clean)
        df['healthy_types_list'] = df['Healthy_Type'].apply(safe_split_and_clean)
        df['meal_types_list'] = df['meal_type'].apply(safe_split_and_clean)
        df['cook_time_label'] = df['cook_time'].apply(create_node_label) # Single value expected

        # Handle 'Diet_Types' specifically for the placeholder logic
        df['diet_types_list'] = df['Diet_Types'].apply(
            lambda x: safe_split_and_clean(x) if pd.notna(x) and str(x).strip() else [UNKNOWN_PLACEHOLDER.lower()]
        )
        # Filter out placeholder if other diet types exist
        df['diet_types_list'] = df['diet_types_list'].apply(
            lambda lst: [d for d in lst if d != UNKNOWN_PLACEHOLDER.lower()] if len(lst) > 1 else lst
        )

        df['region_countries_list'] = df['CleanedRegion'].apply(safe_split_and_clean)

        # Log counts of lists for a sample row (optional debugging)
        # logging.debug(f"Sample processed row lists:\n{df.iloc[0][['best_foodentityname_list', 'healthy_types_list', 'meal_types_list', 'diet_types_list', 'region_countries_list']]}")

        duration = time.time() - start_time
        logging.info(f"Recipe data loaded and preprocessed in {duration:.2f} seconds.")
        return df
    except FileNotFoundError:
        logging.error(f"Error: Recipe file not found at {file_path}")
        raise
    except Exception as e:
        logging.error(f"Error loading or preprocessing recipes: {e}")
        raise

def process_recipes_to_dict(df):
    """
    Converts the preprocessed DataFrame into a dictionary format suitable for graph creation.
    Uses the cleaned list/label columns.

    Args:
        df (pd.DataFrame): Preprocessed recipe DataFrame.

    Returns:
        dict: Dictionary where keys are cleaned recipe names and values are dicts
              of their attributes (lists or single labels).
    """
    recipes_dict = {}
    logging.info("Converting DataFrame to recipes dictionary...")
    for _, row in df.iterrows():
        # Use the cleaned recipe name as the primary key/node
        recipe_name_clean = row['Name_clean']
        if not recipe_name_clean or recipe_name_clean == UNKNOWN_PLACEHOLDER.lower():
             logging.warning(f"Skipping recipe with invalid cleaned name: Original='{row.get('Name', 'N/A')}'")
             continue

        recipes_dict[recipe_name_clean] = {
            # Attributes expected by create_graph_and_triples
            "ingredients": row['best_foodentityname_list'],
            "diet_types": row['diet_types_list'],
            "meal_type": row['meal_types_list'],
            "cook_time": row['cook_time_label'], # Single value expected
            "region_countries": row['region_countries_list'],
            "healthy_types": row['healthy_types_list'],
        }
    logging.info(f"Converted DataFrame to recipes dictionary with {len(recipes_dict)} entries.")
    return recipes_dict

# --- Graph and Triple Creation ---

def get_relation_for_healthy_type(element):
    """Determines the specific relation based on the healthy type string."""
    element_lower = element.lower() # Ensure case-insensitivity
    # More specific checks first
    if 'saturated_fat' in element_lower: return 'HasSaturatedFatLevel'
    if 'cholesterol' in element_lower: return 'HasCholesterolLevel'
    if 'protein' in element_lower: return 'HasProteinLevel'
    if 'carb' in element_lower: return 'HasCarbLevel'
    if 'fat' in element_lower: return 'HasFatLevel' # General fat after specific ones
    if 'calorie' in element_lower: return 'HasCalorieLevel'
    if 'sodium' in element_lower: return 'HasSodiumLevel'
    if 'sugar' in element_lower: return 'HasSugarLevel'
    if 'fiber' in element_lower: return 'HasFiberLevel'
    return 'HasHealthAttribute' # Default for others

def get_relation_for_attribute(relation_key):
    """Maps the original attribute key to a standardized relation name."""
    mapping = {
        'ingredients': 'contains',
        'diet_types': 'hasDietType',
        'meal_type': 'isForMealType',
        'cook_time': 'needTimeToCook',
        'region_countries': 'isFromRegion',
        'healthy_types': 'hasHealthAttribute' # Default for healthy, overridden by specific func
    }
    return mapping.get(relation_key, 'hasAttribute') # Default relation if key not found

def create_graph_and_triples(recipes_dict):
    """
    Creates a NetworkX graph and a NumPy array of triples from the recipes dictionary.

    Args:
        recipes_dict (dict): Dictionary of recipes (cleaned names) and their attributes.

    Returns:
        tuple: (networkx.Graph, numpy.ndarray) containing the graph and triples.
    """
    start_time = time.time()
    G = nx.Graph() # Using Undirected graph as in original code
    triples = []
    processed_nodes = set() # Track nodes added to avoid redundant type setting

    logging.info("Starting graph and triple creation...")
    for recipe_clean, details in recipes_dict.items():
        # Ensure recipe node exists
        if recipe_clean not in processed_nodes:
            G.add_node(recipe_clean, type='recipe')
            processed_nodes.add(recipe_clean)

        for relation_key, elements in details.items():
            # Ensure elements is iterable (list or wrap single element)
            items_to_process = elements if isinstance(elements, list) else [elements]

            for element in items_to_process:
                # Skip invalid elements (empty, placeholder)
                if not element or element == UNKNOWN_PLACEHOLDER.lower():
                    continue

                # Determine relation and node type based on the relation key
                if relation_key == 'healthy_types':
                    relation = get_relation_for_healthy_type(element)
                    node_type = relation # e.g., 'HasProteinLevel' nodes have this type
                else:
                    relation = get_relation_for_attribute(relation_key)
                    # Determine node type based on the key
                    node_type_mapping = {
                        'ingredients': 'ingredient',
                        'diet_types': 'diet_type',
                        'meal_type': 'meal_type',
                        'cook_time': 'cook_time',
                        'region_countries': 'region',
                    }
                    node_type = node_type_mapping.get(relation_key, 'attribute') # Default type

                # Add the attribute node if not already processed
                if element not in processed_nodes:
                    G.add_node(element, type=node_type)
                    processed_nodes.add(element)
                # elif G.nodes[element].get('type') != node_type: # Check if type needs updating (less likely with clean data)
                #     logging.warning(f"Node '{element}' found with multiple types: {G.nodes[element].get('type')} and {node_type}. Keeping first.")

                # Add edge and triple (Head, Relation, Tail)
                G.add_edge(recipe_clean, element, relation=relation)
                triples.append((recipe_clean, relation, element))

    triples_array = np.array(triples, dtype=str)
    duration = time.time() - start_time
    logging.info(f"Created graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
    logging.info(f"Generated {len(triples_array)} triples in {duration:.2f} seconds.")
    return G, triples_array

# --- Model Training ---

def train_kge_model(triples_factory, model_name, epochs, use_stopper):
    """
    Trains a PyKEEN Knowledge Graph Embedding model.

    Args:
        triples_factory (TriplesFactory): The PyKEEN triples factory.
        model_name (str): The name of the KGE model (e.g., 'RotatE').
        epochs (int): The number of training epochs.
        use_stopper (bool): Whether to use early stopping.

    Returns:
        PipelineResult: The result object from the PyKEEN pipeline.
    """
    start_time = time.time()
    logging.info(f"Starting training for model: {model_name}...")

    # Add model-specific or general training kwargs if defined in constants
    # model_kwargs = MODEL_KWARGS.get(model_name, {}) # Example
    # training_kwargs = TRAINING_KWARGS # Example

    pipeline_kwargs = dict(
        model=model_name,
        training=triples_factory,
        testing=triples_factory,  # Evaluate link prediction on the full graph
        validation=triples_factory, # Use full graph for validation if using early stopping
        epochs=epochs,
        random_seed=RANDOM_SEED, # Ensure reproducibility in training
        device='cuda', # Explicitly request CUDA if available
        # model_kwargs=model_kwargs, # Pass model-specific kwargs
        # training_kwargs=training_kwargs, # Pass training loop kwargs
    )
    if use_stopper:
        pipeline_kwargs['stopper'] = 'early'
        # Configure stopper if needed:
        # pipeline_kwargs['stopper_kwargs'] = dict(patience=5, delta=0.001)

    try:
        result = pipeline(**pipeline_kwargs)
        duration = time.time() - start_time
        logging.info(f"Training finished for model: {model_name} in {duration:.2f} seconds.")

        # Log standard evaluation metrics from the pipeline result
        logging.info(f"--- {model_name} Standard Evaluation Metrics ---")
        mr_results = result.metric_results
        hits_metrics = {k: mr_results.get_metric(f'hits_at_{k}') for k in [1, 3, 10]}
        mr_metric = mr_results.get_metric('mean_rank') # Check specific key if needed e.g., 'both.realistic.mean_rank'
        mrr_metric = mr_results.get_metric('mean_reciprocal_rank') # Check specific key e.g., 'both.realistic.mean_reciprocal_rank'

        for k, v in hits_metrics.items():
            logging.info(f"  Hits@{k:<2}: {v:.4f}" if v is not None else f"  Hits@{k:<2}: N/A")
        logging.info(f"  Mean Rank: {mr_metric:.2f}" if mr_metric is not None else "  Mean Rank: N/A")
        logging.info(f"  MRR      : {mrr_metric:.4f}" if mrr_metric is not None else "  MRR      : N/A")
        logging.info("-------------------------------------------")

        return result

    except Exception as e:
         # Catch potential errors during pipeline execution (e.g., CUDA issues)
        logging.error(f"Error during pipeline training for {model_name}: {e}")
        # Depending on severity, you might want to raise e or return None
        return None

# --- Scenario Generation and Evaluation ---

def get_relation_options(recipes_df, top_n_ingredients):
    """
    Defines the possible relation values (entities) based on the dataset columns.
    Uses the cleaned list/label columns.

    Args:
        recipes_df (pd.DataFrame): The preprocessed recipe DataFrame.
        top_n_ingredients (int): How many top ingredients to include.

    Returns:
        dict: A dictionary where keys are relation names and values are lists
              of possible entity values (cleaned labels) for that relation.
    """
    logging.info("Defining relation options for scenario generation...")
    # Extract unique, cleaned values from list columns or label columns
    relation_options = {}

    # Health Levels (assuming fixed categories based on cleaning)
    health_relations = [
        "HasProteinLevel", "HasCarbLevel", "HasFatLevel", "HasSaturatedFatLevel",
        "HasFiberLevel", "HasSodiumLevel", "HasSugarLevel", "HasCholesterolLevel",
        "HasCalorieLevel"
    ]
    for rel in health_relations:
         # Assuming labels like 'low_protein', 'medium_protein', 'high_protein' exist after cleaning
        base_type = rel.replace("Has", "").replace("Level", "").lower()
        relation_options[rel] = [f"low_{base_type}", f"medium_{base_type}", f"high_{base_type}"] # Example structure

    # Other attributes from lists/labels
    relation_options["isForMealType"] = sorted(list(set(item for sublist in recipes_df['meal_types_list'] for item in sublist)))
    # Exclude placeholder explicitly when defining options
    relation_options["hasDietType"] = sorted(list(set(item for sublist in recipes_df['diet_types_list'] for item in sublist if item != UNKNOWN_PLACEHOLDER.lower())))
    relation_options["isFromRegion"] = sorted(list(set(item for sublist in recipes_df['region_countries_list'] for item in sublist)))
    relation_options["needTimeToCook"] = sorted(list(recipes_df['cook_time_label'].unique()))

    # Get top N ingredients from the list column
    try:
        ingredient_counts = pd.Series(
            item for sublist in recipes_df['best_foodentityname_list'] if isinstance(sublist, list) for item in sublist
        ).value_counts()
        top_ingredients_list = ingredient_counts.head(top_n_ingredients).index.tolist()
        relation_options["contains"] = top_ingredients_list
    except Exception as e:
        logging.warning(f"Could not extract top ingredients: {e}. 'contains' relation might be empty.")
        relation_options["contains"] = []


    # Log counts for verification and remove empty option lists
    final_relation_options = {}
    for key, values in relation_options.items():
        if values:
             final_relation_options[key] = values
             logging.debug(f"Relation '{key}': {len(values)} options. Example: {values[:3]}")
        else:
             logging.warning(f"Relation '{key}' has no options defined. It will not be used in scenario generation.")

    return final_relation_options


def generate_test_scenarios(relation_options, combination_size, num_samples):
    """
    Generates random test scenarios (combinations of criteria).
    Uses itertools.combinations for relations and itertools.product for values,
    then samples using random.sample for unique scenarios.

    Args:
        relation_options (dict): Possible values for each relation.
        combination_size (int): The number of criteria per scenario (e.g., 2 for pairs).
        num_samples (int): The desired number of unique random scenarios.

    Returns:
        list: A list of test scenarios, where each scenario is a list of
              (value, relation) tuples. Returns empty list if generation fails.
    """
    logging.info(f"Generating scenarios with {combination_size} criteria (target: {num_samples})...")
    start_time = time.time()
    all_possible_scenarios = []
    valid_relations_items = list(relation_options.items()) # Use relations with non-empty options

    if len(valid_relations_items) < combination_size:
        logging.warning(f"Cannot generate combinations of size {combination_size} "
                        f"with only {len(valid_relations_items)} valid relations available.")
        return []

    # Generate all possible combinations of *relations*
    for relation_comb in combinations(valid_relations_items, combination_size):
        # Get the possible values for the chosen relations
        value_options_for_comb = [relation[1] for relation in relation_comb]

        # Check if any relation in the combination has empty values (shouldn't happen with pre-filtering)
        if not all(value_options_for_comb):
            logging.warning(f"Skipping relation combination due to empty value list: {relation_comb}")
            continue

        # Generate all combinations of *values* for the chosen relations
        for values_comb in product(*value_options_for_comb):
            # Create the test case (scenario) as a list of tuples
            test_case = list(zip(values_comb, [relation[0] for relation in relation_comb]))
            all_possible_scenarios.append(test_case)

    duration_gen = time.time() - start_time
    logging.info(f"Generated {len(all_possible_scenarios)} total possible unique combinations in {duration_gen:.2f}s.")

    # Sample randomly if the total number of combinations is more than requested
    if not all_possible_scenarios:
        logging.warning(f"No scenarios could be generated for combination size {combination_size}.")
        return []
    elif len(all_possible_scenarios) > num_samples:
        logging.info(f"Sampling {num_samples} scenarios randomly...")
        selected_scenarios = random.sample(all_possible_scenarios, num_samples)
        logging.info(f"Randomly sampled {len(selected_scenarios)} unique scenarios.")
        return selected_scenarios
    else:
         # Use all generated scenarios if fewer than requested
         logging.info(f"Using all {len(all_possible_scenarios)} generated scenarios.")
         return all_possible_scenarios


def calculate_average_occurrence(scenarios, recipes_df):
    """
    Calculates the average number of recipes matching the scenarios in the dataset.
    This helps determine a reasonable 'N' for Top-N evaluation (Precision@N).
    Uses pre-calculated list/label columns for filtering.

    Args:
        scenarios (list): List of test scenarios.
        recipes_df (pd.DataFrame): The preprocessed recipe DataFrame.

    Returns:
        int: The calculated average occurrence (rounded down, minimum 1).
    """
    if not scenarios:
        logging.warning("Cannot calculate average occurrence: No scenarios provided.")
        return 1 # Return a default value > 0

    logging.info(f"Calculating average occurrence for {len(scenarios)} scenarios...")
    total_matching_recipes = 0
    scenarios_with_matches = 0
    zero_match_scenarios_count = 0
    start_time = time.time()

    # Use .apply with set intersection or direct comparison for potentially faster filtering
    for scenario in scenarios:
        # Start with an index of all True
        match_index = pd.Series([True] * len(recipes_df))

        for value, relation in scenario:
            if not match_index.any(): # Optimization: if no matches remain, stop filtering
                 break

            try:
                # Apply filters based on relation type using pre-calculated lists/labels
                if relation.endswith("Level") or relation == "HasHealthAttribute":
                    match_index &= recipes_df['healthy_types_list'].apply(lambda lst: value in lst if isinstance(lst, list) else False)
                elif relation == "isForMealType":
                    match_index &= recipes_df['meal_types_list'].apply(lambda lst: value in lst if isinstance(lst, list) else False)
                elif relation == "hasDietType":
                    match_index &= recipes_df['diet_types_list'].apply(lambda lst: value in lst if isinstance(lst, list) else False)
                elif relation == "isFromRegion":
                    match_index &= recipes_df['region_countries_list'].apply(lambda lst: value in lst if isinstance(lst, list) else False)
                elif relation == "needTimeToCook":
                    match_index &= (recipes_df['cook_time_label'] == value)
                elif relation == "contains":
                     match_index &= recipes_df['best_foodentityname_list'].apply(lambda lst: value in lst if isinstance(lst, list) else False)
                else:
                    logging.warning(f"Unrecognized relation '{relation}' in scenario: {scenario}. Skipping this criterion in average calculation.")
            except KeyError as e:
                 logging.error(f"Missing expected column '{e}' in DataFrame during average calculation. Skipping criterion {relation}.")
                 match_index &= False # Invalidate matches for this scenario if column missing
                 break # Stop processing this scenario

        count = match_index.sum()

        if count > 0:
            total_matching_recipes += count
            scenarios_with_matches += 1
        else:
            # Log only if needed, can be verbose
            # logging.debug(f"Scenario resulted in 0 matches: {scenario}")
            zero_match_scenarios_count += 1

    duration = time.time() - start_time
    logging.info(f"Finished calculating occurrences in {duration:.2f} seconds.")

    if zero_match_scenarios_count > 0:
        logging.warning(f"{zero_match_scenarios_count} scenarios resulted in 0 matches in the dataset during average calculation.")

    if scenarios_with_matches > 0:
        average = total_matching_recipes / scenarios_with_matches
        # Ensure N is at least 1
        avg_occurrence_int = max(1, int(average))
        logging.info(f"Average occurrence calculated: {average:.2f}, using N={avg_occurrence_int} for Precision@N.")
        return avg_occurrence_int
    else:
        logging.warning("No scenarios had any matches in the dataset. Defaulting average occurrence (N for Precision) to 1.")
        return 1


def filter_recipes_by_scenario(recipes_df, scenario):
    """
    Filters the DataFrame to find recipes matching all criteria in a scenario.
    Uses pre-calculated list/label columns.

    Args:
        recipes_df (pd.DataFrame): The preprocessed recipe DataFrame.
        scenario (list): The test scenario [(value, relation), ...].

    Returns:
        pd.DataFrame: Filtered DataFrame containing only matching recipes.
                      Returns empty DataFrame if no matches or error.
    """
    # Start with an index of all True
    match_index = pd.Series([True] * len(recipes_df))

    for tail_entity, relation in scenario:
        if not match_index.any(): # Optimization
            break
        try:
            # Apply filters similarly to calculate_average_occurrence
            if relation.endswith("Level") or relation == "HasHealthAttribute":
                match_index &= recipes_df['healthy_types_list'].apply(lambda lst: tail_entity in lst if isinstance(lst, list) else False)
            elif relation == "isForMealType":
                match_index &= recipes_df['meal_types_list'].apply(lambda lst: tail_entity in lst if isinstance(lst, list) else False)
            elif relation == "hasDietType":
                match_index &= recipes_df['diet_types_list'].apply(lambda lst: tail_entity in lst if isinstance(lst, list) else False)
            elif relation == "isFromRegion":
                 match_index &= recipes_df['region_countries_list'].apply(lambda lst: tail_entity in lst if isinstance(lst, list) else False)
            elif relation == "needTimeToCook":
                match_index &= (recipes_df['cook_time_label'] == tail_entity)
            elif relation == "contains":
                match_index &= recipes_df['best_foodentityname_list'].apply(lambda lst: tail_entity in lst if isinstance(lst, list) else False)
            else:
                logging.warning(f"Unrecognized relation '{relation}' during ground truth filtering for scenario: {scenario}. Skipping criterion.")
        except KeyError as e:
            logging.error(f"Missing expected column '{e}' in DataFrame during ground truth filtering. Scenario results might be inaccurate.")
            return pd.DataFrame() # Return empty if crucial column missing

    return recipes_df[match_index]


def evaluate_scenario_predictions(scenario, ground_truth_df, model_result, top_n_precision):
    """
    Predicts recipes for a scenario using the KGE model, compares with ground truth,
    and calculates Precision@N, Recall@K, F1@N/K, Accuracy@K.

    Args:
        scenario (list): The test scenario [(value, relation), ...].
        ground_truth_df (pd.DataFrame): DataFrame of recipes actually matching the scenario.
                                        Must contain the 'Name_clean' column.
        model_result (PipelineResult): The trained PyKEEN model result.
        top_n_precision (int): The 'N' value for calculating Precision@N (based on avg occurrence).

    Returns:
        dict: A dictionary containing evaluation metrics and counts for the scenario.
              Returns None if ground truth is empty or prediction fails.
    """
    expected_match_count = len(ground_truth_df)
    # K for Recall and Accuracy is the number of actual relevant items
    top_n_recall_accuracy = max(1, expected_match_count)
    # N for Precision is based on average occurrence
    top_n_precision = max(1, top_n_precision)

    if expected_match_count == 0:
        # This scenario has no real matches in the data, cannot calculate meaningful Recall/Accuracy.
        # We could potentially still calculate Precision@N (if predictions exist), but it's often skipped.
        # logging.debug(f"Skipping evaluation for scenario with 0 ground truth matches: {scenario}")
        return None

    # --- Perform Prediction using KGE Model ---
    all_predictions = []
    try:
        for tail_entity, relation in scenario:
            # Check if entities/relations exist in the model's mapping before predicting
            if relation not in model_result.model.relation_to_id:
                 logging.warning(f"Relation '{relation}' not in model mapping. Skipping criterion in prediction for {scenario}.")
                 continue
            if tail_entity not in model_result.model.entity_to_id:
                 logging.warning(f"Entity '{tail_entity}' not in model mapping. Skipping criterion in prediction for {scenario}.")
                 continue

            # Predict heads (recipes) for the given relation and tail entity
            predicted_heads_df = predict_target(
                model=model_result.model,
                relation=relation,
                tail=tail_entity,
                triples_factory=model_result.training # Use training factory for id mappings
            ).df
            # Filter out non-recipe predictions if head_label might include other entity types
            # Assuming recipe names were stored in 'Name_clean' and used as keys in recipes_dict
            # We need the set of all known recipe names from the triples factory or df
            # Note: This assumes predict_target returns labels matching the training data.
            # predicted_heads_df = predicted_heads_df[predicted_heads_df['head_label'].isin(recipes_dict.keys())] # Filter if necessary

            all_predictions.append(predicted_heads_df)

    except Exception as e:
        logging.error(f"Error during KGE prediction for scenario {scenario}: {e}")
        return None # Skip scenario if prediction fails

    if not all_predictions:
        # No valid predictions could be made (e.g., all criteria involved unknown entities/relations)
        logging.warning(f"No KGE predictions generated for scenario: {scenario}")
        # Return metrics as 0, as no predictions mean no true positives.
        return {
            "Precision": 0.0, "Recall": 0.0, "F1 Score": 0.0, "Accuracy": 0.0,
            "Expected Matches (K)": expected_match_count, "Top N for Precision": top_n_precision,
            "Predicted Count (Top N Precision)": 0, "Predicted Count (Top K Recall/Acc)": 0,
            "True Positives (Precision N)": 0, "True Positives (Recall/Acc K)": 0
         }

    # Aggregate scores (summing scores for recipes predicted by multiple criteria)
    all_predictions_df = pd.concat(all_predictions, ignore_index=True)
    # Group by the predicted head label (recipe name) and sum the scores
    aggregated_predictions = all_predictions_df.groupby('head_label').agg(
        total_score=('score', 'sum') # Sum scores if a recipe matches multiple criteria
    ).reset_index()
    # Sort recipes by the aggregated score in descending order
    final_predictions_sorted = aggregated_predictions.sort_values(by="total_score", ascending=False)

    # --- Calculate Metrics ---
    # Get the sets of predicted recipe names for Precision@N and Recall@K/Accuracy@K
    predicted_recipes_precision_set = set(final_predictions_sorted["head_label"].head(top_n_precision))
    predicted_recipes_recall_accuracy_set = set(final_predictions_sorted["head_label"].head(top_n_recall_accuracy))

    # Get the set of ground truth (relevant) recipe names (using the cleaned names)
    try:
        relevant_recipes_set = set(ground_truth_df['Name_clean'])
    except KeyError:
        logging.error("Ground truth DataFrame missing 'Name_clean' column. Cannot calculate metrics.")
        return None

    # Calculate True Positives for both N and K
    true_positives_precision_n = predicted_recipes_precision_set.intersection(relevant_recipes_set)
    true_positives_recall_accuracy_k = predicted_recipes_recall_accuracy_set.intersection(relevant_recipes_set)

    # Precision@N = TP@N / N (or / |Predicted@N| if N > |Predicted|)
    # Using |Predicted@N| as denominator is common: TP@N / |Predicted@N|
    precision_at_n = len(true_positives_precision_n) / len(predicted_recipes_precision_set) if predicted_recipes_precision_set else 0.0

    # Recall@K = TP@K / |Relevant|
    recall_at_k = len(true_positives_recall_accuracy_k) / len(relevant_recipes_set) if relevant_recipes_set else 0.0

    # F1 Score: Harmonic mean of Precision@N and Recall@K
    # Note: This combines metrics based on potentially different cutoff values (N and K).
    # Consider if calculating F1 based on a single cutoff (e.g., K) is more appropriate.
    # F1 = 2 * (Precision@K * Recall@K) / (Precision@K + Recall@K) is another option.
    # Let's stick to the original approach (using Precision@N and Recall@K).
    f1 = 2 * (precision_at_n * recall_at_k) / (precision_at_n + recall_at_k) if (precision_at_n + recall_at_k) > 0 else 0.0

    # Accuracy@K = TP@K / K
    accuracy_at_k = len(true_positives_recall_accuracy_k) / top_n_recall_accuracy if top_n_recall_accuracy > 0 else 0.0

    # Store results in a dictionary
    result_dict = {
        "Expected Matches (K)": expected_match_count, # K = number of relevant items
        "Top N for Precision": top_n_precision,       # N based on average occurrence
        "Predicted Count (Top N Precision)": len(predicted_recipes_precision_set),
        "Predicted Count (Top K Recall/Acc)": len(predicted_recipes_recall_accuracy_set),
        "True Positives (Precision N)": len(true_positives_precision_n),
        "True Positives (Recall/Acc K)": len(true_positives_recall_accuracy_k),
        "Precision": precision_at_n, # Precision @ N
        "Recall": recall_at_k,       # Recall @ K
        "F1 Score": f1,              # F1 based on P@N, R@K
        "Accuracy": accuracy_at_k    # Accuracy @ K
    }

    return result_dict

# --- Main Execution ---

def main():
    """Main function to run the complete workflow."""
    overall_start_time = time.time()
    logging.info("Starting main execution workflow...")

    # Ensure result directories exist
    os.makedirs(ROTAT_E_RESULTS_DIR, exist_ok=True)
    os.makedirs(QUAT_E_RESULTS_DIR, exist_ok=True)
    logging.info(f"Ensured result directories exist: {ROTAT_E_RESULTS_DIR}, {QUAT_E_RESULTS_DIR}")

    # 1. Load and Preprocess Data
    try:
        recipes_df = load_and_preprocess_recipes(INPUT_RECIPE_FILE)
        recipes_dict = process_recipes_to_dict(recipes_df)
    except Exception as e:
        logging.critical(f"Failed to load or process recipe data: {e}. Exiting.")
        return # Stop execution if data loading fails

    # 2. Create Graph and Triples
    try:
        _, triples_array = create_graph_and_triples(recipes_dict)
        if triples_array.size == 0:
            logging.critical("No triples were generated. Cannot proceed. Check data and KG creation logic.")
            return
        triples_df = pd.DataFrame(triples_array, columns=['Head', 'Relation', 'Tail'])
        triples_df = triples_df.dropna().drop_duplicates() # Add drop_duplicates for extra safety
        logging.info(f"Final triple count after dropna/duplicates: {len(triples_df)}")

        # Save triples (optional but good practice)
        triples_df.to_csv(TRIPLES_FILE, index=False)
        logging.info(f"Cleaned triples saved to {TRIPLES_FILE}")
    except Exception as e:
        logging.critical(f"Failed to create graph or triples: {e}. Exiting.")
        return

    # 3. Create Triples Factory
    try:
        # Load from saved file or use DataFrame directly
        df_for_factory = pd.read_csv(TRIPLES_FILE) # Reloading ensures consistency
        triples = df_for_factory[['Head', 'Relation', 'Tail']].values
        triples_factory = TriplesFactory.from_labeled_triples(triples)
        logging.info(f"TriplesFactory created with {triples_factory.num_entities} entities, "
                     f"{triples_factory.num_relations} relations, {triples_factory.num_triples} triples.")
    except Exception as e:
        logging.critical(f"Failed to create TriplesFactory: {e}. Exiting.")
        return

    # 4. Define Relation Options for Scenarios
    try:
        relation_options = get_relation_options(recipes_df, TOP_N_INGREDIENTS)
        if not relation_options:
             logging.critical("No relation options could be defined. Cannot generate scenarios. Exiting.")
             return
    except Exception as e:
        logging.critical(f"Failed to define relation options: {e}. Exiting.")
        return


    # 5. Loop through Models and Criteria Sizes for Training and Evaluation
    all_model_results = {} # Store trained model results (PipelineResult objects)

    for model_name in KGE_MODELS:
        logging.info(f"===== Processing Model: {model_name} =====")
        # Train the model
        model_result = train_kge_model(
            triples_factory,
            model_name,
            epochs=EPOCHS,
            use_stopper=EARLY_STOPPING
        )

        if model_result is None:
            logging.error(f"Training failed for model {model_name}. Skipping evaluation.")
            continue # Skip to the next model if training failed

        all_model_results[model_name] = model_result

        # Determine output directory based on model
        output_dir = ROTAT_E_RESULTS_DIR if model_name == 'RotatE' else QUAT_E_RESULTS_DIR

        # Evaluate for different criteria sizes defined for this model
        if model_name not in CRITERIA_SIZES:
            logging.warning(f"No criteria sizes defined in CRITERIA_SIZES for model {model_name}. Skipping evaluation.")
            continue

        for size in CRITERIA_SIZES[model_name]:
            logging.info(f"--- Evaluating {model_name} with {size} criteria ---")
            model_size_start_time = time.time()

            num_scenarios_for_size = NUM_RANDOM_SCENARIOS.get(size, DEFAULT_NUM_SCENARIOS)
            test_scenarios = generate_test_scenarios(relation_options, size, num_scenarios_for_size)

            if not test_scenarios:
                logging.warning(f"No test scenarios generated for {model_name} with {size} criteria. Skipping.")
                continue

            # Calculate average occurrence (for Precision@N) based on these scenarios
            avg_occurrence_n = calculate_average_occurrence(test_scenarios, recipes_df)

            # Evaluate each scenario using the trained model
            evaluation_results_list = []
            scenarios_evaluated = 0
            skipped_scenarios_no_gt = 0 # Count scenarios skipped due to 0 ground truth

            for idx, scenario in enumerate(test_scenarios):
                # Find ground truth recipes matching the scenario
                ground_truth_recipes = filter_recipes_by_scenario(recipes_df, scenario)

                # Predict and evaluate
                scenario_metrics = evaluate_scenario_predictions(
                    scenario,
                    ground_truth_recipes,
                    model_result,
                    avg_occurrence_n # Pass N for Precision@N
                )

                if scenario_metrics:
                    scenario_metrics["Scenario Index"] = idx + 1 # Add index for reference
                    scenario_metrics["Criteria"] = str(scenario) # Store scenario criteria as string
                    evaluation_results_list.append(scenario_metrics)
                    scenarios_evaluated += 1
                elif len(ground_truth_recipes) == 0:
                    skipped_scenarios_no_gt += 1
                # else: scenario skipped due to prediction error (already logged)

                # Log progress periodically
                if (idx + 1) % 100 == 0:
                    logging.info(f"  Processed {idx + 1}/{len(test_scenarios)} scenarios for {model_name}/{size} criteria...")


            # Save results for this model and criteria size
            if evaluation_results_list:
                results_df = pd.DataFrame(evaluation_results_list)
                # Define column order for better readability
                cols_order = [
                    "Scenario Index", "Criteria", "Expected Matches (K)", "Top N for Precision",
                    "Predicted Count (Top N Precision)", "Predicted Count (Top K Recall/Acc)",
                    "True Positives (Precision N)", "True Positives (Recall/Acc K)",
                    "Precision", "Recall", "F1 Score", "Accuracy"
                ]
                # Ensure all expected columns are present, add missing ones with NaN if necessary
                for col in cols_order:
                    if col not in results_df.columns:
                        results_df[col] = pd.NA
                results_df = results_df[cols_order] # Reorder

                output_filename = os.path.join(output_dir, f'{model_name.lower()}_{size}_criteria_results.csv')
                try:
                    results_df.to_csv(output_filename, index=False)
                    logging.info(f"Evaluation results ({scenarios_evaluated} scenarios) saved to {output_filename}")
                    # Log summary statistics for the evaluated scenarios
                    logging.info(f"  Avg Precision: {results_df['Precision'].mean():.4f}")
                    logging.info(f"  Avg Recall:    {results_df['Recall'].mean():.4f}")
                    logging.info(f"  Avg F1 Score:  {results_df['F1 Score'].mean():.4f}")
                    logging.info(f"  Avg Accuracy:  {results_df['Accuracy'].mean():.4f}")
                    if skipped_scenarios_no_gt > 0:
                         logging.info(f"  ({skipped_scenarios_no_gt} scenarios skipped due to 0 ground truth matches)")

                except Exception as e:
                    logging.error(f"Failed to save results to {output_filename}: {e}")
            else:
                logging.warning(f"No evaluation results generated (or all scenarios skipped) for {model_name} with {size} criteria.")

            model_size_duration = time.time() - model_size_start_time
            logging.info(f"--- Finished evaluation for {model_name} with {size} criteria in {model_size_duration:.2f} seconds ---")

        logging.info(f"===== Finished Processing Model: {model_name} =====")

    overall_duration = time.time() - overall_start_time
    logging.info(f"Script execution finished in {overall_duration:.2f} seconds.")

# --- Entry Point ---
if __name__ == "__main__":
    main()
