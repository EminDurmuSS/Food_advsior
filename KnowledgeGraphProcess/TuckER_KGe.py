# -*- coding: utf-8 -*-
"""
Refactored script for building a recipe knowledge graph, training a KGE model (TuckER),
and evaluating its performance on recommendation scenarios with varying criteria complexity.
"""

import logging
import os
import random
from itertools import combinations, product
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from pykeen.pipeline import PipelineResult, pipeline
from pykeen.predict import predict_target
from pykeen.triples import TriplesFactory

# --- Configuration & Constants ---

# File Paths
BASE_DIR = Path("/app") # Use Pathlib for better path handling
INPUT_CSV_PATH = BASE_DIR / "BalancedRecipe_entity_linking.csv"
TRIPLES_OUTPUT_PATH = BASE_DIR / "recipes_triples_clean.csv"
RESULTS_DIR = BASE_DIR / "TuckER_EvaluationResults_Clean"

# Data Processing
UNKNOWN_PLACEHOLDER = "unknown" # Consistent naming
TOP_N_INGREDIENTS = 20 # Number of top ingredients to consider for scenarios

# Model Training & Evaluation
KGE_MODEL = 'TuckER'
EPOCHS = 200 # Keep moderate for reasonable training time, adjust as needed
EARLY_STOPPING_PATIENCE = 5 # Example: Stop if no improvement after 5 epochs
RANDOM_SEED = 42
MAX_CRITERIA_COMBINATIONS = 8 # Evaluate scenarios from 1 to this number of criteria
NUM_RANDOM_SAMPLES_PER_SIZE = 10000 # Reduced for faster testing, increase back to 25k/50k if needed

# Mapping for relation names (more maintainable)
RELATION_MAPPING = {
    'ingredients': 'contains',
    'diet_types': 'hasDietType',
    'meal_type': 'isForMealType',
    'cook_time': 'needTimeToCook',
    'region_countries': 'isFromRegion',
    'healthy_types': 'hasHealthAttribute' # Default for health types
}

# Specific Health Relation Mapping (Case-insensitive prefixes)
HEALTH_RELATION_PREFIX_MAP = {
    'protein': 'HasProteinLevel',
    'carb': 'HasCarbLevel',
    'fat': 'HasFatLevel', # General fat
    'saturated fat': 'HasSaturatedFatLevel', # Specific saturated fat
    'calorie': 'HasCalorieLevel',
    'sodium': 'HasSodiumLevel',
    'sugar': 'HasSugarLevel',
    'fiber': 'HasFiberLevel',
    'cholesterol': 'HasCholesterolLevel',
}

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Helper Functions ---

def create_node_label(label: Any) -> str:
    """
    Cleans and standardizes a label to be used as a node name in the graph.
    Handles potential non-string inputs and replaces problematic characters.
    """
    if pd.isna(label):
        return UNKNOWN_PLACEHOLDER
    label_str = str(label).strip().lower()
    # Replace common problematic characters for node names/URIs
    label_str = label_str.replace(" ", "_").replace("-", "_")
    label_str = label_str.replace(">", "greater_than_").replace("<", "less_than_")
    # Remove any other potentially invalid characters (adjust regex as needed)
    # label_str = re.sub(r'[^\w_]', '', label_str) # Example: Keep only word chars and underscore
    return label_str if label_str else UNKNOWN_PLACEHOLDER

def safe_split_and_clean(data: Any, delimiter: str = ',') -> List[str]:
    """Safely splits a string (if not NaN) and cleans each resulting label."""
    if pd.isna(data):
        return []
    return [create_node_label(item.strip()) for item in str(data).split(delimiter) if item.strip()]

def get_health_relation(health_type_label: str) -> str:
    """Determines the specific relation type based on the health attribute label."""
    health_type_lower = health_type_label.lower()
    for prefix, relation in HEALTH_RELATION_PREFIX_MAP.items():
        # Check if the label *starts* with the prefix (more specific)
        # or just contains it if prefix matching is too strict
        if health_type_lower.startswith(prefix) or prefix in health_type_lower:
             # Handle cases like "low_saturated_fat" should map correctly
            if prefix == 'fat' and 'saturated fat' in health_type_lower:
                continue # Let 'saturated fat' handle this
            return relation
    # Fallback relation if no specific prefix matches
    return RELATION_MAPPING['healthy_types']

def load_and_preprocess_recipes(file_path: Path) -> pd.DataFrame:
    """Loads recipe data, drops duplicates by name, and resets index."""
    logging.info(f"Loading recipes from: {file_path}")
    try:
        df = pd.read_csv(file_path)
        logging.info(f"Initial recipe count: {len(df)}")
        df = df.drop_duplicates(subset='Name', keep='first').reset_index(drop=True)
        logging.info(f"Recipe count after dropping duplicates: {len(df)}")
        return df
    except FileNotFoundError:
        logging.error(f"Error: Input CSV file not found at {file_path}")
        raise
    except Exception as e:
        logging.error(f"Error loading or preprocessing CSV: {e}")
        raise

def create_graph_and_triples(df: pd.DataFrame) -> Tuple[nx.Graph, np.ndarray]:
    """
    Processes the DataFrame to create a NetworkX graph and a NumPy array of triples.
    """
    logging.info("Creating graph and extracting triples...")
    G = nx.Graph()
    triples_list = []

    required_columns = ['Name', 'best_foodentityname', 'Healthy_Type', 'meal_type', 'cook_time', 'Diet_Types', 'CleanedRegion']
    if not all(col in df.columns for col in required_columns):
        missing = [col for col in required_columns if col not in df.columns]
        raise ValueError(f"Missing required columns in DataFrame: {missing}")

    for _, row in df.iterrows():
        recipe_name = create_node_label(row['Name'])
        if recipe_name == UNKNOWN_PLACEHOLDER:
            continue # Skip recipes with unknown names

        G.add_node(recipe_name, type='recipe')

        details = {
            "ingredients": safe_split_and_clean(row['best_foodentityname']),
            "diet_types": safe_split_and_clean(row.get('Diet_Types', '')), # Use .get for safety
            "meal_type": safe_split_and_clean(row['meal_type']),
            "cook_time": [create_node_label(row['cook_time'])], # Treat as list for consistency
            "region_countries": safe_split_and_clean(row['CleanedRegion']),
            "healthy_types": safe_split_and_clean(row['Healthy_Type']),
        }

        # Special handling for diet_types if it becomes empty after cleaning
        if not details["diet_types"] and pd.notna(row.get('Diet_Types', '')):
             # If original Diet_Types existed but cleaned to nothing, maybe add Unknown?
             # Or decide based on requirements. Here, we add Unknown if it was specified.
             if UNKNOWN_PLACEHOLDER in str(row.get('Diet_Types', '')).lower():
                 details["diet_types"] = [UNKNOWN_PLACEHOLDER]


        for relation_key, elements in details.items():
            for element in elements:
                if element == UNKNOWN_PLACEHOLDER or not element:
                    continue # Skip unknown or empty elements

                # Determine relation type
                if relation_key == 'healthy_types':
                    relation = get_health_relation(element)
                    node_type = relation # Use the specific relation as node type
                else:
                    relation = RELATION_MAPPING.get(relation_key, 'hasAttribute') # Default relation
                    node_type = relation_key # e.g., 'ingredients', 'diet_types'

                # Add node and edge to graph
                G.add_node(element, type=node_type)
                G.add_edge(recipe_name, element, relation=relation)

                # Add triple to list
                triples_list.append((recipe_name, relation, element))

    triples_array = np.array(triples_list, dtype=str)
    logging.info(f"Graph created with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
    logging.info(f"Extracted {len(triples_array)} triples.")
    return G, triples_array

def save_triples(triples_array: np.ndarray, output_path: Path):
    """Saves the triples array to a CSV file."""
    logging.info(f"Saving triples to: {output_path}")
    triples_df = pd.DataFrame(triples_array, columns=['Head', 'Relation', 'Tail'])
    triples_df.dropna(inplace=True) # Ensure no NaN rows are saved
    triples_df.to_csv(output_path, index=False)
    logging.info("Triples saved successfully.")

def train_kge_model(triples_factory: TriplesFactory, model_name: str, epochs: int, results_dir: Path, early_stopping_patience: int) -> PipelineResult:
    """Trains a KGE model using the PyKEEN pipeline."""
    logging.info(f"Starting KGE model training (Model: {model_name}, Epochs: {epochs})...")

    # Ensure results directory exists for model checkpointing etc.
    results_dir.mkdir(parents=True, exist_ok=True)

    # Use the same factory for training, validation, and testing as per original code
    # Note: This means evaluation metrics reflect performance on the *training* data,
    # which isn't standard practice for generalization assessment.
    # For real-world evaluation, use triples_factory.split()
    pipeline_kwargs = dict(
        model=model_name,
        training=triples_factory,
        testing=triples_factory,
        validation=triples_factory, # Using training data for validation too
        training_kwargs=dict(num_epochs=epochs),
        stopper='early',
        stopper_kwargs=dict(
            frequency=5, # Check every 5 epochs
            patience=early_stopping_patience,
            metric='hits@10', # Metric to monitor for early stopping
            relative_delta=0.002 # Minimum improvement threshold
            ),
        evaluation_kwargs=dict(batch_size=128), # Adjust batch size as needed
        random_seed=RANDOM_SEED,
        # device='cuda' # Uncomment if GPU is available and configured
    )

    result = pipeline(**pipeline_kwargs)

    logging.info("KGE model training completed.")
    logging.info("--- Training Performance Metrics ---")
    # Print key metrics (adjust which metrics are most important)
    for metric in ['hits_at_1', 'hits_at_3', 'hits_at_10', 'mean_rank', 'mean_reciprocal_rank']:
        try:
            # Access metrics correctly (they might be nested under evaluation splits)
            # Assuming evaluation on 'testing' set here which is same as training
            metric_key_realistic = f'testing.realistic.{metric}'
            metric_value = result.metric_results.get_metric(name=metric_key_realistic)
            # Check if value is NaN or None before formatting
            if pd.notna(metric_value):
                 logging.info(f"{metric.replace('_', ' ').title()} (Realistic): {metric_value:.4f}")
            else:
                 logging.info(f"{metric.replace('_', ' ').title()} (Realistic): Not Available")

        except KeyError:
            logging.warning(f"Metric '{metric}' not found in results.")
    logging.info("---------------------------------")

    # Save the trained model and results
    model_save_dir = results_dir / f"{model_name}_model"
    result.save_to_directory(model_save_dir)
    logging.info(f"Trained model and results saved to: {model_save_dir}")

    return result

def preprocess_dataframe_for_evaluation(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocesses the DataFrame specifically for evaluation filtering."""
    logging.info("Preprocessing DataFrame for evaluation filtering...")
    df_processed = df.copy()
    # Apply cleaning consistent with triple generation
    df_processed['Name'] = df_processed['Name'].apply(create_node_label)
    # Clean columns used for filtering in evaluation
    for col in ['Healthy_Type', 'meal_type', 'Diet_Types', 'CleanedRegion', 'best_foodentityname']:
        # Apply safe_split_and_clean and join back, ensuring lowercase for consistency
        df_processed[col] = df_processed[col].apply(lambda x: ','.join(sorted(safe_split_and_clean(x))))
    df_processed['cook_time'] = df_processed['cook_time'].apply(create_node_label)
    return df_processed

def get_top_ingredients(df_processed: pd.DataFrame, n: int) -> List[str]:
    """Extracts the top N most common ingredients from the processed DataFrame."""
    logging.info(f"Determining top {n} ingredients...")
    # Assumes 'best_foodentityname' is comma-separated cleaned ingredients
    all_ingredients = df_processed['best_foodentityname'].str.split(',').explode()
    # Filter out empty strings that might result from splitting
    all_ingredients = all_ingredients[all_ingredients != '']
    top_ingredients = all_ingredients.value_counts().head(n).index.tolist()
    logging.info(f"Top {n} ingredients identified.")
    return top_ingredients

def define_relation_options(top_ingredients: List[str]) -> Dict[str, List[str]]:
    """Defines the possible values for each relation type used in scenarios."""
    return {
        # Use the dynamically determined relation names from HEALTH_RELATION_PREFIX_MAP
        **{relation: [f"{level}_{key.split('Has')[1].lower().replace('level','')}" for level in ['low', 'medium', 'high']]
           for key, relation in HEALTH_RELATION_PREFIX_MAP.items()},
        # Manually define options for other relations
        RELATION_MAPPING['meal_type']: ["breakfast", "lunch", "dinner", "snack", "dessert", "starter", "brunch", "drink"],
        RELATION_MAPPING['diet_types']: ["vegetarian", "vegan", "paleo", "standard"], # Make sure these match create_node_label output
        RELATION_MAPPING['region_countries']: ["global", "asia", "north_america", "europe", "middle_east", "latin_america_and_caribbean", "oceania", "africa"],
        RELATION_MAPPING['cook_time']: ["less_than_60_mins", "less_than_15_mins", "less_than_30_mins", "less_than_6_hours", "less_than_4_hours", "more_than_6_hours"],
        RELATION_MAPPING['ingredients']: top_ingredients
    }

def generate_scenario_combinations(relation_options: Dict[str, List[str]], combination_size: int) -> List[List[Tuple[str, str]]]:
    """Generates all possible combinations of criteria for a given size."""
    all_test_cases = []
    # Use items() to get (relation_name, list_of_values) pairs
    relations_with_values = list(relation_options.items())

    # Check if combination_size is valid
    if combination_size > len(relations_with_values) or combination_size < 1:
        logging.warning(f"Combination size {combination_size} is invalid for {len(relations_with_values)} relation types. Skipping.")
        return []

    # Get combinations of relation types (e.g., ('hasDietType', 'isForMealType'))
    for relation_comb in combinations(relations_with_values, combination_size):
        # relation_comb is like: (('hasDietType', ['veg', 'vegan']), ('isForMealType', ['lunch', 'dinner']))
        # Extract just the value lists: (['veg', 'vegan'], ['lunch', 'dinner'])
        value_lists = [relation[1] for relation in relation_comb]

        # Get product of values for the chosen relations (e.g., ('veg', 'lunch'), ('veg', 'dinner'), ...)
        for values_tuple in product(*value_lists):
            # values_tuple is like: ('veg', 'lunch')
            # Zip it with the corresponding relation names from relation_comb
            # relation_comb structure: ((relation_name1, values1), (relation_name2, values2))
            test_case = list(zip(values_tuple, [relation[0] for relation in relation_comb]))
            # test_case is like: [('veg', 'hasDietType'), ('lunch', 'isForMealType')]
            all_test_cases.append(test_case)

    return all_test_cases

def filter_recipes_for_scenario(df_processed: pd.DataFrame, scenario: List[Tuple[str, str]]) -> pd.DataFrame:
    """Filters the DataFrame based on the criteria in a scenario."""
    filtered_df = df_processed.copy()
    # Dynamically map relations back to dataframe columns for filtering
    column_mapping = {
        RELATION_MAPPING['diet_types']: 'Diet_Types',
        RELATION_MAPPING['meal_type']: 'meal_type',
        RELATION_MAPPING['region_countries']: 'CleanedRegion',
        RELATION_MAPPING['cook_time']: 'cook_time',
        RELATION_MAPPING['ingredients']: 'best_foodentityname',
        # Health relations all map to 'Healthy_Type' column
        **{relation: 'Healthy_Type' for relation in HEALTH_RELATION_PREFIX_MAP.values()}
    }

    for value, relation in scenario:
        if relation not in column_mapping:
            logging.warning(f"Relation '{relation}' not found in column mapping for filtering. Skipping criterion.")
            continue

        column_to_filter = column_mapping[relation]

        if column_to_filter not in filtered_df.columns:
             logging.warning(f"Column '{column_to_filter}' for relation '{relation}' not found in DataFrame. Skipping criterion.")
             continue

        # Apply filter: check if the cleaned 'value' string is present in the corresponding column
        # Ensure case-insensitivity and handle comma-separated values correctly
        # We check if the specific cleaned value is present as a whole word/item
        # surrounded by commas or at the start/end of the string.
        # Example: Searching for 'low_fat' in 'high_protein,low_fat,low_carb'
        pattern = r'(?:^|,)' + pd.io.common.escape(value) + r'(?:,|$)'
        try:
            filtered_df = filtered_df[filtered_df[column_to_filter].str.contains(pattern, case=False, na=False, regex=True)]
        except Exception as e:
            logging.error(f"Error filtering column '{column_to_filter}' with value '{value}' using regex: {e}")
            # Fallback to simpler contains check if regex fails (less accurate for substrings)
            # filtered_df = filtered_df[filtered_df[column_to_filter].str.contains(value, case=False, na=False)]
            return pd.DataFrame() # Return empty if filtering fails catastrophically


    return filtered_df

def calculate_average_ground_truth_size(scenarios: List[List[Tuple[str, str]]], df_processed: pd.DataFrame) -> int:
    """Calculates the average number of recipes matching the scenarios in the ground truth."""
    logging.info(f"Calculating average ground truth size for {len(scenarios)} scenarios...")
    total_count = 0
    valid_scenario_count = 0
    zero_result_scenarios = []

    for i, scenario in enumerate(scenarios):
        if (i + 1) % 1000 == 0: # Log progress
             logging.debug(f"Processing scenario {i+1}/{len(scenarios)} for average size calculation...")
        count = len(filter_recipes_for_scenario(df_processed, scenario))
        if count > 0:
            total_count += count
            valid_scenario_count += 1
        else:
            zero_result_scenarios.append(scenario)

    if zero_result_scenarios:
        logging.warning(f"{len(zero_result_scenarios)} scenarios resulted in 0 ground truth matches.")
        # Optionally log the first few zero-result scenarios for debugging
        # logging.debug("Example scenarios with 0 results:")
        # for z in zero_result_scenarios[:5]:
        #     logging.debug(z)

    average = total_count / valid_scenario_count if valid_scenario_count > 0 else 0
    logging.info(f"Average ground truth size (for non-zero scenarios): {average:.2f}")
    # Return integer average as K for precision needs an integer
    return max(1, int(round(average))) # Ensure K is at least 1

def predict_and_aggregate(model_result: PipelineResult, scenario: List[Tuple[str, str]]) -> pd.DataFrame:
    """Gets predictions from the KGE model for a scenario and aggregates scores."""
    all_predictions = []
    for tail_entity, relation in scenario:
        try:
            predicted_heads = predict_target(
                model=model_result.model,
                relation=relation,
                tail=tail_entity,
                triples_factory=model_result.training # Use the factory the model was trained on
            ).df
            # Check if 'head_label' and 'score' columns exist
            if 'head_label' not in predicted_heads.columns or 'score' not in predicted_heads.columns:
                 logging.warning(f"Prediction for ({relation}, {tail_entity}) did not return 'head_label' or 'score'. Skipping.")
                 continue
            all_predictions.append(predicted_heads[['head_label', 'score']])
        except Exception as e:
            logging.error(f"Error during prediction for ({relation}, {tail_entity}): {e}")
            continue # Skip this criterion if prediction fails

    if not all_predictions:
        return pd.DataFrame(columns=['head_label', 'total_score']) # Return empty if no predictions

    # Concatenate and aggregate scores
    all_predictions_df = pd.concat(all_predictions, ignore_index=True)
    aggregated_predictions = all_predictions_df.groupby('head_label').agg(
        total_score=('score', 'sum')
    ).reset_index()

    # Sort by aggregated score
    final_predictions_sorted = aggregated_predictions.sort_values(by="total_score", ascending=False)
    return final_predictions_sorted

def calculate_metrics(predicted_labels_top_k: Set[str],
                      predicted_labels_top_relevant: Set[str],
                      relevant_labels: Set[str],
                      k_for_precision: int,
                      num_relevant: int) -> Dict[str, float]:
    """
    Calculates Precision@K, Recall, F1 Score, and Accuracy@Relevant.

    Args:
        predicted_labels_top_k: Set of predicted recipe labels in the top K results (for Precision).
        predicted_labels_top_relevant: Set of predicted recipe labels in the top 'num_relevant' results (for Accuracy).
        relevant_labels: Set of actual relevant recipe labels (ground truth).
        k_for_precision: The value K used for Precision calculation (average ground truth size).
        num_relevant: The actual number of relevant recipes for this scenario.

    Returns:
        A dictionary containing 'Precision', 'Recall', 'F1', 'Accuracy'.
    """
    # Intersection for Precision calculation (using top K predictions)
    true_positives_at_k = predicted_labels_top_k.intersection(relevant_labels)
    # Intersection for Recall and Accuracy calculation (using top N=num_relevant predictions)
    true_positives_at_relevant = predicted_labels_top_relevant.intersection(relevant_labels)

    # Precision@K
    precision = len(true_positives_at_k) / k_for_precision if k_for_precision > 0 else 0.0

    # Recall (based on all relevant items)
    recall = len(true_positives_at_relevant) / num_relevant if num_relevant > 0 else 0.0 # Recall often uses intersection at K too, depends on definition. Let's use intersection at N=num_relevant here as it reflects if *all* relevant items could be found within the top N predictions. If recall should be @K, use true_positives_at_k.

    # F1 Score
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    # Accuracy@N (where N = num_relevant) - How many of the top N predictions were actually relevant?
    # This is essentially Precision@N where N=num_relevant
    accuracy = len(true_positives_at_relevant) / num_relevant if num_relevant > 0 else 0.0


    return {
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "Accuracy": accuracy,
        "TruePositives@K": len(true_positives_at_k),
        "TruePositives@Relevant": len(true_positives_at_relevant)
    }


def evaluate_scenarios(
    model_result: PipelineResult,
    recipes_df_processed: pd.DataFrame,
    relation_options: Dict[str, List[str]],
    combination_size: int,
    num_samples: int,
    results_dir: Path
):
    """Generates scenarios, runs predictions, calculates metrics, and saves results."""
    logging.info(f"--- Starting Evaluation for Combination Size: {combination_size} ---")

    output_filename = results_dir / f"{combination_size}_criteria_{KGE_MODEL}.csv"
    if output_filename.exists():
        logging.warning(f"Output file {output_filename} already exists. Skipping evaluation for size {combination_size}.")
        # return # Uncomment this line if you want to strictly avoid re-running

    # 1. Generate all possible combinations for the size
    all_combinations = generate_scenario_combinations(relation_options, combination_size)
    if not all_combinations:
        logging.warning(f"No combinations generated for size {combination_size}. Skipping evaluation.")
        return

    logging.info(f"Generated {len(all_combinations)} total combinations for size {combination_size}.")

    # 2. Sample a subset of scenarios if needed
    if len(all_combinations) > num_samples:
        logging.info(f"Sampling {num_samples} random scenarios from the total combinations.")
        random.seed(RANDOM_SEED) # Ensure reproducibility of sampling
        test_scenarios = random.sample(all_combinations, num_samples)
    else:
        logging.info("Using all generated combinations as test scenarios.")
        test_scenarios = all_combinations

    # 3. Calculate average ground truth size (K for Precision)
    # Use the *sampled* scenarios for calculating K if sampling was done
    k_for_precision = calculate_average_ground_truth_size(test_scenarios, recipes_df_processed)
    logging.info(f"Using K = {k_for_precision} for Precision calculation.")

    # 4. Iterate through scenarios, predict, and evaluate
    test_results = []
    evaluated_count = 0
    skipped_count = 0

    for idx, scenario in enumerate(test_scenarios):
        if (idx + 1) % (len(test_scenarios)//10 or 1) == 0: # Log progress roughly every 10%
            logging.info(f"Evaluating scenario {idx+1}/{len(test_scenarios)} (Size {combination_size})...")

        # Get ground truth relevant recipes
        relevant_recipes_df = filter_recipes_for_scenario(recipes_df_processed, scenario)
        relevant_recipe_labels = set(relevant_recipes_df['Name'])
        actual_num_relevant = len(relevant_recipe_labels)

        if actual_num_relevant == 0:
            # logging.debug(f"Skipping Scenario {idx+1} (Size {combination_size}) as ground truth is empty.")
            skipped_count += 1
            continue

        # Get model predictions
        predictions_df = predict_and_aggregate(model_result, scenario)
        if predictions_df.empty:
            logging.warning(f"No predictions generated for Scenario {idx+1} (Size {combination_size}). Skipping.")
            skipped_count +=1
            continue


        # Get top K predicted labels (for Precision@K)
        predicted_labels_top_k = set(predictions_df['head_label'].head(k_for_precision))

        # Get top N=num_relevant predicted labels (for Recall/Accuracy@N)
        predicted_labels_top_relevant = set(predictions_df['head_label'].head(actual_num_relevant))

        # Calculate metrics
        metrics = calculate_metrics(
            predicted_labels_top_k,
            predicted_labels_top_relevant,
            relevant_recipe_labels,
            k_for_precision,
            actual_num_relevant
        )

        # Store results
        result_dict = {
            "Scenario_Index": idx + 1,
            "Combination_Size": combination_size,
            "Criteria": str(scenario), # Store scenario as string for CSV
            "GroundTruth_Size (N)": actual_num_relevant,
            "K_for_Precision": k_for_precision,
            "Precision_at_K": metrics["Precision"],
            "Recall_at_N": metrics["Recall"],
            "F1_Score": metrics["F1 Score"],
            "Accuracy_at_N": metrics["Accuracy"],
            "Predicted_Count_TopK": len(predicted_labels_top_k),
            "Predicted_Count_TopN": len(predicted_labels_top_relevant),
            "TP_at_K": metrics["TruePositives@K"],
            "TP_at_N": metrics["TruePositives@Relevant"],
        }
        test_results.append(result_dict)
        evaluated_count += 1


    logging.info(f"Finished evaluation for size {combination_size}.")
    logging.info(f"Evaluated: {evaluated_count}, Skipped (0 ground truth or prediction errors): {skipped_count}")


    # 5. Save results to CSV
    if test_results:
        results_df = pd.DataFrame(test_results)
        results_df.to_csv(output_filename, index=False)
        logging.info(f"Results for size {combination_size} saved to {output_filename}")

        # Log average metrics for this size
        avg_precision = results_df["Precision_at_K"].mean()
        avg_recall = results_df["Recall_at_N"].mean()
        avg_f1 = results_df["F1_Score"].mean()
        avg_accuracy = results_df["Accuracy_at_N"].mean()
        logging.info(f"Average Metrics (Size {combination_size}): "
                     f"Precision@{k_for_precision}={avg_precision:.4f}, "
                     f"Recall@{k_for_precision}={avg_recall:.4f}, " # Note: Recall was calculated @N, rename log message if needed
                     f"F1={avg_f1:.4f}, "
                     f"Accuracy@N={avg_accuracy:.4f}")
    else:
        logging.warning(f"No valid results generated for size {combination_size}. No CSV file saved.")

    logging.info(f"--- Finished Evaluation for Combination Size: {combination_size} ---")

# --- Main Execution ---

def main():
    """Main function to orchestrate the workflow."""
    logging.info("Starting Recipe KGE Pipeline...")
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    # Ensure results directory exists
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # --- Data Loading and Graph Creation ---
    recipes_df_raw = load_and_preprocess_recipes(INPUT_CSV_PATH)
    _, triples_array = create_graph_and_triples(recipes_df_raw) # Graph G is not used later, can omit assignment
    save_triples(triples_array, TRIPLES_OUTPUT_PATH)

    # --- KGE Model Training ---
    # Load triples from the saved file
    try:
        triples_for_training = pd.read_csv(TRIPLES_OUTPUT_PATH).dropna().values
        triples_factory = TriplesFactory.from_labeled_triples(triples_for_training)
        logging.info("TriplesFactory created successfully.")
    except Exception as e:
        logging.error(f"Failed to load triples or create TriplesFactory: {e}")
        return # Stop execution if triples can't be loaded

    model_result = train_kge_model(triples_factory, KGE_MODEL, EPOCHS, RESULTS_DIR, EARLY_STOPPING_PATIENCE)

    # --- Evaluation Phase ---
    logging.info("--- Starting Scenario Evaluation Phase ---")
    # Preprocess the original DataFrame for filtering during evaluation
    recipes_df_processed = preprocess_dataframe_for_evaluation(recipes_df_raw)

    # Determine top ingredients and relation options
    top_ingredients = get_top_ingredients(recipes_df_processed, TOP_N_INGREDIENTS)
    relation_options = define_relation_options(top_ingredients)

    # Run evaluation for each combination size
    for i in range(1, MAX_CRITERIA_COMBINATIONS + 1):
        evaluate_scenarios(
            model_result=model_result,
            recipes_df_processed=recipes_df_processed,
            relation_options=relation_options,
            combination_size=i,
            num_samples=NUM_RANDOM_SAMPLES_PER_SIZE,
            results_dir=RESULTS_DIR
        )

    logging.info("Recipe KGE Pipeline finished successfully.")


if __name__ == "__main__":
    main()
