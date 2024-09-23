# Import necessary libraries
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from itertools import combinations, product
import random
import torch
import warnings
import logging
from tqdm.auto import tqdm
import os
import pickle

# Set the random seed for reproducibility
random.seed(42)
np.random.seed(42)

# Suppress warnings and error messages
warnings.filterwarnings('ignore')  # Suppress warnings
logging.getLogger("tensorflow").setLevel(logging.ERROR)  # Suppress TensorFlow messages

# Function to load and process data, embed text, generate scenarios, and evaluate
def run_recipe_evaluation(file_path):
    # Load the data
    recipes_df = pd.read_csv(file_path)
    recipes_df = recipes_df.drop_duplicates(subset='Name', keep='first').reset_index(drop=True)

    # Optionally reduce dataset size for testing
    # Uncomment the following line to use a smaller subset
    # recipes_df = recipes_df.sample(n=5000, random_state=42).reset_index(drop=True)

    # Helper function to create valid node labels
    def create_node_label(label):
        if isinstance(label, str):
            return label.replace(" ", "_").replace("-", "_").replace(">", "").replace("<", "less_than_").strip().lower()
        return str(label)

    # Process dataframe and clean labels
    def process_recipes_dataframe(df):
        df['Name'] = df['Name'].apply(create_node_label)
        df['best_foodentityname'] = df['best_foodentityname'].apply(lambda x: ','.join([create_node_label(ing.strip()) for ing in x.split(',')]))
        df['Healthy_Type'] = df['Healthy_Type'].apply(lambda x: ','.join([create_node_label(ht.strip()) for ht in x.split(',')]) if pd.notna(x) else '')
        df['meal_type'] = df['meal_type'].apply(lambda x: ','.join([create_node_label(mt.strip()) for mt in x.split(',')]) if pd.notna(x) else '')
        df['cook_time'] = df['cook_time'].apply(create_node_label)
        df['Diet_Types'] = df['Diet_Types'].apply(lambda x: ','.join([create_node_label(dt.strip()) for dt in x.split(',')]) if pd.notna(x) else 'UNKNOWN_PLACEHOLDER')
        df['CleanedRegion'] = df['CleanedRegion'].apply(lambda x: ','.join([create_node_label(rc.strip()) for rc in x.split(',')]) if pd.notna(x) else '')
        return df

    # Process dataframe
    recipes_df = process_recipes_dataframe(recipes_df)

    # Check if GPU is available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load Sentence-BERT model and move it to the appropriate device
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2', device=device)

    # Helper function to embed text using GPU/CPU based on availability
    def embed_text(texts, show_progress_bar=False, batch_size=64):
        return model.encode(
            texts,
            show_progress_bar=show_progress_bar,
            device=device,
            #batch_size=batch_size,
            convert_to_numpy=True  # Ensure embeddings are NumPy arrays
        )

    # Extract and embed relevant textual features
    recipes_df['text_for_embedding'] = recipes_df.apply(
        lambda row: f"{row['Name']} {row['best_foodentityname']} {row['Healthy_Type']} {row['meal_type']} {row['Diet_Types']} {row['CleanedRegion']} {row['cook_time']}",
        axis=1
    )

    # Check if embeddings are already cached
    embeddings_cache_path = 'recipe_embeddings.pkl'
    if os.path.exists(embeddings_cache_path):
        print("Loading cached recipe embeddings...")
        with open(embeddings_cache_path, 'rb') as f:
            recipe_embedding_dict = pickle.load(f)
    else:
        print("Embedding recipe texts...")
        recipe_embeddings = embed_text(recipes_df['text_for_embedding'].tolist(), show_progress_bar=True)
        # Create a dictionary to store recipe embeddings
        recipe_embedding_dict = dict(zip(recipes_df['Name'], recipe_embeddings))
        # Cache the embeddings
        with open(embeddings_cache_path, 'wb') as f:
            pickle.dump(recipe_embedding_dict, f)

    # Define relation options
    ingredient_counts = recipes_df['best_foodentityname'].apply(lambda x: [ingredient.strip() for ingredient in x.split(',')]).explode().value_counts()
    top_ingredients = ingredient_counts.head(20).index.tolist()
    relation_options = {
        "HasProteinLevel": ["low_protein", "medium_protein", "high_protein"],
        "HasCarbLevel": ["low_carb", "medium_carb", "high_carb"],
        "HasFatLevel": ["low_fat", "medium_fat", "high_fat"],
        "HasFiberLevel": ["low_fiber", "medium_fiber", "high_fiber"],
        "HasSodiumLevel": ["low_sodium", "medium_sodium", "high_sodium"],
        "HasSugarLevel": ["low_sugar", "medium_sugar", "high_sugar"],
        "HasCholesterolLevel": ["low_cholesterol", "medium_cholesterol", "high_cholesterol"],
        "HasCalorieLevel": ["low_calorie", "medium_calorie", "high_calorie"],
        "isForMealType": ["breakfast", "lunch", "dinner", "snack", "dessert", "starter", "brunch", "drink"],
        "hasDietType": ["vegetarian", "vegan", "paleo", "standard"],
        "isFromRegion": ["global", "asia", "north_america", "europe", "middle_east", "latin_america_and_caribbean", "oceania", "africa"],
        "needTimeToCook": ["less_than_60_mins", "less_than_15_mins", "less_than_30_mins", "less_than_6_hours", "less_than_4_hours", "more_than_6_hours"],
        "contains": top_ingredients
    }

    # Function to generate test cases by sampling over relation values
    def generate_specific_combinations(relation_options, combination_size, sample_size=None):
        relation_keys = list(relation_options.keys())
        all_relation_combinations = list(combinations(relation_keys, combination_size))
        all_test_cases = []
        
        for comb in all_relation_combinations:
            value_lists = [relation_options[relation] for relation in comb]
            all_value_combinations = list(product(*value_lists))
            for values in all_value_combinations:
                test_case = list(zip(values, comb))
                all_test_cases.append(test_case)
        
        # If total test cases are fewer than sample_size, use all; otherwise, sample
        if sample_size is None or sample_size >= len(all_test_cases):
            sampled_test_cases = all_test_cases
        else:
            sampled_test_cases = random.sample(all_test_cases, sample_size)
        
        return sampled_test_cases

    # Function to calculate average occurrence
    def calculate_average_occurrence(scenarios, df):
        total_count = 0  # Total number of recipes for all scenarios
        valid_scenario_count = 0  # Number of valid scenarios
        zero_scenarios = []  # List to hold scenarios that give zero results

        # Iterate through each scenario
        for scenario in scenarios:
            filter_condition = pd.Series([True] * len(df))  # Start with all rows included

            # Apply filters for each criterion
            for value, relation in scenario:
                if relation == "HasProteinLevel":
                    filter_condition &= df['Healthy_Type'].str.contains(value, case=False, na=False)
                elif relation == "HasCarbLevel":
                    filter_condition &= df['Healthy_Type'].str.contains(value, case=False, na=False)
                elif relation == "HasFatLevel":
                    filter_condition &= df['Healthy_Type'].str.contains(value, case=False, na=False)
                elif relation == "HasFiberLevel":
                    filter_condition &= df['Healthy_Type'].str.contains(value, case=False, na=False)
                elif relation == "HasSodiumLevel":
                    filter_condition &= df['Healthy_Type'].str.contains(value, case=False, na=False)
                elif relation == "HasSugarLevel":
                    filter_condition &= df['Healthy_Type'].str.contains(value, case=False, na=False)
                elif relation == "HasCholesterolLevel":
                    filter_condition &= df['Healthy_Type'].str.contains(value, case=False, na=False)
                elif relation == "HasCalorieLevel":
                    filter_condition &= df['Healthy_Type'].str.contains(value, case=False, na=False)
                elif relation == "isForMealType":
                    filter_condition &= df['meal_type'].str.contains(value, case=False, na=False)
                elif relation == "hasDietType":
                    filter_condition &= df['Diet_Types'].str.contains(value, case=False, na=False)
                elif relation == "isFromRegion":
                    filter_condition &= df['CleanedRegion'].str.contains(value, case=False, na=False)
                elif relation == "needTimeToCook":
                    filter_condition &= df['cook_time'].str.contains(value, case=False, na=False)
                elif relation == "contains":
                    filter_condition &= df['best_foodentityname'].str.contains(value, case=False, na=False)

            # Find the number of remaining rows after all filters
            count = df[filter_condition].shape[0]

            # If there is a valid count, increase the total and valid scenario count
            if count > 0:
                total_count += count
                valid_scenario_count += 1
            else:
                zero_scenarios.append(scenario)  # If result is zero, save the scenario

        # Calculate average
        if valid_scenario_count > 0:
            average = total_count / valid_scenario_count
        else:
            average = 0  # If no valid criteria, average is zero

        # Print scenarios with zero results
        if zero_scenarios:
            print("\nScenarios with 0 results:")
            for z in zero_scenarios:
                print(z)
        else:
            print("\nNo scenarios returned 0 results.")

        return average

    # Updated function to calculate similarity and evaluate
    def calculate_similarity_with_text_embedding(test_scenarios, average_occurrence, recipe_embedding_dict, recipes_df, total_scenarios):
        test_results = []
        recipe_names = np.array(list(recipe_embedding_dict.keys()))
        recipe_embeddings = np.array(list(recipe_embedding_dict.values()))

        with tqdm(total=total_scenarios, desc="Processing Scenarios") as pbar:
            for idx, scenario in enumerate(test_scenarios):
                scenario_text = " ".join([f"{relation}: {value}" for value, relation in scenario])
                scenario_embedding = embed_text([scenario_text])  # Returns array of shape (1, embedding_dim)

                # Compute cosine similarities
                similarities = cosine_similarity(scenario_embedding, recipe_embeddings)[0]  # similarities shape: (num_recipes,)

                # Filter recipes based on criteria
                filtered_recipes = recipes_df
                for value, relation in scenario:
                    if relation == "HasProteinLevel":
                        filtered_recipes = filtered_recipes[filtered_recipes['Healthy_Type'].str.contains(value, case=False, na=False)]
                    elif relation == "HasCarbLevel":
                        filtered_recipes = filtered_recipes[filtered_recipes['Healthy_Type'].str.contains(value, case=False, na=False)]
                    elif relation == "HasFatLevel":
                        filtered_recipes = filtered_recipes[filtered_recipes['Healthy_Type'].str.contains(value, case=False, na=False)]
                    elif relation == "HasFiberLevel":
                        filtered_recipes = filtered_recipes[filtered_recipes['Healthy_Type'].str.contains(value, case=False, na=False)]
                    elif relation == "HasSodiumLevel":
                        filtered_recipes = filtered_recipes[filtered_recipes['Healthy_Type'].str.contains(value, case=False, na=False)]
                    elif relation == "HasSugarLevel":
                        filtered_recipes = filtered_recipes[filtered_recipes['Healthy_Type'].str.contains(value, case=False, na=False)]
                    elif relation == "HasCholesterolLevel":
                        filtered_recipes = filtered_recipes[filtered_recipes['Healthy_Type'].str.contains(value, case=False, na=False)]
                    elif relation == "HasCalorieLevel":
                        filtered_recipes = filtered_recipes[filtered_recipes['Healthy_Type'].str.contains(value, case=False, na=False)]
                    elif relation == "isForMealType":
                        filtered_recipes = filtered_recipes[filtered_recipes['meal_type'].str.contains(value, case=False, na=False)]
                    elif relation == "hasDietType":
                        filtered_recipes = filtered_recipes[filtered_recipes['Diet_Types'].str.contains(value, case=False, na=False)]
                    elif relation == "isFromRegion":
                        filtered_recipes = filtered_recipes[filtered_recipes['CleanedRegion'].str.contains(value, case=False, na=False)]
                    elif relation == "needTimeToCook":
                        filtered_recipes = filtered_recipes[filtered_recipes['cook_time'].str.contains(value, case=False, na=False)]
                    elif relation == "contains":
                        filtered_recipes = filtered_recipes[filtered_recipes['best_foodentityname'].str.contains(value, case=False, na=False)]

                expected_criteria_match_number = len(filtered_recipes)
                if expected_criteria_match_number == 0:
                    pbar.update(1)
                    continue
                relevant_recipes = set(filtered_recipes['Name'])

                # Now we can proceed to get top N indices
                sorted_indices = np.argsort(-similarities)
                mean_count = max(int(average_occurrence), 1)

                # For Precision and Recall
                top_n_indices = sorted_indices[:mean_count]
                top_n_recipes = recipe_names[top_n_indices]
                predicted_recipes = set(top_n_recipes)

                # For Accuracy
                top_n_indices_accuracy = sorted_indices[:expected_criteria_match_number]
                top_n_recipes_accuracy = recipe_names[top_n_indices_accuracy]
                predicted_recipes_accuracy = set(top_n_recipes_accuracy)

                # Calculate true positives for Precision and Recall
                true_positives = predicted_recipes.intersection(relevant_recipes)
                precision = len(true_positives) / len(predicted_recipes) if len(predicted_recipes) > 0 else 0
                recall = len(true_positives) / len(relevant_recipes) if len(relevant_recipes) > 0 else 0
                f1_score = (2 * precision 
                * recall) / (precision + recall) if precision + recall > 0 else 0

                # Calculate matching recipes for Accuracy
                matching_recipes_accuracy = relevant_recipes.intersection(predicted_recipes_accuracy)
                accuracy = len(matching_recipes_accuracy) / expected_criteria_match_number if expected_criteria_match_number > 0 else 0

                # Store results
                result_dict = {
                    "Scenario": idx + 1,
                    "Criteria": scenario,
                    "Expected Matches For Recall": expected_criteria_match_number,
                    "Predicted Matches (Recall)": len(true_positives),
                    "Recall": recall,
                    "Expected Matches For Precision": mean_count,
                    "Predicted Matches (Precision)": len(true_positives),
                    "Precision": precision,
                    "Expected Matched For Accuracy": expected_criteria_match_number,
                    "Predicted Matched (Accuracy)": len(matching_recipes_accuracy),
                    "Accuracy": accuracy,
                    "F1 Score": f1_score
                }
                test_results.append(result_dict)
                pbar.update(1)  # Update progress bar
        return pd.DataFrame(test_results)

    # Run the evaluation for all combination sizes
    for combination_size, samples, output_file_suffix in [
        (1, None, 'one_criteria_text_embedding.csv'),
        (2, None, 'two_criteria_text_embedding.csv'),
        (3, 25000, 'three_criteria_text_embedding.csv'),
        (4, 50000, 'four_criteria_text_embedding.csv'),
        (5, 50000, 'five_criteria_text_embedding.csv'),
        (6, 50000, 'six_criteria_text_embedding.csv'),
        (7, 50000, 'seven_criteria_text_embedding.csv'),
        (8, 50000, 'eight_criteria_text_embedding.csv')
    ]:

        print(f"\nProcessing combination size {combination_size} with sample size {samples}")

        # Generate test scenarios
        test_scenarios = generate_specific_combinations(
            relation_options, combination_size, sample_size=samples
        )

        # Calculate the total number of scenarios for the progress bar
        total_scenarios = len(test_scenarios)
        print(f"Total scenarios to process: {total_scenarios}")

        # Calculate average occurrence
        average_occurrence = calculate_average_occurrence(test_scenarios, recipes_df)
        average_occurrence = int(average_occurrence)
        print(f"\nAverage Occurrence for Scenarios: {average_occurrence}")

        # Run the evaluation
        text_embedding_results_df = calculate_similarity_with_text_embedding(
            test_scenarios, average_occurrence, recipe_embedding_dict, recipes_df, total_scenarios
        )

        # Ensure the output directory exists
        output_dir = '/app/TextEmbeddingEvulationResults'
        os.makedirs(output_dir, exist_ok=True)

        # Save results to CSV in the specific folder path
        output_file = os.path.join(output_dir, output_file_suffix)
        text_embedding_results_df.to_csv(output_file, index=False)
        print(f"Results saved for combination size {combination_size} to {output_file}")

# Run the evaluation
run_recipe_evaluation('/app/BalancedRecipe_entity_linking.csv')
