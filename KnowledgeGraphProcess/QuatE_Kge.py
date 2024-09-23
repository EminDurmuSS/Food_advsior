#import library 
import networkx as nx
import numpy as np
from pykeen.pipeline import pipeline
from pykeen.triples import TriplesFactory
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from pykeen.hpo import hpo_pipeline


# Load the CSV file
file_path = '/app/BalancedRecipe_entity_linking.csv'
recipes_df = pd.read_csv(file_path)
recipes_df = recipes_df.drop_duplicates(subset='Name', keep='first')  # Ensure unique recipes
recipes_df = recipes_df.reset_index(drop=True)  #

UNKNOWN_PLACEHOLDER = 'Unknown'

# Helper function to create valid node labels
def create_node_label(label):
    if isinstance(label, str):
        return label.replace(" ", "_").replace("-", "_").replace(">", "").replace("<", "less_than_").strip().lower()
    return str(label)

# Convert the dataframe into the required format for the graph
def process_recipes(df):
    recipes = {}
    for _, row in df.iterrows():
        recipe_name = create_node_label(row['Name'])
        best_foodentityname = [create_node_label(ing.strip()) for ing in row['best_foodentityname'].split(',')]
        healthy_types = [create_node_label(ht.strip()) for ht in row['Healthy_Type'].split(',')] if pd.notna(row['Healthy_Type']) else []
        meal_types = [create_node_label(mt.strip()) for mt in row['meal_type'].split(',')] if pd.notna(row['meal_type']) else []
        cook_time = create_node_label(row['cook_time'])
        diet_types = [create_node_label(dt.strip()) for dt in row.get('Diet_Types', '').split(',')] if pd.notna(row['Diet_Types']) else [UNKNOWN_PLACEHOLDER]
        region_countries = [create_node_label(rc.strip()) for rc in row['CleanedRegion'].split(',')] if pd.notna(row['CleanedRegion']) else []

        recipes[recipe_name] = {
            "ingredients": best_foodentityname,
            "diet_types": diet_types,
            "meal_type": meal_types,
            "cook_time": cook_time,
            "region_countries": region_countries,
            "healthy_types": healthy_types,
        }
    return recipes

recipes = process_recipes(recipes_df)

# Create graph and triples based on correct relationships
def create_graph_and_triples(recipes):
    G = nx.Graph()
    triples = []

    for recipe, details in recipes.items():
        G.add_node(recipe, type='recipe')
        for relation_type, elements in details.items():
            if isinstance(elements, list):  # Handle lists
                for element in elements:
                    if element != UNKNOWN_PLACEHOLDER:
                        if relation_type == 'healthy_types':
                            # Generalize the relation to HasProteinLevel, HasCarbLevel, etc.
                            if 'protein' in element:
                                relation = 'HasProteinLevel'
                            elif 'carb' in element:
                                relation = 'HasCarbLevel'
                            elif 'fat' in element and 'Saturated' not in element:
                                relation = 'HasFatLevel'
                            elif 'Saturated Fat' in element:
                                relation = 'HasSaturatedFatLevel'
                            elif 'calorie' in element:
                                relation = 'HasCalorieLevel'
                            elif 'sodium' in element:
                                relation = 'HasSodiumLevel'
                            elif 'sugar' in element:
                                relation = 'HasSugarLevel'
                            elif 'fiber' in element:
                                relation = 'HasFiberLevel'
                            elif 'cholesterol' in element:
                                relation = 'HasCholesterolLevel'
                            else:
                                relation = 'HasHealthAttribute'  # For other health attributes
                            G.add_node(element, type=relation)
                        else:## Has camel Case 
                            G.add_node(element, type=relation_type)
                            relation = {
                                'ingredients': 'contains',
                                'diet_types': 'hasDietType',
                                'meal_type': 'isForMealType',
                                'cook_time': 'needTimeToCook',
                                'region_countries': 'isFromRegion',
                            }.get(relation_type, 'hasAttribute')
                        G.add_edge(recipe, element, relation=relation)
                        triples.append((recipe, relation, element))
            else:  # Handle single elements like cook_time
                if elements != UNKNOWN_PLACEHOLDER:
                    G.add_node(elements, type=relation_type)
                    relation = {
                        'cook_time': 'needTimeToCook',
                        'ingredients': 'contains',
                        'diet_types': 'hasDietType',
                        'meal_type': 'isForMealType',
                        'cook_time': 'needTimeToCook',
                        'region_countries': 'isFromRegion',
                    }.get(relation_type, 'hasAttribute')
                    G.add_edge(recipe, elements, relation=relation)
                    triples.append((recipe, relation, elements))

    return G, np.array(triples, dtype=str)

# Process the recipes to create the graph and triples
G, triples_array = create_graph_and_triples(recipes)


# Convert the triples array into a pandas DataFrame
triples_df = pd.DataFrame(triples_array, columns=['Head', 'Relation', 'Tail'])


triples_df.to_csv('/app/recipes_triples_10000sample.csv',index=False)



import pandas as pd 
df=pd.read_csv('/app/recipes_triples_10000sample.csv')
df=df.dropna()
#df_filtered = df[df['Relation'] != 'contains']

# Assuming df is your DataFrame
triples = df[['Head', 'Relation', 'Tail']].values  # This returns a NumPy array directly

# Create a TriplesFactory
triples_factory = TriplesFactory.from_labeled_triples(triples)

#training, testing, validation = triples_factory.split([7., .2, .1],randomize_cleanup=True)


# Use the same triples_factory for both training and testing
result = pipeline(
    model='RotatE',
    training=triples_factory,  # Use the entire triples factory for training
    testing=triples_factory,   # Use the same factory for testing
    validation=triples_factory,
    epochs=150,
    stopper='early',
)

print(f"Hits@1: {result.metric_results.get_metric('hits_at_1')}")
print(f"Hits@3: {result.metric_results.get_metric('hits_at_3')}")
print(f"Hits@10: {result.metric_results.get_metric('hits_at_10')}")
print(f"Mean Rank (Realistic): {result.metric_results.get_metric('both.realistic.mean_rank')}")
print(f"Mean Reciprocal Rank (Realistic): {result.metric_results.get_metric('both.realistic.mean_reciprocal_rank')}")




import pandas as pd
from pykeen.predict import predict_target
from itertools import product
#nohup python3 testPytorchRotatE.py > output.log &
# Load the data
triples_df = pd.read_csv('/app/recipes_triples_10000sample.csv')
file_path = '/app/BalancedRecipe_entity_linking.csv'
recipes_df = pd.read_csv(file_path)
recipes_df = recipes_df.drop_duplicates(subset='Name', keep='first')  # Ensure unique recipes
recipes_df = recipes_df.reset_index(drop=True)

# Process the dataframe and clean labels
def process_recipes_dataframe(df):
    df['Name'] = df['Name'].apply(create_node_label)
    df['RecipeIngredientParts'] = df['RecipeIngredientParts'].apply(lambda x: ','.join([create_node_label(ing.strip()) for ing in x.split(',')]))
    df['Healthy_Type'] = df['Healthy_Type'].apply(lambda x: ','.join([create_node_label(ht.strip()) for ht in x.split(',')]) if pd.notna(x) else '')
    df['meal_type'] = df['meal_type'].apply(lambda x: ','.join([create_node_label(mt.strip()) for mt in x.split(',')]) if pd.notna(x) else '')
    df['cook_time'] = df['cook_time'].apply(create_node_label)
    df['Diet_Types'] = df['Diet_Types'].apply(lambda x: ','.join([create_node_label(dt.strip()) for dt in x.split(',')]) if pd.notna(x) else 'UNKNOWN_PLACEHOLDER')
    df['CleanedRegion'] = df['CleanedRegion'].apply(lambda x: ','.join([create_node_label(rc.strip()) for rc in x.split(',')]) if pd.notna(x) else '')
    df['best_foodentityname'] = df['best_foodentityname'].apply(lambda x: ','.join([create_node_label(entity.strip()) for entity in x.split(',')]) if pd.notna(x) else '')
    return df

# Process the dataframe
recipes_df = process_recipes_dataframe(recipes_df)

# Extract the top 50 most common ingredients from 'best_foodentityname'
ingredient_counts = recipes_df['best_foodentityname'].apply(lambda x: [ingredient.strip() for ingredient in x.split(',')]).explode().value_counts()
top_ingredients = ingredient_counts.head(25).index.tolist()


from itertools import combinations, product
import random

# Set the random seed for reproducibility
random.seed(42)

# Extract the top 20 most common ingredients from 'best_foodentityname'
ingredient_counts = recipes_df['best_foodentityname'].apply(lambda x: [ingredient.strip() for ingredient in x.split(',')]).explode().value_counts()
top_ingredients = ingredient_counts.head(20).index.tolist()

# Define relation options based on the columns in your dataset
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

# Function to generate test cases for specific combination size (binary or triple)
def generate_specific_combinations(relation_options, combination_size):
    all_test_cases = []
    relations = list(relation_options.items())
    
    for comb in combinations(relations, combination_size):
        for values_comb in product(*[relation[1] for relation in comb]):
            test_case = [(value, relation[0]) for value, relation in zip(values_comb, comb)]
            all_test_cases.append(test_case)
    
    return all_test_cases

# Generate binary combinations (pairs of criteria)
three_criteria = generate_specific_combinations(relation_options, combination_size=4)

# List of 5000 random samples from the generated three_criteria list
random_4_user_criteria = random.choices(three_criteria, k=50000)

print(random_4_user_criteria)

# Display the first 20 combinations
random_4_user_criteria[0:20]


test_scenarios = random_4_user_criteria
def calculate_average_occurrence(scenarios, df):
    total_count = 0  # Tüm senaryolar için toplam tarif sayısı
    valid_scenario_count = 0  # Geçerli senaryoların sayısı
    zero_scenarios = []  # Sıfır sonuç veren senaryoları tutacak liste

    # Her bir senaryo için döngü
    for scenario in scenarios:
        filter_condition = pd.Series([True] * len(df))  # Başlangıçta tüm satırlar dahil
        
        # Her bir kriter için filtre uygulama
        for value, relation in scenario:
            
            if relation == "HasProteinLevel":
                filter_condition &= df['Healthy_Type'].apply(lambda x: isinstance(x, str) and value in x.lower())
            elif relation == "HasCarbLevel":
                filter_condition &= df['Healthy_Type'].apply(lambda x: isinstance(x, str) and value in x.lower())
            elif relation == "HasFatLevel":
                filter_condition &= df['Healthy_Type'].apply(lambda x: isinstance(x, str) and value in x.lower())
            elif relation == "HasFiberLevel":
                filter_condition &= df['Healthy_Type'].apply(lambda x: isinstance(x, str) and value in x.lower())
            elif relation == "HasSodiumLevel":
                filter_condition &= df['Healthy_Type'].apply(lambda x: isinstance(x, str) and value in x.lower())
            elif relation == "HasSugarLevel":
                filter_condition &= df['Healthy_Type'].apply(lambda x: isinstance(x, str) and value in x.lower())
            elif relation == "HasCholesterolLevel":
                filter_condition &= df['Healthy_Type'].apply(lambda x: isinstance(x, str) and value in x.lower())
            elif relation == "HasCalorieLevel":
                filter_condition &= df['Healthy_Type'].apply(lambda x: isinstance(x, str) and value in x.lower())
            elif relation == "isForMealType":
                filter_condition &= df['meal_type'].apply(lambda x: isinstance(x, str) and value in x.lower())
            elif relation == "hasDietType":
                filter_condition &= df['Diet_Types'].apply(lambda x: isinstance(x, str) and value in x.lower())
            elif relation == "isFromRegion":
                filter_condition &= df['CleanedRegion'].apply(lambda x: isinstance(x, str) and value in x.lower())
            elif relation == "needTimeToCook":
                filter_condition &= df['cook_time'].apply(lambda x: isinstance(x, str) and value in x.lower())
            elif relation == "contains":
                filter_condition &= df['best_foodentityname'].apply(lambda x: isinstance(x, str) and value in x.lower())
            

        # Tüm filtreler uygulandıktan sonra, kalan satırların sayısını bul
        count = df[filter_condition].shape[0]

        # Eğer geçerli bir sayım varsa toplamı ve geçerli senaryo sayısını artırıyoruz
        if count > 0:
            total_count += count
            valid_scenario_count += 1
        else:
            zero_scenarios.append(scenario)  # Eğer sonuç sıfır ise senaryoyu kaydet

    # Ortalama hesaplama
    if valid_scenario_count > 0:
        average = total_count / valid_scenario_count
    else:
        average = 0  # Eğer geçerli kriter yoksa ortalama sıfırdır

    # Sıfır sonuç veren senaryoları yazdırma
    if zero_scenarios:
        print("\nScenarios with 0 results:")
        for z in zero_scenarios:
            print(z)
    else:
        print("\nNo scenarios returned 0 results.")

    return average


# Sonucu hesapla
average_occurrence = calculate_average_occurrence(test_scenarios, recipes_df)
average_occurrence = int(average_occurrence)
# Sonucu yazdırma
print("\nAverage Occurrence for Scenarios:", average_occurrence)

# Initialize a list to store test results
test_results = []

# Iterate through each test scenario
for idx, scenario in enumerate(test_scenarios):
    
    filtered_recipes = recipes_df

    for tail_entity, relation in scenario:
        if relation == "hasDietType":
            filtered_recipes = filtered_recipes[filtered_recipes['Diet_Types'].str.contains(tail_entity, case=False, na=False)]
        elif relation == "isForMealType":
            filtered_recipes = filtered_recipes[filtered_recipes['meal_type'].str.contains(tail_entity, case=False, na=False)]
        elif relation == "isFromRegion":
            filtered_recipes = filtered_recipes[filtered_recipes['CleanedRegion'].str.contains(tail_entity, case=False, na=False)]
        elif relation == "needTimeToCook":
            filtered_recipes = filtered_recipes[filtered_recipes['cook_time'].str.contains(tail_entity, case=False, na=False)]
        elif relation in ["HasCarbLevel", "HasProteinLevel", "HasFatLevel", "HasFiberLevel", "HasSodiumLevel", "HasSugarLevel", "HasCholesterolLevel", "HasCalorieLevel"]:
            filtered_recipes = filtered_recipes[filtered_recipes['Healthy_Type'].str.contains(tail_entity, case=False, na=False)]

    expected_criteria_match_number = len(filtered_recipes)
    # Skip the scenario if there are no matches
    if expected_criteria_match_number == 0:
        #print(f"Skipping Scenario {idx+1} as no relevant matches were found.")
        continue
    mean_count = average_occurrence

    # Perform prediction using the model
    all_predictions = []
    
    for tail_entity, relation in scenario:
        predicted_heads = predict_target(
            model=result.model,
            relation=relation,
            tail=tail_entity,
            triples_factory=result.training
        ).df
        all_predictions.append(predicted_heads)

    all_predictions_df = pd.concat(all_predictions, ignore_index=True)
    aggregated_predictions = all_predictions_df.groupby('head_label').agg(
        total_score=('score', 'sum')
    ).reset_index()

    final_predictions_sorted = aggregated_predictions.sort_values(by="total_score", ascending=False)

    # Calculate Precision and Recall using the same Top N
    top_n_scores = final_predictions_sorted["head_label"].head(mean_count)
    matching_recipes = filtered_recipes[filtered_recipes['Name'].isin(top_n_scores)].drop_duplicates(subset='Name')

    # Calculate sets of predicted and relevant recipes
    predicted_recipes = set(top_n_scores)
    relevant_recipes = set(filtered_recipes['Name'])

    # Calculate Precision and Recall using the same Top N
    top_n_scores_accuracy = final_predictions_sorted["head_label"].head(expected_criteria_match_number)
    matching_recipes_accuracy = filtered_recipes[filtered_recipes['Name'].isin(top_n_scores_accuracy)]

    true_positives = predicted_recipes.intersection(relevant_recipes)
    precision = len(true_positives) / len(predicted_recipes) if len(predicted_recipes) > 0 else 0
    # Recall Calculation
    Recall = len(true_positives) / len(relevant_recipes) if len(relevant_recipes) > 0 else 0

    Accuracy = len(matching_recipes_accuracy) / expected_criteria_match_number if expected_criteria_match_number > 0 else 0


    # Precision Calculation
    predicted_recipes = set(top_n_scores)
    relevant_recipes = set(filtered_recipes['Name'])


    # F1 Score Calculation
    if precision + Recall > 0:
        F1 = 2 * (precision * Recall) / (precision + Recall)
    else:
        F1 = 0

    # Store results
    result_dict = {
        "Scenario": idx + 1,
        "Criteria": scenario,
        "Expected Matches For Recall": expected_criteria_match_number,
        "Predicted Matches (Recall)": len(matching_recipes),
        "Recall": Recall,
        "Expected Matches For Precision": mean_count,
        "Predicted Matches (Precision)": len(matching_recipes),
        "Precision": precision,
        "F1 Score": F1,  # Add F1 Score to results,
        "Expected Matched For Accuracy":expected_criteria_match_number,
        "Predicted Matched (Accuracy)":len(matching_recipes_accuracy),
        "Accuracy":Accuracy
    }
    
    test_results.append(result_dict)

# Convert results into DataFrame
results_df = pd.DataFrame(test_results)


# Display the results
results_df.to_csv('/app/RotatEEvulationResults/four_criteria_rotate.csv',index=False)

from itertools import combinations, product
import random

# Set the random seed for reproducibility
random.seed(42)

# Extract the top 20 most common ingredients from 'best_foodentityname'
ingredient_counts = recipes_df['best_foodentityname'].apply(lambda x: [ingredient.strip() for ingredient in x.split(',')]).explode().value_counts()
top_ingredients = ingredient_counts.head(20).index.tolist()

# Define relation options based on the columns in your dataset
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

# Generate binary combinations (pairs of criteria)
three_criteria = generate_specific_combinations(relation_options, combination_size=5)

# List of 5000 random samples from the generated three_criteria list
random_5_user_criteria = random.choices(three_criteria, k=50000)

print(random_5_user_criteria)

# Display the first 20 combinations
random_5_user_criteria[0:20]

test_scenarios = random_5_user_criteria
# Sonucu hesapla
average_occurrence = calculate_average_occurrence(test_scenarios, recipes_df)
average_occurrence = int(average_occurrence)
# Sonucu yazdırma
print("\nAverage Occurrence for Scenarios:", average_occurrence)

# Initialize a list to store test results
test_results = []

# Iterate through each test scenario
for idx, scenario in enumerate(test_scenarios):
    
    filtered_recipes = recipes_df

    for tail_entity, relation in scenario:
        if relation == "hasDietType":
            filtered_recipes = filtered_recipes[filtered_recipes['Diet_Types'].str.contains(tail_entity, case=False, na=False)]
        elif relation == "isForMealType":
            filtered_recipes = filtered_recipes[filtered_recipes['meal_type'].str.contains(tail_entity, case=False, na=False)]
        elif relation == "isFromRegion":
            filtered_recipes = filtered_recipes[filtered_recipes['CleanedRegion'].str.contains(tail_entity, case=False, na=False)]
        elif relation == "needTimeToCook":
            filtered_recipes = filtered_recipes[filtered_recipes['cook_time'].str.contains(tail_entity, case=False, na=False)]
        elif relation in ["HasCarbLevel", "HasProteinLevel", "HasFatLevel", "HasFiberLevel", "HasSodiumLevel", "HasSugarLevel", "HasCholesterolLevel", "HasCalorieLevel"]:
            filtered_recipes = filtered_recipes[filtered_recipes['Healthy_Type'].str.contains(tail_entity, case=False, na=False)]

    expected_criteria_match_number = len(filtered_recipes)
    # Skip the scenario if there are no matches
    if expected_criteria_match_number == 0:
        #print(f"Skipping Scenario {idx+1} as no relevant matches were found.")
        continue
    mean_count = average_occurrence

    # Perform prediction using the model
    all_predictions = []
    
    for tail_entity, relation in scenario:
        predicted_heads = predict_target(
            model=result.model,
            relation=relation,
            tail=tail_entity,
            triples_factory=result.training
        ).df
        all_predictions.append(predicted_heads)

    all_predictions_df = pd.concat(all_predictions, ignore_index=True)
    aggregated_predictions = all_predictions_df.groupby('head_label').agg(
        total_score=('score', 'sum')
    ).reset_index()

    final_predictions_sorted = aggregated_predictions.sort_values(by="total_score", ascending=False)

    # Calculate Precision and Recall using the same Top N
    top_n_scores = final_predictions_sorted["head_label"].head(mean_count)
    matching_recipes = filtered_recipes[filtered_recipes['Name'].isin(top_n_scores)].drop_duplicates(subset='Name')

    # Calculate sets of predicted and relevant recipes
    predicted_recipes = set(top_n_scores)
    relevant_recipes = set(filtered_recipes['Name'])

    # Calculate Precision and Recall using the same Top N
    top_n_scores_accuracy = final_predictions_sorted["head_label"].head(expected_criteria_match_number)
    matching_recipes_accuracy = filtered_recipes[filtered_recipes['Name'].isin(top_n_scores_accuracy)]

    true_positives = predicted_recipes.intersection(relevant_recipes)
    precision = len(true_positives) / len(predicted_recipes) if len(predicted_recipes) > 0 else 0
    # Recall Calculation
    Recall = len(true_positives) / len(relevant_recipes) if len(relevant_recipes) > 0 else 0

    Accuracy = len(matching_recipes_accuracy) / expected_criteria_match_number if expected_criteria_match_number > 0 else 0


    # Precision Calculation
    predicted_recipes = set(top_n_scores)
    relevant_recipes = set(filtered_recipes['Name'])


    # F1 Score Calculation
    if precision + Recall > 0:
        F1 = 2 * (precision * Recall) / (precision + Recall)
    else:
        F1 = 0

    # Store results
    result_dict = {
        "Scenario": idx + 1,
        "Criteria": scenario,
        "Expected Matches For Recall": expected_criteria_match_number,
        "Predicted Matches (Recall)": len(matching_recipes),
        "Recall": Recall,
        "Expected Matches For Precision": mean_count,
        "Predicted Matches (Precision)": len(matching_recipes),
        "Precision": precision,
        "F1 Score": F1,  # Add F1 Score to results,
        "Expected Matched For Accuracy":expected_criteria_match_number,
        "Predicted Matched (Accuracy)":len(matching_recipes_accuracy),
        "Accuracy":Accuracy
    }
    
    test_results.append(result_dict)

# Convert results into DataFrame
results_df = pd.DataFrame(test_results)


results_df.to_csv('/app/RotatEEvulationResults/five_criteria_rotate.csv',index=False)

from itertools import combinations, product
import random

# Set the random seed for reproducibility
random.seed(42)

# Extract the top 20 most common ingredients from 'best_foodentityname'
ingredient_counts = recipes_df['best_foodentityname'].apply(lambda x: [ingredient.strip() for ingredient in x.split(',')]).explode().value_counts()
top_ingredients = ingredient_counts.head(20).index.tolist()

# Define relation options based on the columns in your dataset
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


# Generate binary combinations (pairs of criteria)
three_criteria = generate_specific_combinations(relation_options, combination_size=6)

# List of 5000 random samples from the generated three_criteria list
random_6_user_criteria = random.choices(three_criteria, k=50000)

print(random_6_user_criteria)

# Display the first 20 combinations
random_6_user_criteria[0:20]


test_scenarios = random_6_user_criteria

# Sonucu hesapla
average_occurrence = calculate_average_occurrence(test_scenarios, recipes_df)
average_occurrence = int(average_occurrence)
# Sonucu yazdırma
print("\nAverage Occurrence for Scenarios:", average_occurrence)

# Initialize a list to store test results
test_results = []

# Iterate through each test scenario
for idx, scenario in enumerate(test_scenarios):
    
    filtered_recipes = recipes_df

    for tail_entity, relation in scenario:
        if relation == "hasDietType":
            filtered_recipes = filtered_recipes[filtered_recipes['Diet_Types'].str.contains(tail_entity, case=False, na=False)]
        elif relation == "isForMealType":
            filtered_recipes = filtered_recipes[filtered_recipes['meal_type'].str.contains(tail_entity, case=False, na=False)]
        elif relation == "isFromRegion":
            filtered_recipes = filtered_recipes[filtered_recipes['CleanedRegion'].str.contains(tail_entity, case=False, na=False)]
        elif relation == "needTimeToCook":
            filtered_recipes = filtered_recipes[filtered_recipes['cook_time'].str.contains(tail_entity, case=False, na=False)]
        elif relation in ["HasCarbLevel", "HasProteinLevel", "HasFatLevel", "HasFiberLevel", "HasSodiumLevel", "HasSugarLevel", "HasCholesterolLevel", "HasCalorieLevel"]:
            filtered_recipes = filtered_recipes[filtered_recipes['Healthy_Type'].str.contains(tail_entity, case=False, na=False)]

    expected_criteria_match_number = len(filtered_recipes)
    # Skip the scenario if there are no matches
    if expected_criteria_match_number == 0:
        #print(f"Skipping Scenario {idx+1} as no relevant matches were found.")
        continue
    mean_count = average_occurrence

    # Perform prediction using the model
    all_predictions = []
    
    for tail_entity, relation in scenario:
        predicted_heads = predict_target(
            model=result.model,
            relation=relation,
            tail=tail_entity,
            triples_factory=result.training
        ).df
        all_predictions.append(predicted_heads)

    all_predictions_df = pd.concat(all_predictions, ignore_index=True)
    aggregated_predictions = all_predictions_df.groupby('head_label').agg(
        total_score=('score', 'sum')
    ).reset_index()

    final_predictions_sorted = aggregated_predictions.sort_values(by="total_score", ascending=False)

    # Calculate Precision and Recall using the same Top N
    top_n_scores = final_predictions_sorted["head_label"].head(mean_count)
    matching_recipes = filtered_recipes[filtered_recipes['Name'].isin(top_n_scores)].drop_duplicates(subset='Name')

    # Calculate sets of predicted and relevant recipes
    predicted_recipes = set(top_n_scores)
    relevant_recipes = set(filtered_recipes['Name'])

    # Calculate Precision and Recall using the same Top N
    top_n_scores_accuracy = final_predictions_sorted["head_label"].head(expected_criteria_match_number)
    matching_recipes_accuracy = filtered_recipes[filtered_recipes['Name'].isin(top_n_scores_accuracy)]

    true_positives = predicted_recipes.intersection(relevant_recipes)
    precision = len(true_positives) / len(predicted_recipes) if len(predicted_recipes) > 0 else 0
    # Recall Calculation
    Recall = len(true_positives) / len(relevant_recipes) if len(relevant_recipes) > 0 else 0

    Accuracy = len(matching_recipes_accuracy) / expected_criteria_match_number if expected_criteria_match_number > 0 else 0


    # Precision Calculation
    predicted_recipes = set(top_n_scores)
    relevant_recipes = set(filtered_recipes['Name'])


    # F1 Score Calculation
    if precision + Recall > 0:
        F1 = 2 * (precision * Recall) / (precision + Recall)
    else:
        F1 = 0

    # Store results
    result_dict = {
        "Scenario": idx + 1,
        "Criteria": scenario,
        "Expected Matches For Recall": expected_criteria_match_number,
        "Predicted Matches (Recall)": len(matching_recipes),
        "Recall": Recall,
        "Expected Matches For Precision": mean_count,
        "Predicted Matches (Precision)": len(matching_recipes),
        "Precision": precision,
        "F1 Score": F1,  # Add F1 Score to results,
        "Expected Matched For Accuracy":expected_criteria_match_number,
        "Predicted Matched (Accuracy)":len(matching_recipes_accuracy),
        "Accuracy":Accuracy
    }
    
    test_results.append(result_dict)

# Convert results into DataFrame
results_df = pd.DataFrame(test_results)

results_df.to_csv('/app/RotatEEvulationResults/six_criteria_rotate.csv',index=False)

from itertools import combinations, product
import random

# Set the random seed for reproducibility
random.seed(42)

# Extract the top 20 most common ingredients from 'best_foodentityname'
ingredient_counts = recipes_df['best_foodentityname'].apply(lambda x: [ingredient.strip() for ingredient in x.split(',')]).explode().value_counts()
top_ingredients = ingredient_counts.head(20).index.tolist()

# Define relation options based on the columns in your dataset
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


# Generate binary combinations (pairs of criteria)
three_criteria = generate_specific_combinations(relation_options, combination_size=7)

# List of 5000 random samples from the generated three_criteria list
random_7_user_criteria = random.choices(three_criteria, k=50000)

print(random_7_user_criteria)

# Display the first 20 combinations
random_7_user_criteria[0:20]



test_scenarios = random_7_user_criteria

# Sonucu hesapla
average_occurrence = calculate_average_occurrence(test_scenarios, recipes_df)
average_occurrence = int(average_occurrence)
# Sonucu yazdırma
print("\nAverage Occurrence for Scenarios:", average_occurrence)

# Initialize a list to store test results
test_results = []

# Iterate through each test scenario
for idx, scenario in enumerate(test_scenarios):
    
    filtered_recipes = recipes_df

    for tail_entity, relation in scenario:
        if relation == "hasDietType":
            filtered_recipes = filtered_recipes[filtered_recipes['Diet_Types'].str.contains(tail_entity, case=False, na=False)]
        elif relation == "isForMealType":
            filtered_recipes = filtered_recipes[filtered_recipes['meal_type'].str.contains(tail_entity, case=False, na=False)]
        elif relation == "isFromRegion":
            filtered_recipes = filtered_recipes[filtered_recipes['CleanedRegion'].str.contains(tail_entity, case=False, na=False)]
        elif relation == "needTimeToCook":
            filtered_recipes = filtered_recipes[filtered_recipes['cook_time'].str.contains(tail_entity, case=False, na=False)]
        elif relation in ["HasCarbLevel", "HasProteinLevel", "HasFatLevel", "HasFiberLevel", "HasSodiumLevel", "HasSugarLevel", "HasCholesterolLevel", "HasCalorieLevel"]:
            filtered_recipes = filtered_recipes[filtered_recipes['Healthy_Type'].str.contains(tail_entity, case=False, na=False)]

    expected_criteria_match_number = len(filtered_recipes)
    # Skip the scenario if there are no matches
    if expected_criteria_match_number == 0:
        #print(f"Skipping Scenario {idx+1} as no relevant matches were found.")
        continue
    mean_count = average_occurrence

    # Perform prediction using the model
    all_predictions = []
    
    for tail_entity, relation in scenario:
        predicted_heads = predict_target(
            model=result.model,
            relation=relation,
            tail=tail_entity,
            triples_factory=result.training
        ).df
        all_predictions.append(predicted_heads)

    all_predictions_df = pd.concat(all_predictions, ignore_index=True)
    aggregated_predictions = all_predictions_df.groupby('head_label').agg(
        total_score=('score', 'sum')
    ).reset_index()

    final_predictions_sorted = aggregated_predictions.sort_values(by="total_score", ascending=False)

    # Calculate Precision and Recall using the same Top N
    top_n_scores = final_predictions_sorted["head_label"].head(mean_count)
    matching_recipes = filtered_recipes[filtered_recipes['Name'].isin(top_n_scores)].drop_duplicates(subset='Name')

    # Calculate sets of predicted and relevant recipes
    predicted_recipes = set(top_n_scores)
    relevant_recipes = set(filtered_recipes['Name'])

    # Calculate Precision and Recall using the same Top N
    top_n_scores_accuracy = final_predictions_sorted["head_label"].head(expected_criteria_match_number)
    matching_recipes_accuracy = filtered_recipes[filtered_recipes['Name'].isin(top_n_scores_accuracy)]

    true_positives = predicted_recipes.intersection(relevant_recipes)
    precision = len(true_positives) / len(predicted_recipes) if len(predicted_recipes) > 0 else 0
    # Recall Calculation
    Recall = len(true_positives) / len(relevant_recipes) if len(relevant_recipes) > 0 else 0

    Accuracy = len(matching_recipes_accuracy) / expected_criteria_match_number if expected_criteria_match_number > 0 else 0


    # Precision Calculation
    predicted_recipes = set(top_n_scores)
    relevant_recipes = set(filtered_recipes['Name'])


    # F1 Score Calculation
    if precision + Recall > 0:
        F1 = 2 * (precision * Recall) / (precision + Recall)
    else:
        F1 = 0

    # Store results
    result_dict = {
        "Scenario": idx + 1,
        "Criteria": scenario,
        "Expected Matches For Recall": expected_criteria_match_number,
        "Predicted Matches (Recall)": len(matching_recipes),
        "Recall": Recall,
        "Expected Matches For Precision": mean_count,
        "Predicted Matches (Precision)": len(matching_recipes),
        "Precision": precision,
        "F1 Score": F1,  # Add F1 Score to results,
        "Expected Matched For Accuracy":expected_criteria_match_number,
        "Predicted Matched (Accuracy)":len(matching_recipes_accuracy),
        "Accuracy":Accuracy
    }
    
    test_results.append(result_dict)

# Convert results into DataFrame
results_df = pd.DataFrame(test_results)


results_df.to_csv('/app/RotatEEvulationResults/seven_criteria_rotate.csv',index=False)

from itertools import combinations, product
import random

# Set the random seed for reproducibility
random.seed(42)

# Extract the top 20 most common ingredients from 'best_foodentityname'
ingredient_counts = recipes_df['best_foodentityname'].apply(lambda x: [ingredient.strip() for ingredient in x.split(',')]).explode().value_counts()
top_ingredients = ingredient_counts.head(20).index.tolist()

# Define relation options based on the columns in your dataset
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


# Generate binary combinations (pairs of criteria)
three_criteria = generate_specific_combinations(relation_options, combination_size=8)

# List of 5000 random samples from the generated three_criteria list
random_8_user_criteria = random.choices(three_criteria, k=50000)

print(random_8_user_criteria)

# Display the first 20 combinations
random_8_user_criteria[0:20]


test_scenarios = random_8_user_criteria

# Sonucu hesapla
average_occurrence = calculate_average_occurrence(test_scenarios, recipes_df)
average_occurrence = int(average_occurrence)
# Sonucu yazdırma
print("\nAverage Occurrence for Scenarios:", average_occurrence)

# Initialize a list to store test results
test_results = []

# Iterate through each test scenario
for idx, scenario in enumerate(test_scenarios):
    
    filtered_recipes = recipes_df

    for tail_entity, relation in scenario:
        if relation == "hasDietType":
            filtered_recipes = filtered_recipes[filtered_recipes['Diet_Types'].str.contains(tail_entity, case=False, na=False)]
        elif relation == "isForMealType":
            filtered_recipes = filtered_recipes[filtered_recipes['meal_type'].str.contains(tail_entity, case=False, na=False)]
        elif relation == "isFromRegion":
            filtered_recipes = filtered_recipes[filtered_recipes['CleanedRegion'].str.contains(tail_entity, case=False, na=False)]
        elif relation == "needTimeToCook":
            filtered_recipes = filtered_recipes[filtered_recipes['cook_time'].str.contains(tail_entity, case=False, na=False)]
        elif relation in ["HasCarbLevel", "HasProteinLevel", "HasFatLevel", "HasFiberLevel", "HasSodiumLevel", "HasSugarLevel", "HasCholesterolLevel", "HasCalorieLevel"]:
            filtered_recipes = filtered_recipes[filtered_recipes['Healthy_Type'].str.contains(tail_entity, case=False, na=False)]

    expected_criteria_match_number = len(filtered_recipes)
    # Skip the scenario if there are no matches
    if expected_criteria_match_number == 0:
        #print(f"Skipping Scenario {idx+1} as no relevant matches were found.")
        continue
    mean_count = average_occurrence

    # Perform prediction using the model
    all_predictions = []
    
    for tail_entity, relation in scenario:
        predicted_heads = predict_target(
            model=result.model,
            relation=relation,
            tail=tail_entity,
            triples_factory=result.training
        ).df
        all_predictions.append(predicted_heads)

    all_predictions_df = pd.concat(all_predictions, ignore_index=True)
    aggregated_predictions = all_predictions_df.groupby('head_label').agg(
        total_score=('score', 'sum')
    ).reset_index()

    final_predictions_sorted = aggregated_predictions.sort_values(by="total_score", ascending=False)

    # Calculate Precision and Recall using the same Top N
    top_n_scores = final_predictions_sorted["head_label"].head(mean_count)
    matching_recipes = filtered_recipes[filtered_recipes['Name'].isin(top_n_scores)].drop_duplicates(subset='Name')

    # Calculate sets of predicted and relevant recipes
    predicted_recipes = set(top_n_scores)
    relevant_recipes = set(filtered_recipes['Name'])

    # Calculate Precision and Recall using the same Top N
    top_n_scores_accuracy = final_predictions_sorted["head_label"].head(expected_criteria_match_number)
    matching_recipes_accuracy = filtered_recipes[filtered_recipes['Name'].isin(top_n_scores_accuracy)]

    true_positives = predicted_recipes.intersection(relevant_recipes)
    precision = len(true_positives) / len(predicted_recipes) if len(predicted_recipes) > 0 else 0
    # Recall Calculation
    Recall = len(true_positives) / len(relevant_recipes) if len(relevant_recipes) > 0 else 0

    Accuracy = len(matching_recipes_accuracy) / expected_criteria_match_number if expected_criteria_match_number > 0 else 0


    # Precision Calculation
    predicted_recipes = set(top_n_scores)
    relevant_recipes = set(filtered_recipes['Name'])


    # F1 Score Calculation
    if precision + Recall > 0:
        F1 = 2 * (precision * Recall) / (precision + Recall)
    else:
        F1 = 0

    # Store results
    result_dict = {
        "Scenario": idx + 1,
        "Criteria": scenario,
        "Expected Matches For Recall": expected_criteria_match_number,
        "Predicted Matches (Recall)": len(matching_recipes),
        "Recall": Recall,
        "Expected Matches For Precision": mean_count,
        "Predicted Matches (Precision)": len(matching_recipes),
        "Precision": precision,
        "F1 Score": F1,  # Add F1 Score to results,
        "Expected Matched For Accuracy":expected_criteria_match_number,
        "Predicted Matched (Accuracy)":len(matching_recipes_accuracy),
        "Accuracy":Accuracy
    }
    
    test_results.append(result_dict)

# Convert results into DataFrame
results_df = pd.DataFrame(test_results)

results_df.to_csv('/app/RotatEEvulationResults/eight_criteria_rotate.csv',index=False)




#import library 
import networkx as nx
import numpy as np
from pykeen.pipeline import pipeline
from pykeen.triples import TriplesFactory
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from pykeen.hpo import hpo_pipeline


# Load the CSV file
file_path = '/app/BalancedRecipe_entity_linking.csv'
recipes_df = pd.read_csv(file_path)
recipes_df = recipes_df.drop_duplicates(subset='Name', keep='first')  # Ensure unique recipes
recipes_df = recipes_df.reset_index(drop=True)  #

UNKNOWN_PLACEHOLDER = 'Unknown'


# Convert the dataframe into the required format for the graph

recipes = process_recipes(recipes_df)

# Process the recipes to create the graph and triples
G, triples_array = create_graph_and_triples(recipes)


# Convert the triples array into a pandas DataFrame
triples_df = pd.DataFrame(triples_array, columns=['Head', 'Relation', 'Tail'])


triples_df.to_csv('/app/recipes_triples_10000sample.csv',index=False)



import pandas as pd 
df=pd.read_csv('/app/recipes_triples_10000sample.csv')
df=df.dropna()
#df_filtered = df[df['Relation'] != 'contains']

# Assuming df is your DataFrame
triples = df[['Head', 'Relation', 'Tail']].values  # This returns a NumPy array directly

# Create a TriplesFactory
triples_factory = TriplesFactory.from_labeled_triples(triples)

#training, testing, validation = triples_factory.split([7., .2, .1],randomize_cleanup=True)


# Use the same triples_factory for both training and testing
result = pipeline(
    model='QuatE',
    training=triples_factory,  # Use the entire triples factory for training
    testing=triples_factory,   # Use the same factory for testing
    validation=triples_factory,
    epochs=150,
    stopper='early',
)

print(f"Hits@1: {result.metric_results.get_metric('hits_at_1')}")
print(f"Hits@3: {result.metric_results.get_metric('hits_at_3')}")
print(f"Hits@10: {result.metric_results.get_metric('hits_at_10')}")
print(f"Mean Rank (Realistic): {result.metric_results.get_metric('both.realistic.mean_rank')}")
print(f"Mean Reciprocal Rank (Realistic): {result.metric_results.get_metric('both.realistic.mean_reciprocal_rank')}")




import pandas as pd
from pykeen.predict import predict_target
from itertools import product

# Load the data
triples_df = pd.read_csv('/app/recipes_triples_10000sample.csv')
file_path = '/app/BalancedRecipe_entity_linking.csv'
recipes_df = pd.read_csv(file_path)
recipes_df = recipes_df.drop_duplicates(subset='Name', keep='first')  # Ensure unique recipes
recipes_df = recipes_df.reset_index(drop=True)


# Process the dataframe
recipes_df = process_recipes_dataframe(recipes_df)

# Extract the top 50 most common ingredients from 'best_foodentityname'
ingredient_counts = recipes_df['best_foodentityname'].apply(lambda x: [ingredient.strip() for ingredient in x.split(',')]).explode().value_counts()
top_ingredients = ingredient_counts.head(25).index.tolist()

from itertools import combinations, product

# Extract the top 20 most common ingredients from 'best_foodentityname'
ingredient_counts = recipes_df['best_foodentityname'].apply(lambda x: [ingredient.strip() for ingredient in x.split(',')]).explode().value_counts()
top_ingredients = ingredient_counts.head(20).index.tolist()

# Define relation options based on the columns in your dataset
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

# Generate binary combinations (pairs of criteria)
one_criteria = generate_specific_combinations(relation_options, combination_size=1)

test_scenarios = one_criteria


# Sonucu hesapla
average_occurrence = calculate_average_occurrence(test_scenarios, recipes_df)
average_occurrence = int(average_occurrence)
# Sonucu yazdırma
print("\nAverage Occurrence for Scenarios:", average_occurrence)

# Initialize a list to store test results
test_results = []

# Iterate through each test scenario
for idx, scenario in enumerate(test_scenarios):
    
    filtered_recipes = recipes_df

    for tail_entity, relation in scenario:
        if relation == "hasDietType":
            filtered_recipes = filtered_recipes[filtered_recipes['Diet_Types'].str.contains(tail_entity, case=False, na=False)]
        elif relation == "isForMealType":
            filtered_recipes = filtered_recipes[filtered_recipes['meal_type'].str.contains(tail_entity, case=False, na=False)]
        elif relation == "isFromRegion":
            filtered_recipes = filtered_recipes[filtered_recipes['CleanedRegion'].str.contains(tail_entity, case=False, na=False)]
        elif relation == "needTimeToCook":
            filtered_recipes = filtered_recipes[filtered_recipes['cook_time'].str.contains(tail_entity, case=False, na=False)]
        elif relation in ["HasCarbLevel", "HasProteinLevel", "HasFatLevel", "HasFiberLevel", "HasSodiumLevel", "HasSugarLevel", "HasCholesterolLevel", "HasCalorieLevel"]:
            filtered_recipes = filtered_recipes[filtered_recipes['Healthy_Type'].str.contains(tail_entity, case=False, na=False)]

    expected_criteria_match_number = len(filtered_recipes)
    # Skip the scenario if there are no matches
    if expected_criteria_match_number == 0:
        #print(f"Skipping Scenario {idx+1} as no relevant matches were found.")
        continue
    mean_count = average_occurrence

    # Perform prediction using the model
    all_predictions = []
    
    for tail_entity, relation in scenario:
        predicted_heads = predict_target(
            model=result.model,
            relation=relation,
            tail=tail_entity,
            triples_factory=result.training
        ).df
        all_predictions.append(predicted_heads)

    all_predictions_df = pd.concat(all_predictions, ignore_index=True)
    aggregated_predictions = all_predictions_df.groupby('head_label').agg(
        total_score=('score', 'sum')
    ).reset_index()

    final_predictions_sorted = aggregated_predictions.sort_values(by="total_score", ascending=False)

    # Calculate Precision and Recall using the same Top N
    top_n_scores = final_predictions_sorted["head_label"].head(mean_count)
    matching_recipes = filtered_recipes[filtered_recipes['Name'].isin(top_n_scores)].drop_duplicates(subset='Name')

    # Calculate sets of predicted and relevant recipes
    predicted_recipes = set(top_n_scores)
    relevant_recipes = set(filtered_recipes['Name'])

    # Calculate Precision and Recall using the same Top N
    top_n_scores_accuracy = final_predictions_sorted["head_label"].head(expected_criteria_match_number)
    matching_recipes_accuracy = filtered_recipes[filtered_recipes['Name'].isin(top_n_scores_accuracy)]

    true_positives = predicted_recipes.intersection(relevant_recipes)
    precision = len(true_positives) / len(predicted_recipes) if len(predicted_recipes) > 0 else 0
    # Recall Calculation
    Recall = len(true_positives) / len(relevant_recipes) if len(relevant_recipes) > 0 else 0

    Accuracy = len(matching_recipes_accuracy) / expected_criteria_match_number if expected_criteria_match_number > 0 else 0


    # Precision Calculation
    predicted_recipes = set(top_n_scores)
    relevant_recipes = set(filtered_recipes['Name'])


    # F1 Score Calculation
    if precision + Recall > 0:
        F1 = 2 * (precision * Recall) / (precision + Recall)
    else:
        F1 = 0

    # Store results
    result_dict = {
        "Scenario": idx + 1,
        "Criteria": scenario,
        "Expected Matches For Recall": expected_criteria_match_number,
        "Predicted Matches (Recall)": len(matching_recipes),
        "Recall": Recall,
        "Expected Matches For Precision": mean_count,
        "Predicted Matches (Precision)": len(matching_recipes),
        "Precision": precision,
        "F1 Score": F1,  # Add F1 Score to results,
        "Expected Matched For Accuracy":expected_criteria_match_number,
        "Predicted Matched (Accuracy)":len(matching_recipes_accuracy),
        "Accuracy":Accuracy
    }
    
    test_results.append(result_dict)

# Convert results into DataFrame
results_df = pd.DataFrame(test_results)

# Display the results
#results_df


results_df.to_csv('/app/QuatEEvulationResults/one_criteria_QuatE.csv',index=False)

from itertools import combinations, product

# Extract the top 20 most common ingredients from 'best_foodentityname'
ingredient_counts = recipes_df['best_foodentityname'].apply(lambda x: [ingredient.strip() for ingredient in x.split(',')]).explode().value_counts()
top_ingredients = ingredient_counts.head(20).index.tolist()

# Define relation options based on the columns in your dataset
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

# Generate binary combinations (pairs of criteria)
two_criteria = generate_specific_combinations(relation_options, combination_size=2)


test_scenarios = two_criteria

# Sonucu hesapla
average_occurrence = calculate_average_occurrence(test_scenarios, recipes_df)
average_occurrence = int(average_occurrence)
# Sonucu yazdırma
print("\nAverage Occurrence for Scenarios:", average_occurrence)

# Initialize a list to store test results
test_results = []

# Iterate through each test scenario
for idx, scenario in enumerate(test_scenarios):
    
    filtered_recipes = recipes_df

    for tail_entity, relation in scenario:
        if relation == "hasDietType":
            filtered_recipes = filtered_recipes[filtered_recipes['Diet_Types'].str.contains(tail_entity, case=False, na=False)]
        elif relation == "isForMealType":
            filtered_recipes = filtered_recipes[filtered_recipes['meal_type'].str.contains(tail_entity, case=False, na=False)]
        elif relation == "isFromRegion":
            filtered_recipes = filtered_recipes[filtered_recipes['CleanedRegion'].str.contains(tail_entity, case=False, na=False)]
        elif relation == "needTimeToCook":
            filtered_recipes = filtered_recipes[filtered_recipes['cook_time'].str.contains(tail_entity, case=False, na=False)]
        elif relation in ["HasCarbLevel", "HasProteinLevel", "HasFatLevel", "HasFiberLevel", "HasSodiumLevel", "HasSugarLevel", "HasCholesterolLevel", "HasCalorieLevel"]:
            filtered_recipes = filtered_recipes[filtered_recipes['Healthy_Type'].str.contains(tail_entity, case=False, na=False)]

    expected_criteria_match_number = len(filtered_recipes)
    # Skip the scenario if there are no matches
    if expected_criteria_match_number == 0:
        #print(f"Skipping Scenario {idx+1} as no relevant matches were found.")
        continue
    mean_count = average_occurrence

    # Perform prediction using the model
    all_predictions = []
    
    for tail_entity, relation in scenario:
        predicted_heads = predict_target(
            model=result.model,
            relation=relation,
            tail=tail_entity,
            triples_factory=result.training
        ).df
        all_predictions.append(predicted_heads)

    all_predictions_df = pd.concat(all_predictions, ignore_index=True)
    aggregated_predictions = all_predictions_df.groupby('head_label').agg(
        total_score=('score', 'sum')
    ).reset_index()

    final_predictions_sorted = aggregated_predictions.sort_values(by="total_score", ascending=False)

    # Calculate Precision and Recall using the same Top N
    top_n_scores = final_predictions_sorted["head_label"].head(mean_count)
    matching_recipes = filtered_recipes[filtered_recipes['Name'].isin(top_n_scores)].drop_duplicates(subset='Name')

    # Calculate sets of predicted and relevant recipes
    predicted_recipes = set(top_n_scores)
    relevant_recipes = set(filtered_recipes['Name'])

    # Calculate Precision and Recall using the same Top N
    top_n_scores_accuracy = final_predictions_sorted["head_label"].head(expected_criteria_match_number)
    matching_recipes_accuracy = filtered_recipes[filtered_recipes['Name'].isin(top_n_scores_accuracy)]

    true_positives = predicted_recipes.intersection(relevant_recipes)
    precision = len(true_positives) / len(predicted_recipes) if len(predicted_recipes) > 0 else 0
    # Recall Calculation
    Recall = len(true_positives) / len(relevant_recipes) if len(relevant_recipes) > 0 else 0

    Accuracy = len(matching_recipes_accuracy) / expected_criteria_match_number if expected_criteria_match_number > 0 else 0


    # Precision Calculation
    predicted_recipes = set(top_n_scores)
    relevant_recipes = set(filtered_recipes['Name'])


    # F1 Score Calculation
    if precision + Recall > 0:
        F1 = 2 * (precision * Recall) / (precision + Recall)
    else:
        F1 = 0

    # Store results
    result_dict = {
        "Scenario": idx + 1,
        "Criteria": scenario,
        "Expected Matches For Recall": expected_criteria_match_number,
        "Predicted Matches (Recall)": len(matching_recipes),
        "Recall": Recall,
        "Expected Matches For Precision": mean_count,
        "Predicted Matches (Precision)": len(matching_recipes),
        "Precision": precision,
        "F1 Score": F1,  # Add F1 Score to results,
        "Expected Matched For Accuracy":expected_criteria_match_number,
        "Predicted Matched (Accuracy)":len(matching_recipes_accuracy),
        "Accuracy":Accuracy
    }
    
    test_results.append(result_dict)

# Convert results into DataFrame
results_df = pd.DataFrame(test_results)


results_df.to_csv('/app/QuatEEvulationResults/two_criteria_QuatE.csv',index=False)


from itertools import combinations, product
import random

# Set the random seed for reproducibility
random.seed(42)

# Extract the top 20 most common ingredients from 'best_foodentityname'
ingredient_counts = recipes_df['best_foodentityname'].apply(lambda x: [ingredient.strip() for ingredient in x.split(',')]).explode().value_counts()
top_ingredients = ingredient_counts.head(20).index.tolist()

# Define relation options based on the columns in your dataset
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

# Generate binary combinations (pairs of criteria)
three_criteria = generate_specific_combinations(relation_options, combination_size=3)

# List of 5000 random samples from the generated three_criteria list
random_3_user_criteria = random.choices(three_criteria, k=25000)

print(random_3_user_criteria)

# Display the first 20 combinations
random_3_user_criteria[0:20]


test_scenarios = random_3_user_criteria
# Sonucu hesapla
average_occurrence = calculate_average_occurrence(test_scenarios, recipes_df)
average_occurrence = int(average_occurrence)
# Sonucu yazdırma
print("\nAverage Occurrence for Scenarios:", average_occurrence)

# Initialize a list to store test results
test_results = []

# Iterate through each test scenario
for idx, scenario in enumerate(test_scenarios):
    
    filtered_recipes = recipes_df

    for tail_entity, relation in scenario:
        if relation == "hasDietType":
            filtered_recipes = filtered_recipes[filtered_recipes['Diet_Types'].str.contains(tail_entity, case=False, na=False)]
        elif relation == "isForMealType":
            filtered_recipes = filtered_recipes[filtered_recipes['meal_type'].str.contains(tail_entity, case=False, na=False)]
        elif relation == "isFromRegion":
            filtered_recipes = filtered_recipes[filtered_recipes['CleanedRegion'].str.contains(tail_entity, case=False, na=False)]
        elif relation == "needTimeToCook":
            filtered_recipes = filtered_recipes[filtered_recipes['cook_time'].str.contains(tail_entity, case=False, na=False)]
        elif relation in ["HasCarbLevel", "HasProteinLevel", "HasFatLevel", "HasFiberLevel", "HasSodiumLevel", "HasSugarLevel", "HasCholesterolLevel", "HasCalorieLevel"]:
            filtered_recipes = filtered_recipes[filtered_recipes['Healthy_Type'].str.contains(tail_entity, case=False, na=False)]

    expected_criteria_match_number = len(filtered_recipes)
    # Skip the scenario if there are no matches
    if expected_criteria_match_number == 0:
        #print(f"Skipping Scenario {idx+1} as no relevant matches were found.")
        continue
    mean_count = average_occurrence

    # Perform prediction using the model
    all_predictions = []
    
    for tail_entity, relation in scenario:
        predicted_heads = predict_target(
            model=result.model,
            relation=relation,
            tail=tail_entity,
            triples_factory=result.training
        ).df
        all_predictions.append(predicted_heads)

    all_predictions_df = pd.concat(all_predictions, ignore_index=True)
    aggregated_predictions = all_predictions_df.groupby('head_label').agg(
        total_score=('score', 'sum')
    ).reset_index()

    final_predictions_sorted = aggregated_predictions.sort_values(by="total_score", ascending=False)

    # Calculate Precision and Recall using the same Top N
    top_n_scores = final_predictions_sorted["head_label"].head(mean_count)
    matching_recipes = filtered_recipes[filtered_recipes['Name'].isin(top_n_scores)].drop_duplicates(subset='Name')

    # Calculate sets of predicted and relevant recipes
    predicted_recipes = set(top_n_scores)
    relevant_recipes = set(filtered_recipes['Name'])

    # Calculate Precision and Recall using the same Top N
    top_n_scores_accuracy = final_predictions_sorted["head_label"].head(expected_criteria_match_number)
    matching_recipes_accuracy = filtered_recipes[filtered_recipes['Name'].isin(top_n_scores_accuracy)]

    true_positives = predicted_recipes.intersection(relevant_recipes)
    precision = len(true_positives) / len(predicted_recipes) if len(predicted_recipes) > 0 else 0
    # Recall Calculation
    Recall = len(true_positives) / len(relevant_recipes) if len(relevant_recipes) > 0 else 0

    Accuracy = len(matching_recipes_accuracy) / expected_criteria_match_number if expected_criteria_match_number > 0 else 0


    # Precision Calculation
    predicted_recipes = set(top_n_scores)
    relevant_recipes = set(filtered_recipes['Name'])


    # F1 Score Calculation
    if precision + Recall > 0:
        F1 = 2 * (precision * Recall) / (precision + Recall)
    else:
        F1 = 0

    # Store results
    result_dict = {
        "Scenario": idx + 1,
        "Criteria": scenario,
        "Expected Matches For Recall": expected_criteria_match_number,
        "Predicted Matches (Recall)": len(matching_recipes),
        "Recall": Recall,
        "Expected Matches For Precision": mean_count,
        "Predicted Matches (Precision)": len(matching_recipes),
        "Precision": precision,
        "F1 Score": F1,  # Add F1 Score to results,
        "Expected Matched For Accuracy":expected_criteria_match_number,
        "Predicted Matched (Accuracy)":len(matching_recipes_accuracy),
        "Accuracy":Accuracy
    }
    
    test_results.append(result_dict)

# Convert results into DataFrame
results_df = pd.DataFrame(test_results)

results_df.to_csv('/app/QuatEEvulationResults/three_criteria_QuatE.csv',index=False)

from itertools import combinations, product
import random

# Set the random seed for reproducibility
random.seed(42)

# Extract the top 20 most common ingredients from 'best_foodentityname'
ingredient_counts = recipes_df['best_foodentityname'].apply(lambda x: [ingredient.strip() for ingredient in x.split(',')]).explode().value_counts()
top_ingredients = ingredient_counts.head(20).index.tolist()

# Define relation options based on the columns in your dataset
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


# Generate binary combinations (pairs of criteria)
four_criteria = generate_specific_combinations(relation_options, combination_size=4)

# List of 5000 random samples from the generated three_criteria list
random_4_user_criteria = random.choices(four_criteria, k=50000)

print(random_4_user_criteria)

# Display the first 20 combinations
random_4_user_criteria[0:20]

test_scenarios = random_4_user_criteria
# Function to calculate co-occurrence

# Sonucu hesapla
average_occurrence = calculate_average_occurrence(test_scenarios, recipes_df)
average_occurrence = int(average_occurrence)
# Sonucu yazdırma
print("\nAverage Occurrence for Scenarios:", average_occurrence)

# Initialize a list to store test results
test_results = []

# Iterate through each test scenario
for idx, scenario in enumerate(test_scenarios):
    
    filtered_recipes = recipes_df

    for tail_entity, relation in scenario:
        if relation == "hasDietType":
            filtered_recipes = filtered_recipes[filtered_recipes['Diet_Types'].str.contains(tail_entity, case=False, na=False)]
        elif relation == "isForMealType":
            filtered_recipes = filtered_recipes[filtered_recipes['meal_type'].str.contains(tail_entity, case=False, na=False)]
        elif relation == "isFromRegion":
            filtered_recipes = filtered_recipes[filtered_recipes['CleanedRegion'].str.contains(tail_entity, case=False, na=False)]
        elif relation == "needTimeToCook":
            filtered_recipes = filtered_recipes[filtered_recipes['cook_time'].str.contains(tail_entity, case=False, na=False)]
        elif relation in ["HasCarbLevel", "HasProteinLevel", "HasFatLevel", "HasFiberLevel", "HasSodiumLevel", "HasSugarLevel", "HasCholesterolLevel", "HasCalorieLevel"]:
            filtered_recipes = filtered_recipes[filtered_recipes['Healthy_Type'].str.contains(tail_entity, case=False, na=False)]

    expected_criteria_match_number = len(filtered_recipes)
    # Skip the scenario if there are no matches
    if expected_criteria_match_number == 0:
        #print(f"Skipping Scenario {idx+1} as no relevant matches were found.")
        continue
    mean_count = average_occurrence

    # Perform prediction using the model
    all_predictions = []
    
    for tail_entity, relation in scenario:
        predicted_heads = predict_target(
            model=result.model,
            relation=relation,
            tail=tail_entity,
            triples_factory=result.training
        ).df
        all_predictions.append(predicted_heads)

    all_predictions_df = pd.concat(all_predictions, ignore_index=True)
    aggregated_predictions = all_predictions_df.groupby('head_label').agg(
        total_score=('score', 'sum')
    ).reset_index()

    final_predictions_sorted = aggregated_predictions.sort_values(by="total_score", ascending=False)

    # Calculate Precision and Recall using the same Top N
    top_n_scores = final_predictions_sorted["head_label"].head(mean_count)
    matching_recipes = filtered_recipes[filtered_recipes['Name'].isin(top_n_scores)].drop_duplicates(subset='Name')

    # Calculate sets of predicted and relevant recipes
    predicted_recipes = set(top_n_scores)
    relevant_recipes = set(filtered_recipes['Name'])

    # Calculate Precision and Recall using the same Top N
    top_n_scores_accuracy = final_predictions_sorted["head_label"].head(expected_criteria_match_number)
    matching_recipes_accuracy = filtered_recipes[filtered_recipes['Name'].isin(top_n_scores_accuracy)]

    true_positives = predicted_recipes.intersection(relevant_recipes)
    precision = len(true_positives) / len(predicted_recipes) if len(predicted_recipes) > 0 else 0
    # Recall Calculation
    Recall = len(true_positives) / len(relevant_recipes) if len(relevant_recipes) > 0 else 0

    Accuracy = len(matching_recipes_accuracy) / expected_criteria_match_number if expected_criteria_match_number > 0 else 0


    # Precision Calculation
    predicted_recipes = set(top_n_scores)
    relevant_recipes = set(filtered_recipes['Name'])


    # F1 Score Calculation
    if precision + Recall > 0:
        F1 = 2 * (precision * Recall) / (precision + Recall)
    else:
        F1 = 0

    # Store results
    result_dict = {
        "Scenario": idx + 1,
        "Criteria": scenario,
        "Expected Matches For Recall": expected_criteria_match_number,
        "Predicted Matches (Recall)": len(matching_recipes),
        "Recall": Recall,
        "Expected Matches For Precision": mean_count,
        "Predicted Matches (Precision)": len(matching_recipes),
        "Precision": precision,
        "F1 Score": F1,  # Add F1 Score to results,
        "Expected Matched For Accuracy":expected_criteria_match_number,
        "Predicted Matched (Accuracy)":len(matching_recipes_accuracy),
        "Accuracy":Accuracy
    }
    
    test_results.append(result_dict)

# Convert results into DataFrame
results_df = pd.DataFrame(test_results)

results_df.to_csv('/app/QuatEEvulationResults/four_criteria_QuatE.csv',index=False)

from itertools import combinations, product
import random

# Set the random seed for reproducibility
random.seed(42)

# Extract the top 20 most common ingredients from 'best_foodentityname'
ingredient_counts = recipes_df['best_foodentityname'].apply(lambda x: [ingredient.strip() for ingredient in x.split(',')]).explode().value_counts()
top_ingredients = ingredient_counts.head(20).index.tolist()

# Define relation options based on the columns in your dataset
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

# Generate binary combinations (pairs of criteria)
three_criteria = generate_specific_combinations(relation_options, combination_size=5)

# List of 5000 random samples from the generated three_criteria list
random_5_user_criteria = random.choices(three_criteria, k=50000)

print(random_5_user_criteria)

# Display the first 20 combinations
random_5_user_criteria[0:20]

test_scenarios = random_5_user_criteria

# Sonucu hesapla
average_occurrence = calculate_average_occurrence(test_scenarios, recipes_df)
average_occurrence = int(average_occurrence)
# Sonucu yazdırma
print("\nAverage Occurrence for Scenarios:", average_occurrence)

# Initialize a list to store test results
test_results = []

# Iterate through each test scenario
for idx, scenario in enumerate(test_scenarios):
    
    filtered_recipes = recipes_df

    for tail_entity, relation in scenario:
        if relation == "hasDietType":
            filtered_recipes = filtered_recipes[filtered_recipes['Diet_Types'].str.contains(tail_entity, case=False, na=False)]
        elif relation == "isForMealType":
            filtered_recipes = filtered_recipes[filtered_recipes['meal_type'].str.contains(tail_entity, case=False, na=False)]
        elif relation == "isFromRegion":
            filtered_recipes = filtered_recipes[filtered_recipes['CleanedRegion'].str.contains(tail_entity, case=False, na=False)]
        elif relation == "needTimeToCook":
            filtered_recipes = filtered_recipes[filtered_recipes['cook_time'].str.contains(tail_entity, case=False, na=False)]
        elif relation in ["HasCarbLevel", "HasProteinLevel", "HasFatLevel", "HasFiberLevel", "HasSodiumLevel", "HasSugarLevel", "HasCholesterolLevel", "HasCalorieLevel"]:
            filtered_recipes = filtered_recipes[filtered_recipes['Healthy_Type'].str.contains(tail_entity, case=False, na=False)]

    expected_criteria_match_number = len(filtered_recipes)
    # Skip the scenario if there are no matches
    if expected_criteria_match_number == 0:
        #print(f"Skipping Scenario {idx+1} as no relevant matches were found.")
        continue
    mean_count = average_occurrence

    # Perform prediction using the model
    all_predictions = []
    
    for tail_entity, relation in scenario:
        predicted_heads = predict_target(
            model=result.model,
            relation=relation,
            tail=tail_entity,
            triples_factory=result.training
        ).df
        all_predictions.append(predicted_heads)

    all_predictions_df = pd.concat(all_predictions, ignore_index=True)
    aggregated_predictions = all_predictions_df.groupby('head_label').agg(
        total_score=('score', 'sum')
    ).reset_index()

    final_predictions_sorted = aggregated_predictions.sort_values(by="total_score", ascending=False)

    # Calculate Precision and Recall using the same Top N
    top_n_scores = final_predictions_sorted["head_label"].head(mean_count)
    matching_recipes = filtered_recipes[filtered_recipes['Name'].isin(top_n_scores)].drop_duplicates(subset='Name')

    # Calculate sets of predicted and relevant recipes
    predicted_recipes = set(top_n_scores)
    relevant_recipes = set(filtered_recipes['Name'])

    # Calculate Precision and Recall using the same Top N
    top_n_scores_accuracy = final_predictions_sorted["head_label"].head(expected_criteria_match_number)
    matching_recipes_accuracy = filtered_recipes[filtered_recipes['Name'].isin(top_n_scores_accuracy)]

    true_positives = predicted_recipes.intersection(relevant_recipes)
    precision = len(true_positives) / len(predicted_recipes) if len(predicted_recipes) > 0 else 0
    # Recall Calculation
    Recall = len(true_positives) / len(relevant_recipes) if len(relevant_recipes) > 0 else 0

    Accuracy = len(matching_recipes_accuracy) / expected_criteria_match_number if expected_criteria_match_number > 0 else 0


    # Precision Calculation
    predicted_recipes = set(top_n_scores)
    relevant_recipes = set(filtered_recipes['Name'])


    # F1 Score Calculation
    if precision + Recall > 0:
        F1 = 2 * (precision * Recall) / (precision + Recall)
    else:
        F1 = 0

    # Store results
    result_dict = {
        "Scenario": idx + 1,
        "Criteria": scenario,
        "Expected Matches For Recall": expected_criteria_match_number,
        "Predicted Matches (Recall)": len(matching_recipes),
        "Recall": Recall,
        "Expected Matches For Precision": mean_count,
        "Predicted Matches (Precision)": len(matching_recipes),
        "Precision": precision,
        "F1 Score": F1,  # Add F1 Score to results,
        "Expected Matched For Accuracy":expected_criteria_match_number,
        "Predicted Matched (Accuracy)":len(matching_recipes_accuracy),
        "Accuracy":Accuracy
    }
    
    test_results.append(result_dict)

# Convert results into DataFrame
results_df = pd.DataFrame(test_results)

results_df.to_csv('/app/QuatEEvulationResults/five_criteria_QuatE.csv',index=False)

from itertools import combinations, product
import random

# Set the random seed for reproducibility
random.seed(42)

# Extract the top 20 most common ingredients from 'best_foodentityname'
ingredient_counts = recipes_df['best_foodentityname'].apply(lambda x: [ingredient.strip() for ingredient in x.split(',')]).explode().value_counts()
top_ingredients = ingredient_counts.head(20).index.tolist()

# Define relation options based on the columns in your dataset
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


# Generate binary combinations (pairs of criteria)
three_criteria = generate_specific_combinations(relation_options, combination_size=6)

# List of 5000 random samples from the generated three_criteria list
random_6_user_criteria = random.choices(three_criteria, k=50000)

print(random_6_user_criteria)

# Display the first 20 combinations
random_6_user_criteria[0:20]


test_scenarios = random_6_user_criteria

# Sonucu hesapla
average_occurrence = calculate_average_occurrence(test_scenarios, recipes_df)
average_occurrence = int(average_occurrence)
# Sonucu yazdırma
print("\nAverage Occurrence for Scenarios:", average_occurrence)

# Initialize a list to store test results
test_results = []

# Iterate through each test scenario
for idx, scenario in enumerate(test_scenarios):
    
    filtered_recipes = recipes_df

    for tail_entity, relation in scenario:
        if relation == "hasDietType":
            filtered_recipes = filtered_recipes[filtered_recipes['Diet_Types'].str.contains(tail_entity, case=False, na=False)]
        elif relation == "isForMealType":
            filtered_recipes = filtered_recipes[filtered_recipes['meal_type'].str.contains(tail_entity, case=False, na=False)]
        elif relation == "isFromRegion":
            filtered_recipes = filtered_recipes[filtered_recipes['CleanedRegion'].str.contains(tail_entity, case=False, na=False)]
        elif relation == "needTimeToCook":
            filtered_recipes = filtered_recipes[filtered_recipes['cook_time'].str.contains(tail_entity, case=False, na=False)]
        elif relation in ["HasCarbLevel", "HasProteinLevel", "HasFatLevel", "HasFiberLevel", "HasSodiumLevel", "HasSugarLevel", "HasCholesterolLevel", "HasCalorieLevel"]:
            filtered_recipes = filtered_recipes[filtered_recipes['Healthy_Type'].str.contains(tail_entity, case=False, na=False)]

    expected_criteria_match_number = len(filtered_recipes)
    # Skip the scenario if there are no matches
    if expected_criteria_match_number == 0:
        #print(f"Skipping Scenario {idx+1} as no relevant matches were found.")
        continue
    mean_count = average_occurrence

    # Perform prediction using the model
    all_predictions = []
    
    for tail_entity, relation in scenario:
        predicted_heads = predict_target(
            model=result.model,
            relation=relation,
            tail=tail_entity,
            triples_factory=result.training
        ).df
        all_predictions.append(predicted_heads)

    all_predictions_df = pd.concat(all_predictions, ignore_index=True)
    aggregated_predictions = all_predictions_df.groupby('head_label').agg(
        total_score=('score', 'sum')
    ).reset_index()

    final_predictions_sorted = aggregated_predictions.sort_values(by="total_score", ascending=False)

    # Calculate Precision and Recall using the same Top N
    top_n_scores = final_predictions_sorted["head_label"].head(mean_count)
    matching_recipes = filtered_recipes[filtered_recipes['Name'].isin(top_n_scores)].drop_duplicates(subset='Name')

    # Calculate sets of predicted and relevant recipes
    predicted_recipes = set(top_n_scores)
    relevant_recipes = set(filtered_recipes['Name'])

    # Calculate Precision and Recall using the same Top N
    top_n_scores_accuracy = final_predictions_sorted["head_label"].head(expected_criteria_match_number)
    matching_recipes_accuracy = filtered_recipes[filtered_recipes['Name'].isin(top_n_scores_accuracy)]

    true_positives = predicted_recipes.intersection(relevant_recipes)
    precision = len(true_positives) / len(predicted_recipes) if len(predicted_recipes) > 0 else 0
    # Recall Calculation
    Recall = len(true_positives) / len(relevant_recipes) if len(relevant_recipes) > 0 else 0

    Accuracy = len(matching_recipes_accuracy) / expected_criteria_match_number if expected_criteria_match_number > 0 else 0


    # Precision Calculation
    predicted_recipes = set(top_n_scores)
    relevant_recipes = set(filtered_recipes['Name'])


    # F1 Score Calculation
    if precision + Recall > 0:
        F1 = 2 * (precision * Recall) / (precision + Recall)
    else:
        F1 = 0

    # Store results
    result_dict = {
        "Scenario": idx + 1,
        "Criteria": scenario,
        "Expected Matches For Recall": expected_criteria_match_number,
        "Predicted Matches (Recall)": len(matching_recipes),
        "Recall": Recall,
        "Expected Matches For Precision": mean_count,
        "Predicted Matches (Precision)": len(matching_recipes),
        "Precision": precision,
        "F1 Score": F1,  # Add F1 Score to results,
        "Expected Matched For Accuracy":expected_criteria_match_number,
        "Predicted Matched (Accuracy)":len(matching_recipes_accuracy),
        "Accuracy":Accuracy
    }
    
    test_results.append(result_dict)

# Convert results into DataFrame
results_df = pd.DataFrame(test_results)

results_df.to_csv('/app/QuatEEvulationResults/six_criteria_QuatE.csv',index=False)

from itertools import combinations, product
import random

# Set the random seed for reproducibility
random.seed(42)

# Extract the top 20 most common ingredients from 'best_foodentityname'
ingredient_counts = recipes_df['best_foodentityname'].apply(lambda x: [ingredient.strip() for ingredient in x.split(',')]).explode().value_counts()
top_ingredients = ingredient_counts.head(20).index.tolist()

# Define relation options based on the columns in your dataset
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


# Generate binary combinations (pairs of criteria)
three_criteria = generate_specific_combinations(relation_options, combination_size=7)

# List of 5000 random samples from the generated three_criteria list
random_7_user_criteria = random.choices(three_criteria, k=50000)

print(random_7_user_criteria)

# Display the first 20 combinations
random_7_user_criteria[0:20]



test_scenarios = random_7_user_criteria

# Sonucu hesapla
average_occurrence = calculate_average_occurrence(test_scenarios, recipes_df)
average_occurrence = int(average_occurrence)
# Sonucu yazdırma
print("\nAverage Occurrence for Scenarios:", average_occurrence)

# Initialize a list to store test results
test_results = []

# Iterate through each test scenario
for idx, scenario in enumerate(test_scenarios):
    
    filtered_recipes = recipes_df

    for tail_entity, relation in scenario:
        if relation == "hasDietType":
            filtered_recipes = filtered_recipes[filtered_recipes['Diet_Types'].str.contains(tail_entity, case=False, na=False)]
        elif relation == "isForMealType":
            filtered_recipes = filtered_recipes[filtered_recipes['meal_type'].str.contains(tail_entity, case=False, na=False)]
        elif relation == "isFromRegion":
            filtered_recipes = filtered_recipes[filtered_recipes['CleanedRegion'].str.contains(tail_entity, case=False, na=False)]
        elif relation == "needTimeToCook":
            filtered_recipes = filtered_recipes[filtered_recipes['cook_time'].str.contains(tail_entity, case=False, na=False)]
        elif relation in ["HasCarbLevel", "HasProteinLevel", "HasFatLevel", "HasFiberLevel", "HasSodiumLevel", "HasSugarLevel", "HasCholesterolLevel", "HasCalorieLevel"]:
            filtered_recipes = filtered_recipes[filtered_recipes['Healthy_Type'].str.contains(tail_entity, case=False, na=False)]

    expected_criteria_match_number = len(filtered_recipes)
    # Skip the scenario if there are no matches
    if expected_criteria_match_number == 0:
        #print(f"Skipping Scenario {idx+1} as no relevant matches were found.")
        continue
    mean_count = average_occurrence

    # Perform prediction using the model
    all_predictions = []
    
    for tail_entity, relation in scenario:
        predicted_heads = predict_target(
            model=result.model,
            relation=relation,
            tail=tail_entity,
            triples_factory=result.training
        ).df
        all_predictions.append(predicted_heads)

    all_predictions_df = pd.concat(all_predictions, ignore_index=True)
    aggregated_predictions = all_predictions_df.groupby('head_label').agg(
        total_score=('score', 'sum')
    ).reset_index()

    final_predictions_sorted = aggregated_predictions.sort_values(by="total_score", ascending=False)

    # Calculate Precision and Recall using the same Top N
    top_n_scores = final_predictions_sorted["head_label"].head(mean_count)
    matching_recipes = filtered_recipes[filtered_recipes['Name'].isin(top_n_scores)].drop_duplicates(subset='Name')

    # Calculate sets of predicted and relevant recipes
    predicted_recipes = set(top_n_scores)
    relevant_recipes = set(filtered_recipes['Name'])

    # Calculate Precision and Recall using the same Top N
    top_n_scores_accuracy = final_predictions_sorted["head_label"].head(expected_criteria_match_number)
    matching_recipes_accuracy = filtered_recipes[filtered_recipes['Name'].isin(top_n_scores_accuracy)]

    true_positives = predicted_recipes.intersection(relevant_recipes)
    precision = len(true_positives) / len(predicted_recipes) if len(predicted_recipes) > 0 else 0
    # Recall Calculation
    Recall = len(true_positives) / len(relevant_recipes) if len(relevant_recipes) > 0 else 0

    Accuracy = len(matching_recipes_accuracy) / expected_criteria_match_number if expected_criteria_match_number > 0 else 0


    # Precision Calculation
    predicted_recipes = set(top_n_scores)
    relevant_recipes = set(filtered_recipes['Name'])


    # F1 Score Calculation
    if precision + Recall > 0:
        F1 = 2 * (precision * Recall) / (precision + Recall)
    else:
        F1 = 0

    # Store results
    result_dict = {
        "Scenario": idx + 1,
        "Criteria": scenario,
        "Expected Matches For Recall": expected_criteria_match_number,
        "Predicted Matches (Recall)": len(matching_recipes),
        "Recall": Recall,
        "Expected Matches For Precision": mean_count,
        "Predicted Matches (Precision)": len(matching_recipes),
        "Precision": precision,
        "F1 Score": F1,  # Add F1 Score to results,
        "Expected Matched For Accuracy":expected_criteria_match_number,
        "Predicted Matched (Accuracy)":len(matching_recipes_accuracy),
        "Accuracy":Accuracy
    }
    
    test_results.append(result_dict)

# Convert results into DataFrame
results_df = pd.DataFrame(test_results)


results_df.to_csv('/app/QuatEEvulationResults/seven_criteria_QuatE.csv',index=False)

from itertools import combinations, product
import random

# Set the random seed for reproducibility
random.seed(42)

# Extract the top 20 most common ingredients from 'best_foodentityname'
ingredient_counts = recipes_df['best_foodentityname'].apply(lambda x: [ingredient.strip() for ingredient in x.split(',')]).explode().value_counts()
top_ingredients = ingredient_counts.head(20).index.tolist()

# Define relation options based on the columns in your dataset
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


# Generate binary combinations (pairs of criteria)
three_criteria = generate_specific_combinations(relation_options, combination_size=8)

# List of 5000 random samples from the generated three_criteria list
random_8_user_criteria = random.choices(three_criteria, k=50000)

print(random_8_user_criteria)

# Display the first 20 combinations
random_8_user_criteria[0:20]


test_scenarios = random_8_user_criteria

# Sonucu hesapla
average_occurrence = calculate_average_occurrence(test_scenarios, recipes_df)
average_occurrence = int(average_occurrence)
# Sonucu yazdırma
print("\nAverage Occurrence for Scenarios:", average_occurrence)

# Initialize a list to store test results
test_results = []

# Iterate through each test scenario
for idx, scenario in enumerate(test_scenarios):
    
    filtered_recipes = recipes_df

    for tail_entity, relation in scenario:
        if relation == "hasDietType":
            filtered_recipes = filtered_recipes[filtered_recipes['Diet_Types'].str.contains(tail_entity, case=False, na=False)]
        elif relation == "isForMealType":
            filtered_recipes = filtered_recipes[filtered_recipes['meal_type'].str.contains(tail_entity, case=False, na=False)]
        elif relation == "isFromRegion":
            filtered_recipes = filtered_recipes[filtered_recipes['CleanedRegion'].str.contains(tail_entity, case=False, na=False)]
        elif relation == "needTimeToCook":
            filtered_recipes = filtered_recipes[filtered_recipes['cook_time'].str.contains(tail_entity, case=False, na=False)]
        elif relation in ["HasCarbLevel", "HasProteinLevel", "HasFatLevel", "HasFiberLevel", "HasSodiumLevel", "HasSugarLevel", "HasCholesterolLevel", "HasCalorieLevel"]:
            filtered_recipes = filtered_recipes[filtered_recipes['Healthy_Type'].str.contains(tail_entity, case=False, na=False)]

    expected_criteria_match_number = len(filtered_recipes)
    # Skip the scenario if there are no matches
    if expected_criteria_match_number == 0:
        #print(f"Skipping Scenario {idx+1} as no relevant matches were found.")
        continue
    mean_count = average_occurrence

    # Perform prediction using the model
    all_predictions = []
    
    for tail_entity, relation in scenario:
        predicted_heads = predict_target(
            model=result.model,
            relation=relation,
            tail=tail_entity,
            triples_factory=result.training
        ).df
        all_predictions.append(predicted_heads)

    all_predictions_df = pd.concat(all_predictions, ignore_index=True)
    aggregated_predictions = all_predictions_df.groupby('head_label').agg(
        total_score=('score', 'sum')
    ).reset_index()

    final_predictions_sorted = aggregated_predictions.sort_values(by="total_score", ascending=False)

    # Calculate Precision and Recall using the same Top N
    top_n_scores = final_predictions_sorted["head_label"].head(mean_count)
    matching_recipes = filtered_recipes[filtered_recipes['Name'].isin(top_n_scores)].drop_duplicates(subset='Name')

    # Calculate sets of predicted and relevant recipes
    predicted_recipes = set(top_n_scores)
    relevant_recipes = set(filtered_recipes['Name'])

    # Calculate Precision and Recall using the same Top N
    top_n_scores_accuracy = final_predictions_sorted["head_label"].head(expected_criteria_match_number)
    matching_recipes_accuracy = filtered_recipes[filtered_recipes['Name'].isin(top_n_scores_accuracy)]

    true_positives = predicted_recipes.intersection(relevant_recipes)
    precision = len(true_positives) / len(predicted_recipes) if len(predicted_recipes) > 0 else 0
    # Recall Calculation
    Recall = len(true_positives) / len(relevant_recipes) if len(relevant_recipes) > 0 else 0

    Accuracy = len(matching_recipes_accuracy) / expected_criteria_match_number if expected_criteria_match_number > 0 else 0


    # Precision Calculation
    predicted_recipes = set(top_n_scores)
    relevant_recipes = set(filtered_recipes['Name'])


    # F1 Score Calculation
    if precision + Recall > 0:
        F1 = 2 * (precision * Recall) / (precision + Recall)
    else:
        F1 = 0

    # Store results
    result_dict = {
        "Scenario": idx + 1,
        "Criteria": scenario,
        "Expected Matches For Recall": expected_criteria_match_number,
        "Predicted Matches (Recall)": len(matching_recipes),
        "Recall": Recall,
        "Expected Matches For Precision": mean_count,
        "Predicted Matches (Precision)": len(matching_recipes),
        "Precision": precision,
        "F1 Score": F1,  # Add F1 Score to results,
        "Expected Matched For Accuracy":expected_criteria_match_number,
        "Predicted Matched (Accuracy)":len(matching_recipes_accuracy),
        "Accuracy":Accuracy
    }
    
    test_results.append(result_dict)

# Convert results into DataFrame
results_df = pd.DataFrame(test_results)

results_df.to_csv('/app/QuatEEvulationResults/eight_criteria_QuatE.csv',index=False)
