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