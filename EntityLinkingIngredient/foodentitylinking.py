import pandas as pd
import re
import spacy
from rdflib import Graph
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm

# Load the SpaCy model
nlp = spacy.load('en_core_web_lg')

# Preprocessing function
def preprocess_ingredient(ingredient):
    ingredient = ingredient.lower()
    ingredient = re.sub(r'\(.*?\)', '', ingredient)
    ingredient = re.sub(r'[^a-z\s]', '', ingredient)
    doc = nlp(ingredient)
    ingredient = ' '.join([token.lemma_ for token in doc if not token.is_stop])
    ingredient = ingredient.strip()
    return ingredient

# Load DataFrame and select first 300 rows
df = pd.read_csv('FinalRecipes.csv')

# Load the embedding model
embedding_model = SentenceTransformer('multi-qa-mpnet-base-cos-v1')

# Load the FoodOn ontology
g = Graph()
g.parse("foodon.owl", format="xml")

# Function to get food products from the ontology
def get_food_products(graph):
    query = """
    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    PREFIX obo: <http://purl.obolibrary.org/obo/>

    SELECT ?food ?label WHERE {
        ?food rdfs:label ?label .
        ?food rdfs:subClassOf* obo:FOODON_00002403 .
    }
    """
    results = graph.query(query)
    return [(str(result.food), str(result.label)) for result in results]

# Get food products and their labels
food_products = get_food_products(g)
food_uris, food_labels = zip(*food_products)

# Calculate embeddings for food labels
food_embeddings = embedding_model.encode(food_labels, convert_to_tensor=True)

# Function to find the best matching food product
def find_best_food_product(term, top_n=1):
    processed_term = preprocess_ingredient(term)
    term_embedding = embedding_model.encode([processed_term], convert_to_tensor=True)
    cos_sim = util.pytorch_cos_sim(term_embedding, food_embeddings)
    top_matches_idx = cos_sim.topk(top_n).indices[0].tolist()
    matched_labels = [food_labels[idx] for idx in top_matches_idx]
    matched_uris = [food_uris[idx] for idx in top_matches_idx]
    return matched_labels[0], matched_uris[0]

# Function to find the best matching food entities for a list of ingredients with error handling
def find_best_food_entities(ingredients):
    ingredient_list = [ingredient.strip() for ingredient in ingredients.split(',')]
    best_foodentitynames = []
    best_foodentitylinks = []

    for ingredient in tqdm(ingredient_list, desc="Processing Ingredients", leave=False):
        try:
            best_label, best_uri = find_best_food_product(ingredient)
            best_foodentitynames.append(best_label)
            best_foodentitylinks.append(best_uri)
        except Exception as e:
            print(f"Error processing ingredient '{ingredient}': {e}")
            best_foodentitynames.append("Error")
            best_foodentitylinks.append("Error")

    return ', '.join(best_foodentitynames), ', '.join(best_foodentitylinks)

# Apply the function to update the DataFrame with error handling
try:
    df[['best_foodentityname', 'best_foodentitylink']] = df['RecipeIngredientParts'].apply(lambda x: find_best_food_entities(x)).apply(pd.Series)
except Exception as e:
    print(f"Error updating DataFrame: {e}")

# Save the updated DataFrame
df.to_csv('foodentitylinking.csv', index=False)

# Display the results
print(df[['RecipeIngredientParts', 'best_foodentityname', 'best_foodentitylink']])

# Detailed output for each row
for index, row in df[['RecipeIngredientParts', 'best_foodentityname', 'best_foodentitylink']].iterrows():
    print(f"Row {index}:")
    print(f"Ingredients: {row['RecipeIngredientParts']}")
    print(f"Best Food Entities: {row['best_foodentityname']}")
    print(f"Best Food Entity Links: {row['best_foodentitylink']}")
    print("\n" + "-"*50 + "\n")
