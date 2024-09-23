import pandas as pd
df=pd.read_csv('/content/drive/MyDrive/CleanedScrapedIngredientsCookTime2.csv')

import numpy as np

health_keywords = [
    "Low Protein", "High Protein", "Low Cholesterol", "High Cholesterol",
    "Healthy", "Very Low Carbs", "Low Sodium", "High Fiber",
    "Gluten Free", "Sugar Free", "Low Fat", "High Fat", "Low Calorie",
    "High Calorie","Dairy Free","Zero Trans Fat","Low Glycemic","No Added Sugar",
]

# Anahtar kelimeleri analiz ederek sağlıkla ilgili yeni anahtar kelimeler çıkaran fonksiyon
def analyze_and_extract_health_types(keywords_str):
    keywords = keywords_str.split(", ")
    health_types = [keyword for keyword in keywords if any(hk in keyword for hk in health_keywords)]
    return health_types if health_types else np.nan

# Yeni 'Healthy Type' sütununu oluştur
df['Healthy_Type'] = df['Keywords'].apply(analyze_and_extract_health_types)


# DataFrame'i kontrol et
print(df)

print(df['Healthy_Type'].value_counts())

nutrition_columns = [
    'Calories', 'FatContent', 'SaturatedFatContent', 'CholesterolContent',
    'SodiumContent', 'CarbohydrateContent', 'FiberContent', 'SugarContent', 'ProteinContent'
]

# Remove rows where all nutritional values are zero
df = df[~(df[nutrition_columns] == 0).all(axis=1)]



import ast
import pandas as pd

def safe_literal_eval(value):
    try:
        if pd.notna(value) and value.strip():
            return ast.literal_eval(value)
        else:
            return []
    except:
        return []

def categorize_health_types(row):
    existing_labels = safe_literal_eval(row['Healthy_Type'])
    labels = set(existing_labels)


#https://www.fda.gov/food/nutrition-facts-label/how-understand-and-use-nutrition-facts-label#Calories
# Daily Values (DVs) based on a 2,000 calorie diet
DVs = {
    'Fat': 78,  # Total Fat in grams
    'Carbohydrate': 275,  # Total Carbohydrates in grams
    'Protein': 50,  # Protein in grams
    'SaturatedFat': 20,  # Saturated Fat in grams
    'Sodium': 2300,  # Sodium in milligrams
    'DietaryFiber': 28,  # Dietary Fiber in grams
    'AddedSugars': 50,  # Added Sugars in grams
    'Cholesterol': 300  # Cholesterol in milligrams
}

# Function to categorize health types based on nutrient content
def categorize_health_types(row):
    labels = set()

    # Calculate Percent Daily Values (%DV)
    fat_dv = row['FatContent'] / DVs['Fat']
    carb_dv = row['CarbohydrateContent'] / DVs['Carbohydrate']
    protein_dv = row['ProteinContent'] / DVs['Protein']
    saturated_fat_dv = row['SaturatedFatContent'] / DVs['SaturatedFat']
    sodium_dv = row['SodiumContent'] / DVs['Sodium']
    dietary_fiber_dv = row['FiberContent'] / DVs['DietaryFiber']
    added_sugars_dv = row['SugarContent'] / DVs['AddedSugars']
    cholesterol_dv = row['CholesterolContent'] / DVs['Cholesterol']

    # Categorize based on calories
    calories = row['Calories']
    if calories < 200:  # Adjusted for a 2,000 calorie diet
        labels.add('Low Calorie')
    elif calories > 600:  # Adjusted for a 2,000 calorie diet
        labels.add('High Calorie')
    else:
        labels.add('Medium Calorie')

    # Apply nutrient categorizations
    labels.add('Low Fat' if fat_dv < 0.05 else 'High Fat' if fat_dv > 0.20 else 'Medium Fat')
    labels.add('Low Carb' if carb_dv < 0.10 else 'High Carb' if carb_dv > 0.20 else 'Medium Carb')
    labels.add('High Protein' if protein_dv >= 0.20 else 'Low Protein' if protein_dv < 0.10 else 'Medium Protein')
    labels.add('Low Saturated Fat' if saturated_fat_dv < 0.05 else 'High Saturated Fat' if saturated_fat_dv > 0.20 else 'Medium Saturated Fat')
    labels.add('Low Sodium' if sodium_dv < 0.05 else 'High Sodium' if sodium_dv > 0.20 else 'Medium Sodium')
    labels.add('High Fiber' if dietary_fiber_dv > 0.20 else 'Low Fiber' if dietary_fiber_dv < 0.10 else 'Medium Fiber')
    labels.add('High Sugar' if added_sugars_dv > 0.20 else 'Low Sugar' if added_sugars_dv < 0.10 else 'Medium Sugar')
    labels.add('Low Cholesterol' if cholesterol_dv < 0.05 else 'High Cholesterol' if cholesterol_dv > 0.20 else 'Medium Cholesterol')

    # Ensure labels are only added if they are not None
    return list(filter(None, labels))

# Apply the function to fill or update the 'Healthy_Type' column
df['Healthy_Type'] = df.apply(categorize_health_types, axis=1)

