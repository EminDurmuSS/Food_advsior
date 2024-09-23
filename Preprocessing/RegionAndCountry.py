import openai
import pandas as pd
import os
import time
import numpy as np

# Set your API key securely
openai.api_key = os.getenv('OPENAI_API_KEY', 'Api_key')

# Load your dataset
data = pd.read_csv('/content/drive/MyDrive/UpdatedGeneratedFeaturesRegionCountry.csv')

# Bölgeleri ve ülkeleri kontrol etmek için anahtar kelimeler
regions = ['Asian', 'European', 'African', 'American', 'Middle Eastern', 'Oceanian']
keywords_to_check = [
    'Mexican', 'Asian', 'Indian', 'American', 'Japanese', 'Thai', 'Korean', 'Spanish', 'Chinese', 'Greek', 'Dutch',
    'Brazilian', 'Moroccan', 'Turkish', 'Vietnamese', 'Lebanese', 'German', 'Ethiopian', 'Egyptian', 'Russian',
    'Portuguese', 'Swiss', 'Peruvian', 'Cuban', 'Swedish', 'Malaysian', 'Indonesian', 'Polish', 'Scottish',
    'Australian', 'Chilean', 'Finnish', 'Hungarian', 'Icelandic', 'Nigerian', 'South African', 'Austrian', 'Belgian',
    'Czech', 'Danish', 'Georgian', 'Norwegian', 'Pakistani', 'Nepalese', 'Iraqi', 'Sudanese', 'Somali', 'New Zealand', 'Middle Eastern'
]

# Ülke-bölge eşleştirmesi
country_to_region = {
    'Mexican': 'American', 'Indian': 'Asian', 'Japanese': 'Asian', 'Thai': 'Asian', 'Korean': 'Asian',
    'Spanish': 'European', 'Chinese': 'Asian', 'Greek': 'European', 'Dutch': 'European', 'Brazilian': 'American',
    'Moroccan': 'African', 'Turkish': 'Middle Eastern', 'Vietnamese': 'Asian', 'Lebanese': 'Middle Eastern', 'German': 'European',
    'Ethiopian': 'African', 'Egyptian': 'African', 'Russian': 'European', 'Portuguese': 'European', 'Swiss': 'European',
    'Peruvian': 'American', 'Cuban': 'American', 'Swedish': 'European', 'Malaysian': 'Asian', 'Indonesian': 'Asian',
    'Polish': 'European', 'Scottish': 'European', 'Australian': 'Oceanian', 'Chilean': 'American', 'Finnish': 'European',
    'Hungarian': 'European', 'Icelandic': 'European', 'Nigerian': 'African', 'South African': 'African', 'Austrian': 'European',
    'Belgian': 'European', 'Czech': 'European', 'Danish': 'European', 'Georgian': 'European', 'Norwegian': 'European',
    'Pakistani': 'Asian', 'Nepalese': 'Asian', 'Iraqi': 'Middle Eastern', 'Sudanese': 'African', 'Somali': 'African', 'New Zealand': 'Oceanian',
    'Middle Eastern': 'Middle Eastern'
}

# Fonksiyon tanımlaması
def extract_region_country(keywords):
    found_regions = []
    found_countries = []

    for keyword in keywords_to_check:
        if keyword.lower() in keywords.lower():
            if keyword in regions:
                found_regions.append(keyword)
            else:
                found_countries.append(keyword)

    # Eğer sadece ülke bilgisi varsa, ülkenin bölgesini ekle
    if not found_regions and found_countries:
        for country in found_countries:
            region = country_to_region.get(country)
            if region:
                found_regions.append(region)

    # Bulunan bölgeleri ve ülkeleri birleştir
    if found_regions or found_countries:
        return ", ".join(found_regions + found_countries)

    return np.nan

# 'Keywords' sütununu tarayarak 'region' sütununu oluştur
data['region'] = data['Keywords'].apply(extract_region_country)

# Sonuçları göster
print(data[['Keywords', 'region']])

# List of models to use, cycle through them when rate limited
models = [
    "gpt-3.5-turbo-0125",
    "gpt-3.5-turbo",

]    # Replace with actual model identifiers
current_model = 0  # Index of the current model being used

def create_prompt(row):
    existing_info = row['region_country'] if not pd.isna(row['region_country']) and 'null' not in str(row['region_country']).lower() else "unknown"
    return f"Given the recipe name '{row['Name']}', ingredients: {row['RecipeIngredientParts']}', and a brief description: {row['Description'][:150]}. The current region or country information is {existing_info}. What is the likely country or region of origin? Respond only with the region and country, using the format 'Region, Country'. For example, if the dish is Chinese, answer 'Asia, China'. If the dish is common in multiple countries, list them all. If the dish is globally recognized, simply respond with 'Global Food'."

def predict_region(row):
    global current_model
    prompt = create_prompt(row)
    try:
        response = openai.ChatCompletion.create(
            model=models[current_model],
            messages=[
                {"role": "system", "content": "You are an assistant who provides country or region predictions based on recipe data."},
                {"role": "user", "content": prompt}
            ]
        )
        result = response['choices'][0]['message']['content'].strip()
        print(f"Processed with model {models[current_model]}: {result}")
        return result if result not in ['global food', ''] else None
    except openai.error.RateLimitError as e:
        retry_after = int(e.headers.get('Retry-After', 1))
        print(f"Rate limit reached for model {models[current_model]}, switching to next model after {retry_after} seconds...")
        time.sleep(retry_after)
        current_model = (current_model + 1) % len(models)
        print(f"Switched to model {models[current_model]}")
        return predict_region(row)  # Try again with the new model
    except openai.error.OpenAIError as e:
        print(f"Error processing row with model {models[current_model]}: {e}")
        return None  # Return None for other types of errors

def needs_update(value):
    return pd.isna(value) or 'null' in str(value).lower()

# Process each row conditionally based on needs_update
for index, row in data.iterrows():
    if needs_update(row['region_country']):
        data.at[index, 'region_country'] = predict_region(row)
        print(f"Updated row {index} with new region/country data.")

# Display the updated dataset
print(data[['Name', 'region_country']])
