import pandas as pd
import re
import numpy as np
df = pd.read_csv('/content/drive/MyDrive/recipes.csv')
print(df['Keywords'])
print(df['Keywords'].str.contains('High').sum())
print(df['Keywords'].str.contains('Low').sum())

def extract_cook_time(keywords):
    # Look for patterns like "<> 10 Mins" or "<> 2 Hours"
    match = re.search(r'[<>] \d+ (Mins|Hours)', keywords, re.IGNORECASE)
    if match:
        return match.group(0)
    return np.nan

# Apply the function to the 'Keywords' column
df['cook_time'] = df['Keywords'].apply(extract_cook_time)

import pandas as pd
import openai
import time

# Replace 'your-api-key' with your actual OpenAI API key
openai.api_key = 'your-api-key'

# Function to create prompt for the API
def get_estimated_cook_time(prompt):
    user_prompt = f"I have the following recipe with missing `cook_time` value. Please estimate the `cook_time` based on the recipe description and instructions:\n\n{prompt}\n"

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": user_prompt},
                {"role": "user", "content": "Estimate the cook time based on the recipe."},
            ],
            max_tokens=150,
            stop="\n"
        )

        return response['choices'][0]['message']['content'].strip()

    except Exception as e:
        print(f"Error occurred: {e}")
        return None

# Select rows with null `cook_time` values
missing_cook_time = df[df['cook_time'].isnull()]
print(missing_cook_time)

# Get estimated cook time for each null `cook_time`
for index, row in missing_cook_time.iterrows():
    recipe_name = row['Name']
    recipe_instructions = row['RecipeInstructions']

    # Retry loop with exponential backoff (optional)
    max_retries = 3
    for attempt in range(max_retries):
        estimated_time = get_estimated_cook_time(recipe_instructions)
        if estimated_time:
            df.at[index, 'cook_time'] = estimated_time
            print(f"Estimated cook time for '{recipe_name}': {estimated_time}")
            break  # Exit retry loop on success
        else:
            print(f"Failed to estimate cook time for '{recipe_name}'. Retrying...")
            time.sleep(2 ** attempt)  # Exponential backoff

# Review the results
print("\nUpdated DataFrame:")
print(df)














