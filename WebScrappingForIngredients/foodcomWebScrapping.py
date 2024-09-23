import pandas as pd
import requests
from bs4 import BeautifulSoup
import urllib.parse
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

def generate_url(name, recipe_id):
    formatted_name = urllib.parse.quote(name.replace(" ", "-").replace("/", "-").lower())
    return f"https://www.food.com/recipe/{formatted_name}-{recipe_id}"

def fetch_ingredients(session, url, max_retries=3, backoff_factor=1.5):
    retries = 0
    while retries < max_retries:
        try:
            response = session.get(url)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                ingredients_list = soup.find('ul', {'class': 'ingredient-list'})
                ingredients = []
                if ingredients_list:
                    for item in ingredients_list.find_all('li', style='display: contents'):
                        quantity_tag = item.find('span', {'class': 'ingredient-quantity'})
                        text_tag = item.find('span', {'class': 'ingredient-text'})
                        quantity = quantity_tag.get_text(strip=False) if quantity_tag else ''
                        text = text_tag.get_text(strip=False).replace("\xa0", " ").replace('\n','') if text_tag else ''
                        if text:  # Only append if text is not empty
                            ingredients.append(f"{quantity} {text}".strip())
                    if ingredients:  # Ensure the list is not empty
                        return (url, ingredients)
                    else:
                        return (url, ['No ingredients found'])
                else:
                    return (url, ['No ingredients found'])
            else:
                response.raise_for_status()  # Will raise HTTPError for bad responses
        except requests.exceptions.HTTPError:
            return (url, ['HTTP error'])
        except Exception:
            if retries == max_retries - 1:
                return (url, ['HTTP error'])  # Skip on final failure
        retries += 1
        time.sleep(backoff_factor ** retries)  # Exponential backoff
    return (url, ['HTTP error'])  # Return error on failures

def process_data(data):
    # Filter out rows where 'ScrapedIngredients' is empty
    data_to_rescrape = data[data['ScrapedIngredients'] == '[]']
    data_to_rescrape['URL'] = data_to_rescrape.apply(lambda row: generate_url(row['Name'], row['RecipeId']), axis=1)
    urls = data_to_rescrape['URL'].tolist()
    results = []

    with requests.Session() as session:
        with ThreadPoolExecutor(max_workers=10) as executor:
            future_to_url = {executor.submit(fetch_ingredients, session, url): url for url in urls}
            for future in tqdm(as_completed(future_to_url), total=len(future_to_url), desc="Scraping Ingredients"):
                url = future_to_url[future]
                try:
                    url, ingredients = future.result()
                    if ingredients and ingredients != ['HTTP error'] and ingredients != ['No ingredients found']:  # Only add results with valid data
                        results.append((url, ingredients))
                except Exception:
                    continue  # Skip processing on unhandled exceptions

    # Mapping results back to dataframe
    url_to_ingredients = dict(results)
    data_to_rescrape['ScrapedIngredients'] = data_to_rescrape['URL'].map(url_to_ingredients).apply(lambda x: x if isinstance(x, list) else [])

    # Update original dataframe with new scraped ingredients
    data.update(data_to_rescrape)
    return data

# Load the dataset
data = df
# Process the data to re-scrape missing ingredients
result_data = process_data(data)

# Display the results
result_data_display = result_data[['Name', 'URL', 'ScrapedIngredients']]
print(result_data_display.head())

# Save the updated dataset
result_data.to_csv('/content/drive/MyDrive/FixedScrappedBugs.csv', index=False)
