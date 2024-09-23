import pandas as pd
df = pd.read_csv('/content/drive/MyDrive/fixedScrapedIngredients.csv')
print(df.head())
import numpy as np

dietType_keywords = [
    "Vegan", "Vegetarian","Paleo",
    "Gluten Free","Dairy Free Foods","Egg Free","Lactose Free"
]

# Anahtar kelimeleri analiz ederek sağlıkla ilgili yeni anahtar kelimeler çıkaran fonksiyon
def analyze_and_extract_health_types(keywords_str):
    keywords = keywords_str.split(", ")
    health_types = [keyword for keyword in keywords if any(hk in keyword for hk in dietType_keywords)]
    return health_types if health_types else np.nan

# Yeni 'Healthy Type' sütununu oluştur
df['Diet_Type'] = df['Keywords'].apply(analyze_and_extract_health_types)
# Define the non_vegetarian_keywords list
non_vegetarian_keywords = [
    'beef', 'pork', 'chicken', 'turkey', 'lamb', 'duck', 'goose', 'bacon',
    'sausage', 'ham', 'veal', 'salami', 'meat', 'fish', 'shrimp', 'crab',
    'lobster', 'clam', 'oyster', 'mussel', 'tuna', 'salmon', 'cod', 'trout',
    'haddock', 'anchovy', 'sardine', 'squid', 'octopus', 'gelatin',
    'lard', 'tallow', 'broth', 'stock', 'bouillon', 'rennet', 'caviar',
    'pepperoni', 'chorizo', 'fish sauce', 'oxtail', 'marrow', 'kidney',
    'liver', 'heart', 'tongue', 'tripe', 'fowl', 'venison', 'rabbit',
    'pheasant', 'partridge', 'quail', 'wild boar', 'game', 'bison', 'buffalo',
    'anchovies', 'escargot', 'pate', 'prosciutto', 'soppressata', 'pastrami',
    'mortadella', 'bologna', 'black pudding', 'blood sausage', 'lamb chops',
    'mutton', 'crayfish', 'grouper', 'herring', 'mackerel', 'monkfish', 'perch',
    'pike', 'pollock', 'prawn', 'snapper', 'swordfish', 'tilapia', 'walleye',
    'yellowtail', 'abalone', 'conch', 'scallop', 'roe', 'eel', 'frog legs',
    'turtle', 'snail', 'sea urchin', 'shark', 'albacore', 'barramundi', 'bluefish',
    'branzino', 'butterfish', 'carp', 'catfish', 'char', 'cobia', 'dogfish',
    'drum', 'flounder', 'garfish', 'gurnard', 'hake', 'hogfish', 'jackfish',
    'john dory', 'kingfish', 'lingcod', 'mullet', 'opah', 'orange roughy', 'pompano',
    'rockfish', 'sculpin', 'smelt', 'snapper', 'sturgeon', 'tarpon', 'tilefish',
    'triggerfish', 'weakfish', 'whitefish', 'whiting', 'wolffish',
    'brisket', 'capicola', 'chorizo', 'corned beef', 'filet mignon', 'foie gras',
    'ground beef', 'ground turkey', 'hot dog', 'jerky', 'lamb shank', 'meatball',
    'meatloaf', 'moose', 'ox', 'pastrami', 'pepperoni', 'pork belly', 'pork chop',
    'pork loin', 'pulled pork', 'ribeye', 'ribs', 'rib roast', 'short ribs', 'sirloin',
    'spam', 'steak', 'strip steak', 't-bone', 'tenderloin', 'venison sausage',
    'wiener', 'abalone', 'albacore tuna', 'barramundi', 'basa', 'beluga', 'bream',
    'calamari', 'canadian bacon', 'caribou', 'caviar', 'ceviche', 'chicken breast',
    'chicken thigh', 'chicken wing', 'chili dog', 'chipolata', 'chorizo', 'clam chowder',
    'cockle', 'confit', 'crab cake', 'crappie', 'crispy pata', 'croaker', 'cuttlefish',
    'deviled ham', 'dorado', 'eel sauce', 'finnan haddie', 'fish ball', 'fish finger',
    'flaked tuna', 'fluke', 'gator', 'gravalax', 'grouper', 'guanciale', 'haggis',
    'halibut', 'head cheese', 'horsemeat', 'jamaican jerk', 'jellyfish', 'kielbasa',
    'king crab', 'kipper', 'knackwurst', 'kobe beef', 'lap cheong', 'lutefisk', 'mahi mahi',
    'manatee', 'marlin', 'mignon', 'muktuk', 'mussels', 'octopus', 'pancetta', 'parma ham',
    'peameal bacon', 'peppered steak', 'pike', 'pollock', 'rabbit stew', 'red snapper',
    'reindeer', 'rock lobster', 'sablefish', 'scampi', 'scungilli', 'sea bass', 'shad',
    'sheep brains', 'sheep liver', 'shish kebab', 'soft shell crab', 'soused herring', 'spareribs',
    'speck', 'sprat', 'squab', 'steak tartare', 'striped bass', 'surimi', 'swai', 'sweetbreads',
    'swordfish', 'tandoori chicken', 'thresher shark', 'tom yum goong', 'tongue', 'tripas',
    'trota', 'turducken', 'turkey bacon', 'unagi', 'wahoo', 'wagyu', 'whelk', 'whitebait',
    'yellowfin tuna','sushi'
]

# Function to check ingredients for non-vegetarian items
def label_vegetarian(ingredients):
    ingredients_lower = ingredients.lower()  # Convert to lower case for matching
    for keyword in non_vegetarian_keywords:
        if keyword in ingredients_lower:
            return ""
    return "Vegetarian"

# Apply the label_vegetarian function to the ScrapedIngredients column
df['Diet_Type2'] = df['ScrapedIngredients'].apply(label_vegetarian)

# Display the first few rows of the new column to verify
print(df['Diet_Type2'].head())

print(df['Diet_Type2'].value_counts())

animal_products_keywords = [
    'beef', 'pork', 'chicken', 'turkey', 'lamb', 'duck', 'goose', 'bacon',
    'sausage', 'ham', 'veal', 'salami', 'meat', 'fish', 'shrimp', 'crab',
    'lobster', 'clam', 'oyster', 'mussel', 'tuna', 'salmon', 'cod', 'trout',
    'haddock', 'anchovy', 'sardine', 'squid', 'octopus', 'gelatin', 'lard',
    'tallow', 'broth', 'stock', 'bouillon', 'rennet', 'caviar', 'pepperoni',
    'chorizo', 'fish sauce', 'oxtail', 'marrow', 'kidney', 'liver', 'heart',
    'tongue', 'tripe', 'fowl', 'venison', 'rabbit', 'pheasant', 'partridge',
    'quail', 'wild boar', 'game', 'bison', 'buffalo', 'milk', 'cheese',
    'yogurt', 'butter', 'cream', 'ice cream', 'ghee', 'casein', 'whey',
    'eggs', 'honey', 'albumin', 'carmine', 'keratin', 'elastin', 'collagen',
    'pancreatin', 'isinglass', 'lanolin', 'anchovies', 'escargot', 'pate',
    'prosciutto', 'soppressata', 'pastrami', 'mortadella', 'bologna',
    'black pudding', 'blood sausage', 'lamb chops', 'mutton', 'crayfish',
    'grouper', 'herring', 'mackerel', 'monkfish', 'perch', 'pike', 'pollock',
    'prawn', 'snapper', 'swordfish', 'tilapia', 'walleye', 'yellowtail',
    'abalone', 'conch', 'scallop', 'roe', 'eel', 'frog legs', 'turtle',
    'snail', 'sea urchin', 'shark', 'albacore', 'barramundi', 'bluefish',
    'branzino', 'butterfish', 'carp', 'catfish', 'char', 'cobia', 'dogfish',
    'drum', 'flounder', 'garfish', 'gurnard', 'hake', 'hogfish', 'jackfish',
    'john dory', 'kingfish', 'lingcod', 'mullet', 'opah', 'orange roughy',
    'pompano', 'rockfish', 'sculpin', 'smelt', 'sturgeon', 'tarpon',
    'tilefish', 'triggerfish', 'weakfish', 'whitefish', 'whiting', 'wolffish',
    'brisket', 'capicola', 'chorizo', 'corned beef', 'filet mignon',
    'foie gras', 'ground beef', 'ground turkey', 'hot dog', 'jerky',
    'lamb shank', 'meatball', 'meatloaf', 'moose', 'ox', 'pastrami',
    'pepperoni', 'pork belly', 'pork chop', 'pork loin', 'pulled pork',
    'ribeye', 'ribs', 'rib roast', 'short ribs', 'sirloin', 'spam', 'steak',
    'strip steak', 't-bone', 'tenderloin', 'venison sausage', 'wiener',
    'abalone', 'albacore tuna', 'barramundi', 'basa', 'beluga', 'bream',
    'calamari', 'canadian bacon', 'caribou', 'ceviche', 'chicken breast',
    'chicken thigh', 'chicken wing', 'chili dog', 'chipolata', 'clam chowder',
    'cockle', 'confit', 'crab cake', 'crappie', 'crispy pata', 'croaker',
    'cuttlefish', 'deviled ham', 'dorado', 'eel sauce', 'finnan haddie',
    'fish ball', 'fish finger', 'flaked tuna', 'fluke', 'gator', 'gravalax',
    'guanciale', 'haggis', 'halibut', 'head cheese', 'horsemeat',
    'jamaican jerk', 'jellyfish', 'kielbasa', 'king crab', 'kipper',
    'knackwurst', 'kobe beef', 'lap cheong', 'lutefisk', 'mahi mahi',
    'manatee', 'marlin', 'mignon', 'muktuk', 'mussels', 'octopus', 'pancetta',
    'parma ham', 'peameal bacon', 'peppered steak', 'pike', 'pollock',
    'rabbit stew', 'red snapper', 'reindeer', 'rock lobster', 'sablefish',
    'scampi', 'scungilli', 'sea bass', 'shad', 'sheep brains', 'sheep liver',
    'shish kebab', 'soft shell crab', 'soused herring', 'spareribs', 'speck',
    'sprat', 'squab', 'steak tartare', 'striped bass', 'surimi', 'swai',
    'sweetbreads', 'tandoori chicken', 'thresher shark', 'tom yum goong',
    'tongue', 'tripas', 'trota', 'turducken', 'turkey bacon', 'unagi', 'wahoo',
    'wagyu', 'whelk', 'whitebait', 'yellowfin tuna', 'sushi', 'alligator skin',
    'alpha-hydroxy acids', 'ambergris', 'amerchol L101', 'amino acids',
    'aminosuccinate acid', 'angora', 'animal fats and oils', 'animal hair',
    'arachidonic acid', 'arachidyl proprionate', 'bee pollen', 'bee products',
    'beeswax', 'biotin', 'blood', 'boar bristles', 'bone char', 'bone meal',
    'calciferol', 'calfskin', 'caprylamine oxide', 'capryl betaine',
    'caprylic acid', 'caprylic triglyceride', 'carbamide', 'carmine',
    'castoreum', 'chitin', 'cholesterol', 'cochineal', 'collagen', 'confectioner’s glaze',
    'cortisone', 'cystine', 'down', 'elastin', 'emulsifiers', 'esters', 'fish gelatin',
    'fish oil', 'fur', 'gel', 'glucosamine', 'glycerides', 'glycerol', 'glycol',
    'guano', 'hide glue', 'hydrolyzed animal protein', 'hydrolyzed silk',
    'hydrolyzed wool', 'hylauronic acid', 'isopropyl myristate', 'keratin',
    'lac', 'lactic acid', 'lactose', 'laneth', 'lanochol', 'leather', 'lecithin',
    'lipoids', 'myristic acid', 'natural bristle brushes', 'natural flavors',
    'nucleic acids', 'oleic acid', 'oleths', 'polypeptides', 'polysorbates',
    'propolis', 'retinol', 'shellac', 'silk', 'sodium tallowate', 'sponges',
    'squalene', 'stearic acid', 'tallow', 'tromethamine', 'tyrosine', 'urea',
    'uric acid', 'vitamin A', 'vitamin D', 'wool', 'wool fat', 'yellow grease'
]

# Function to label recipes based on dietary type
def label_diet_type(ingredients):
    ingredients_lower = ingredients.lower()  # Convert to lower case for matching
    # Check if any animal product is in the ingredients
    for keyword in animal_products_keywords:
        if keyword in ingredients_lower:
            return ""
    return "Vegan"  # If no animal products are found, it's vegan

df['Diet_Type3'] = df['ScrapedIngredients'].apply(label_diet_type)


non_paleo_ingredients = [
    'wheat', 'barley', 'rye', 'oats', 'corn', 'rice', 'bread', 'pasta',
    'cereal', 'cake', 'cookie', 'sugar', 'candy', 'milk', 'cheese', 'butter',
    'yogurt', 'cream', 'ice cream', 'custard', 'dairy', 'legumes', 'peanuts', 'soy',
    'soybean', 'bean', 'lentil', 'pea', 'peanut', 'tofu', 'tempeh', 'edamame',
    'miso', 'vegetable oil', 'canola oil', 'margarine', 'shortening', 'lard',
    'tallow', 'artificial sweetener', 'high-fructose corn syrup', 'malt', 'starch',
    'food starch', 'modified starch', 'hydrogenated oils', 'maltodextrin', 'syrup',
    'agave', 'inulin', 'sorbitol', 'msg', 'gluten', 'bouillon', 'stock cubes',
    'sodium caseinate', 'whey', 'gelatin', 'collagen', 'gums', 'guar gum',
    'xanthan gum', 'carrageenan', 'pectin', 'lecithin', 'soy lecithin', 'flavoring',
    'artificial flavors', 'colors', 'food dyes', 'maltose', 'dextrose', 'fructose',
    'glucose', 'polyols', 'corn syrup', 'vegemite', 'marmite', 'baking powder',
    'baking soda', 'vital wheat gluten', 'potato', 'sweet potato', 'sugar substitute',
    'alcohol', 'processed meat', 'sausages', 'hot dogs', 'soda', 'fruit juice',
    'refined flour', 'cornmeal', 'couscous', 'quinoa', 'millet', 'spelt', 'teff',
    'kamut', 'amaranth', 'buckwheat', 'dextrin', 'instant coffee', 'powdered milk',
    'processed cheese', 'protein powder', 'cornstarch', 'graham cracker', 'pretzel',
    'sherbet', 'toffee', 'white chocolate', 'margarine', 'hydrogenated fat',
    'microwave popcorn', 'rice cakes', 'croutons', 'pita', 'wrap', 'tortilla',
    'worcestershire sauce', 'teriyaki sauce', 'soy sauce', 'miso', 'mustard',
    'pickle', 'ketchup', 'relish', 'mayonnaise', 'barbecue sauce', 'corn chips',
    'potato chips', 'pudding', 'whipped cream', 'canned fruit', 'jam', 'jelly',
    'marshmallow', 'granola bar', 'energy bar'
]

# Function to check if a recipe is suitable for a Paleo diet
def check_paleo(ingredients, nutrition):

    # Check the ingredients
    ingredients_lower = ingredients.lower()
    for item in non_paleo_ingredients:
        if item in ingredients_lower:
            return ""

    total_macros = nutrition['fat'] + nutrition['protein'] + nutrition['carbs']

    # Check if total macros is zero to avoid division by zero
    if total_macros == 0:
        return ""  # or return an appropriate message such as "Incomplete Data"

    # Calculate macro nutrient ratios based on nutritional values
    fat_ratio = nutrition['fat'] / total_macros
    protein_ratio = nutrition['protein'] / total_macros
    carb_ratio = nutrition['carbs'] / total_macros

    # Check Paleo macro nutrient ratio criteria
    if carb_ratio > 0.30:  # If carbohydrate ratio is higher than 30%
        return ""
    if fat_ratio < 0.35 or protein_ratio < 0.20:  # If fat is less than 35% or protein less than 20%
        return ""

    return "Paleo"

# Load the dataset

# Check Paleo suitability for each recipe
df['Paleo_Type'] = df.apply(lambda x: check_paleo(x['ScrapedIngredients'], {'fat': x['FatContent'], 'protein': x['ProteinContent'], 'carbs': x['CarbohydrateContent']}), axis=1)



