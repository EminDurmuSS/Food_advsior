# Recipe Knowledge Graph and Recommendation System

This project aims to create a Knowledge Graph (KG) containing recipes, train Knowledge Graph Embedding (KGE) models and Text Embedding models on this graph, and perform a detailed **comparative evaluation** of these models using multi-criteria recipe recommendation scenarios. The project also employs advanced Natural Language Processing (NLP) and vector database techniques to link ingredient names to a standard database (USDA) and enriches the recipe data with various diet, health, region, and meal type labels.

## Table of Contents

1.  [Overview](#overview)
2.  [Features](#features)
3.  [Workflow](#workflow)
    *   [Data Preparation and Enrichment](#data-preparation-and-enrichment)
    *   [Ingredient Entity Linking (USDA Entity Linking)](#ingredient-entity-linking-usda-entity-linking)
    *   [Knowledge Graph and Triple Creation](#knowledge-graph-and-triple-creation)
    *   [Model Training](#model-training)
    *   [Multi-Criteria Evaluation](#multi-criteria-evaluation)
4.  [Technologies Used](#technologies-used)
5.  [Setup](#setup)
6.  [Data](#data)
7.  [Usage](#usage)
    *   [Data Cleaning and Labeling (Preprocessing Notebook)](#data-cleaning-and-labeling-preprocessing-notebook)
    *   [Ingredient Data Scraping](#ingredient-data-scraping)
    *   [USDA Ingredient Linking](#usda-ingredient-linking)
    *   [Knowledge Graph and Triple Creation](#knowledge-graph-and-triple-creation-1)
    *   [KGE Model Training (RotatE, QuatE, TuckER)](#kge-model-training-rotate-quate-tucker)
    *   [Text Embedding Model Evaluation](#text-embedding-model-evaluation)
    *   [KGE Models Multi-Criteria Evaluation](#kge-models-multi-criteria-evaluation)
8.  [Experiments and Evaluation Methodology](#experiments-and-evaluation-methodology)
    *   [Motivation: Why Evaluate by Number of Criteria?](#motivation-why-evaluate-by-number-of-criteria)
    *   [Compared Models](#compared-models)
    *   [Scenario Generation](#scenario-generation)
    *   [Ground Truth Determination](#ground-truth-determination)
    *   [Prediction Methods](#prediction-methods)
    *   [Evaluation Metrics and Interpretation](#evaluation-metrics-and-interpretation)
9.  [Results](#results)
10. [License](#license)

## Overview

Recipe platforms often allow users to search for recipes based on specific features. However, users frequently want to combine multiple constraints (e.g., a "vegetarian", "Asian", "under 30 minutes", "low-calorie" dinner). Answering such complex queries effectively is challenging for traditional systems.

This project addresses this challenge through the following steps:
1.  **Data Enrichment:** Takes raw recipe data, completes missing information (ingredients) via web scraping, and derives various labels (diet type, health profile, region, meal type). This involves both rule-based methods and Large Language Models (LLMs).
2.  **Ingredient Standardization:** Implements **Entity Linking** to map differently written ingredient names (e.g., "Tomato", "Tomatoes") to a standard USDA database. This uses OpenAI embeddings, Pinecone vector database, and Gemini LLM, crucial for KG consistency.
3.  **Knowledge Graph Creation:** Builds a Knowledge Graph (KG) representing relationships between recipes, ingredients, diets, cooking times, regions, health labels, and meal types using the enriched and standardized data.
4.  **Model Training and Evaluation:**
    *   Trains **Knowledge Graph Embedding (KGE)** models (RotatE, QuatE, TuckER) on this KG using PyKEEN.
    *   Evaluates **Text Embedding** models (SentenceTransformers) using textual representations of recipes.
    *   Rigorously evaluates model performance on **multi-criteria recommendation scenarios** with an increasing number of criteria (from 1 to 8) and **compares the models against each other**.

The ultimate goal is to develop models capable of providing accurate and diverse recipe recommendations based on complex user preferences and to analyze their performance under varying levels of query difficulty.

## Features

*   **Comprehensive Data Processing:** Web scraping, rule-based labeling (Diet, Health - based on FDA DVs), LLM-based labeling (Region, Meal Type), keyword analysis.
*   **Advanced Ingredient Linking:** Linking raw ingredient names to USDA standards using OpenAI Embeddings, Pinecone, and Gemini LLM. (See: [EntityLinkingIngredient/foodentitylinking.py](https://github.com/EminDurmuSS/Food_advsior/blob/main/EntityLinkingIngredient/foodentitylinking.py))
*   **Knowledge Graph Creation:** A KG capturing rich relationships between recipes and various attributes (ingredients, diet, time, region, health, meal).
*   **KGE & Text Embedding Model Training/Evaluation:** Utilizes PyKEEN (RotatE, QuatE, TuckER) and SentenceTransformers (paraphrase-MiniLM-L6-v2, all-mpnet-base-v2).
*   **Detailed Multi-Criteria Comparative Evaluation:** Tests and compares model performance using random scenarios with 1 to 8 criteria (metrics: Precision@N, Recall@K, F1, Accuracy@K).

## Workflow

The project generally follows these steps:

### Data Preparation and Enrichment
1.  **(Start):** Load raw recipe data (e.g., `fixedScrapedIngredients.csv`).
2.  **(Script 7 - `ingredient_scraping.py`):** Scrape missing `ScrapedIngredients` data from food.com to complete the dataset.
3.  **(Script 6 - `data_cleaning_labeling.ipynb` - [Preprocessing](https://github.com/EminDurmuSS/Food_advsior/tree/main/Preprocessing)):**
    *   **Diet Type Labeling:** Generate `Diet_Types` column (Vegan, Vegetarian, Paleo, Standard) using recipe `Keywords`, lists of non-animal products (`non_vegetarian_keywords`, `animal_products_keywords`), and nutritional/ingredient rules for the Paleo diet.
    *   **Health Type Labeling:** Initially extract health-related tags from `Keywords`. Then, compare recipe nutritional values (`Calories`, `FatContent`, etc.) against FDA Daily Value (DV) recommendations (e.g., based on 78g Fat, 275g Carb, 50g Protein) to derive `Low/Medium/High` labels (Calories, Fat, Carb, Protein, Sodium, Fiber, Sugar, Cholesterol). A nutrient is labeled "Low" if its Nutrient/DV ratio is < 5%, "High" if > 20%, and "Medium" otherwise. Update/create the `Healthy_Type` column.
    *   **Region/Country & Meal Type Labeling (with LLM):** Use recipe name, description, and `Keywords` to prompt an LLM (GPT) to predict/enrich the `RegionCountry` and `meal_type` columns, filling in missing or 'unknown' values.

### Ingredient Entity Linking (USDA Entity Linking)
4.  **(Script 1 - `usda_linking.py` - [EntityLinkingIngredient](https://github.com/EminDurmuSS/Food_advsior/blob/main/EntityLinkingIngredient/foodentitylinking.py)):**
    *   Generate vector embeddings (OpenAI `text-embedding-3-large`) for each ingredient in the USDA database (`Usda_ingredients_id.csv`).
    *   Upload these embeddings to a Pinecone vector database.
    *   Generate an embedding for each raw ingredient from the recipe list (`ingredientList.csv`).
    *   Query Pinecone to find the closest USDA ingredients (top 10-20 candidates) based on cosine similarity.
    *   Use Gemini LLM to determine the best match among the candidates for the raw ingredient name, or decide "no match".
    *   This prevents creating multiple nodes for the same ingredient in the KG, enhancing graph consistency and the quality of learned KGE models.
    *   Integrate the matched standard ingredient names (`best_foodentityname`) into the main recipe dataset (`BalancedRecipe_entity_linking.csv`).

### Knowledge Graph and Triple Creation
5.  **(Script 2 - `kg_creation.py` & other training scripts):** Create triples (Head, Relation, Tail) from the processed, labeled, and standardized recipe dataset (`BalancedRecipe_entity_linking.csv`). (`Head`: recipe name, `Relation`: contains, hasDietType, isForMealType, needTimeToCook, isFromRegion, HasProteinLevel, etc., `Tail`: ingredient, diet label, meal label, etc.). Save triples to `recipes_triples_*.csv` files.

### Model Training
6.  **(Scripts 3, 4, 5 - `kge_training_*.py`):** Train KGE models (RotatE, QuatE, TuckER) using PyKEEN on the generated triples. Log standard link prediction metrics (Hits@k, MRR) during training.

### Multi-Criteria Evaluation
7.  **(Scripts 3, 4 - `kge_training_*.py` & Script 8 - `text_embedding_eval.py`):**
    *   Generate random recommendation scenarios with varying numbers of criteria (1 to 8).
    *   Determine the ground truth recipes for each scenario.
    *   Predict and rank recipes for each scenario using both the trained KGE models and Text Embedding models.
    *   Compare predictions against the ground truth to calculate Precision@N, Recall@K, F1 Score, and Accuracy@K.
    *   Save the results to CSV files for comparative analysis of the models.

## Technologies Used

*   **Programming Language:** Python 3
*   **Data Processing:** Pandas, NumPy
*   **Natural Language Processing (NLP):** OpenAI API (GPT-3.5, GPT-4, Embeddings), Google Gemini API, spaCy, Sentence-Transformers (paraphrase-MiniLM-L6-v2, all-mpnet-base-v2)
*   **Vector Database:** Pinecone
*   **Knowledge Graph Embedding (KGE):** PyKEEN (RotatE, QuatE, TuckER models)
*   **Graph Processing:** NetworkX
*   **Web Scraping:** Requests, BeautifulSoup4
*   **Other:** tqdm (progress bars), concurrent.futures (parallel processing), logging, os, time, ast, retry, scikit-learn (Clustering), PyTorch (for SentenceTransformers), ipywidgets (for Notebook), Matplotlib (for plotting)

## Setup

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/EminDurmuSS/Food_advsior.git
    cd Food_advsior
    ```
2.  **Install Dependencies:**
    ```bash
    pip install pandas numpy openai pinecone-client google-generativeai tqdm networkx pykeen scikit-learn sentence-transformers requests beautifulsoup4 retry spacy torch ipywidgets matplotlib
    # Download spaCy model if needed
    python -m spacy download en_core_web_lg
    ```
    *(Note: Install any missing libraries with `pip install <library_name>`.)*
3.  **Set API Keys:**
    Define the following environment variables on your system or provide them securely within the scripts:
    *   `OPENAI_API_KEY`: Your OpenAI API key.
    *   `PINECONE_API_KEY`: Your Pinecone API key.
    *   `GEMINI_API_KEY`: Your Google Gemini API key.

## Data

*   **Input Files:**
    *   `EntityLinkingIngredient/Usda_ingredients_id.csv`: USDA ingredient database.
    *   `EntityLinkingIngredient/ingredientList.csv`: Raw ingredient list for linking.
    *   `Preprocessing/fixedScrapedIngredients.csv` (or similar): Raw recipe data.
    *   `BalancedRecipe_entity_linking.csv`: Main processed dataset for KG creation and model training (post-linking and labeling).
    *   `recipes_triples_*.csv`: Knowledge graph triples.
    *   `train_new_kge_model/triples_new_without_ct_ss.csv`: Alternative triples for QuatE training.
*   **Main Output Files:**
    *   `EntityLinkingIngredient/ingredients_mapped.csv`: Ingredients mapped to USDA standards.
    *   `RotatEEvulationResults/`, `QuatEEvulationResults/`, `TuckEREvulationResults/`, `TextEmbeddingEvulationResults/`: CSV files containing multi-criteria evaluation results for the models.
    *   `train_new_kge_model/trained_quate_model_new_without_ct_ss/`: Saved trained QuatE model files.
    *   `recipes_mealtype_nounknown.csv`: Final processed recipe data with updated meal types.
*   **Intermediate Files:** Various intermediate CSV and model files (`recipe_embeddings.pkl`, `tag_*.csv`, etc.) may be generated during the process.

## Usage

The different components of the project can be run as follows:

### Data Cleaning and Labeling (Preprocessing Notebook)

*   **Location:** Jupyter Notebook in the [`Preprocessing/`](https://github.com/EminDurmuSS/Food_advsior/tree/main/Preprocessing) directory (e.g., `data_cleaning_labeling.ipynb`).
*   **Purpose:** Load raw data, derive diet and health labels, enrich region/meal type info using LLMs.
*   **Execution:** Run cells sequentially in a Jupyter Notebook or Google Colab environment.

### Ingredient Data Scraping

*   **Script:** `ingredient_scraping.py` (Corresponds to Code Block 7)
*   **Purpose:** Fetch missing ingredient information from food.com.
*   **Execution:** `python ingredient_scraping.py`

### USDA Ingredient Linking

*   **Script:** `foodentitylinking.py` (in [EntityLinkingIngredient/](https://github.com/EminDurmuSS/Food_advsior/blob/main/EntityLinkingIngredient/foodentitylinking.py))
*   **Purpose:** Link raw ingredient names to USDA standards.
*   **Execution:** `python EntityLinkingIngredient/foodentitylinking.py` (after setting API keys and file paths).

### Knowledge Graph and Triple Creation

*   **Scripts:** `kg_creation.py` (Code Block 2) and relevant parts of training scripts.
*   **Purpose:** Create KG triples from `BalancedRecipe_entity_linking.csv`.
*   **Execution:** `python kg_creation.py` (or executed as part of training scripts).

### KGE Model Training (RotatE, QuatE, TuckER)

*   **Scripts:** `kge_training_evaluation_main.py` (Block 3), `kge_training_tucker.py` (Block 4), `kge_training_quate_new.py` (Block 5).
*   **Purpose:** Train the specified KGE models.
*   **Execution:**
    ```bash
    nohup python kge_training_evaluation_main.py > kge_main_output.log &
    nohup python kge_training_tucker.py > tucker_output.log &
    python kge_training_quate_new.py
    ```

### Text Embedding Model Evaluation

*   **Script:** `text_embedding_eval.py` (Corresponds to Code Blocks 8 & 9).
*   **Purpose:** Evaluate the multi-criteria recommendation performance of SentenceTransformer models.
*   **Execution:** `python text_embedding_eval.py` (after configuring model names and paths).

### KGE Models Multi-Criteria Evaluation

*   **Scripts:** `kge_training_evaluation_main.py` (Block 3), `kge_training_tucker.py` (Block 4).
*   **Purpose:** Evaluate the multi-criteria recommendation performance of trained KGE models.
*   **Execution:** The evaluation step runs automatically after training within the respective scripts, saving results.

## Experiments and Evaluation Methodology

A primary goal of this project is to measure how well different models handle complex, multi-criteria user queries and to **compare their performance**.

### Motivation: Why Evaluate by Number of Criteria?

In real-world scenarios, users often search with multiple constraints. Evaluating models with 1 to 8 criteria helps us understand:

1.  **Realism:** Better simulates how users search for recipes.
2.  **Model Robustness:** Tests how consistently models perform as the search space narrows (query difficulty increases). More criteria demand more specific and accurate recommendations.
3.  **Comparison:** Allows us to see how different model types (KGE vs. Text Embedding) or algorithms within a type (e.g., RotatE vs. TuckER) perform at increasing levels of complexity.

### Compared Models

The multi-criteria recommendation performance of the following models was compared:

*   **Knowledge Graph Embedding (KGE) Models (using PyKEEN):**
    *   RotatE
    *   QuatE
    *   TuckER
*   **Text Embedding Models (using SentenceTransformers):**
    *   `paraphrase-MiniLM-L6-v2`
    *   `all-mpnet-base-v2`

### Scenario Generation

1.  **Relation Options (`relation_options`):** Potential criteria (relations and their possible values) are defined (e.g., `hasDietType: vegetarian`, `isFromRegion: asian`).
2.  **Combination Generation:** `itertools.combinations` selects a specific number of relation types (1 to 8). `itertools.product` then generates all possible value combinations for those relations, creating test scenarios (e.g., `[('vegetarian', 'hasDietType'), ('asian', 'isFromRegion')]`).
3.  **Sampling:** Since the number of possible scenarios grows exponentially with the number of criteria, random sampling (`random.choices` or `random.sample`) is used for scenarios with 3 or more criteria to select a manageable subset (e.g., 25,000 or 50,000 scenarios).

### Ground Truth Determination

For each test scenario, the `BalancedRecipe_entity_linking.csv` dataset is filtered to find recipes that match **all** criteria in the scenario. For example, for `[('vegetarian', 'hasDietType'), ('asian', 'isFromRegion')]`, recipes containing both 'vegetarian' in `Diet_Types` and 'asian' in `CleanedRegion` are identified. This filtered set is considered the "correct" or "relevant" set of recipes for that scenario (Ground Truth).

### Prediction Methods

1.  **KGE Models:**
    *   For each (value, relation) pair in the scenario, PyKEEN's `predict_target` function predicts the most likely recipes (heads) given the `relation` and `tail` (value), returning scores.
    *   If multiple criteria exist, predictions from each are combined. If a recipe is predicted by multiple criteria, its scores are **summed**.
    *   Recipes are ranked in descending order based on their final aggregated scores.
2.  **Text Embedding Models:**
    *   All (value, relation) pairs in the scenario are concatenated into a single text string (e.g., `"hasDietType: vegetarian isFromRegion: asian"`).
    *   This scenario text is embedded using the chosen SentenceTransformer model.
    *   The **cosine similarity** between this scenario embedding and the **pre-computed embeddings** of all recipes is calculated.
    *   Recipes are ranked in descending order based on these similarity scores.

### Evaluation Metrics and Interpretation

Using the ranked prediction list (`Predicted`) and the ground truth set (`Relevant`), the performance of this **multi-criteria recipe retrieval** system is measured and compared using the following metrics:

*   **Precision@N:**
    *   **Calculation:** `|(Predicted[:N] ∩ Relevant)| / N`
    *   **Meaning:** Measures how many of the top `N` recommended recipes are actually relevant to the user's criteria. High Precision@N means the user sees mostly relevant results initially, crucial for a good user experience.
    *   **`N` Value:** Set to the *average* number of relevant recipes found in the dataset for scenarios with that specific number of criteria (`calculate_average_occurrence` result, min 1). This reflects a realistic expectation of how many results the system might show for a query of that difficulty.

*   **Recall@K:**
    *   **Calculation:** `|(Predicted[:K] ∩ Relevant)| / |Relevant|`
    *   **Meaning:** Measures how many of the *total* relevant recipes (matching the criteria) were found within the top `K` recommendations. High Recall@K indicates the system doesn't miss many relevant items.
    *   **`K` Value:** Set to the number of actual relevant recipes for that *specific scenario* (`expected_criteria_match_number` or `|Relevant|`).

*   **F1 Score@N/K:**
    *   **Calculation:** `2 * (Precision@N * Recall@K) / (Precision@N + Recall@K)`
    *   **Meaning:** The harmonic mean of Precision@N and Recall@K, providing a balanced measure of both finding relevant items (Recall) and ensuring the top results are relevant (Precision).

*   **Accuracy@K:**
    *   **Calculation:** `|(Predicted[:K] ∩ Relevant)| / K` (Same formula as Recall@K, K = |Relevant|)
    *   **Meaning:** Shows the proportion of correct predictions within the top `K` spots, where `K` is the number of items that *should* have been found. Answers: "If I ask for `K` results (where `K` is the number of truly relevant items), how many of those `K` results were correct?"

These metrics are calculated for every scenario, and then **averaged** for each specific model and number of criteria (e.g., average Precision for TuckER with 4 criteria) to assess overall performance and enable model comparisons.

## Results

The multi-criteria evaluation results for the different KGE models (RotatE, QuatE, TuckER) and Text Embedding models are presented in detail within the CSV files located in the respective `EvulationResults` folders. These results show the **comparative performance** of the models across **Precision@N, Recall@K, F1 Score, and Accuracy@K** metrics for different numbers of criteria (increasing query complexity). For instance, one can compare the average Precision of the trained **TuckER** model on 4-criteria scenarios against the trained **QuatE** model or a **Text Embedding** model (e.g., all-mpnet-base-v2) on the same scenarios. These comparisons provide insights into which models perform better under different query conditions.

## License

This project is licensed under the [License Name - e.g., MIT] License. See the `LICENSE` file for details.
