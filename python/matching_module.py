import pandas as pd
from recipe_matcher import HybridRecipeMatcher

df = pd.read_csv('./labelledData.csv', na_filter=False, encoding='utf-8', encoding_errors='ignore')
recipe_features = df[['url', 'title', 'keywords']].to_dict('records')

# Initialize the hybrid matcher
matcher = HybridRecipeMatcher()

# Search
search_data = {
    "title": "Chocolate Cake",
    "ingredients": ["Flour", "Chocolate"]
}

# If you want different thresholds
strict_results = matcher.search(recipe_features, search_data, threshold=0.6)
lenient_results = matcher.search(recipe_features, search_data, threshold=0.3)