import requests
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import math


class RecipeMatcher:
    def __init__(self):
        self.tfidf = TfidfVectorizer(
            stop_words='english',
            ngram_range=(1, 2),  # Consider both unigrams and bigrams
            max_features=5000
        )

    def clean_text(self, text):
        """Clean and normalize text."""
        if not text:
            return ""
        # Convert to lowercase and remove special characters
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        # Remove extra whitespace
        text = ' '.join(text.split())
        return text

    def extract_structured_data(self, soup):
        """Extract structured recipe data using common recipe page patterns."""
        data = {
            'title': '',
            'ingredients_list': set(),
            'main_content': '',
            'metadata': ''
        }

        # Try to find recipe title (looking for common recipe page patterns)
        title_candidates = [
            soup.find('h1'),  # Most common location
            soup.find('meta', {'property': 'og:title'}),  # Open Graph
            soup.find('meta', {'name': 'title'}),
            soup.title
        ]

        for candidate in title_candidates:
            if candidate:
                title_text = candidate.get_text() if hasattr(candidate, 'get_text') else candidate.get('content', '')
                if title_text:
                    data['title'] = self.clean_text(title_text)
                    break

        # Find ingredients (multiple common patterns)
        ingredients_sections = []

        # Look for elements with 'ingredient' in class or id
        ingredients_sections.extend(soup.find_all(class_=re.compile(r'ingredient', re.I)))
        ingredients_sections.extend(soup.find_all(id=re.compile(r'ingredient', re.I)))

        # Look for common ingredient list patterns
        ingredients_sections.extend(soup.find_all('ul', class_=re.compile(r'ingredient', re.I)))

        for section in ingredients_sections:
            items = section.find_all(['li', 'p', 'span'])
            for item in items:
                cleaned_text = self.clean_text(item.get_text())
                if cleaned_text:
                    data['ingredients_list'].add(cleaned_text)

        # Get main content
        main_content = []
        main_sections = soup.find_all(['article', 'main', 'div'], class_=re.compile(r'(content|recipe)', re.I))
        for section in main_sections:
            main_content.append(self.clean_text(section.get_text()))
        data['main_content'] = ' '.join(main_content)

        return data

    def calculate_match_score(self, page_data, search_data):
        """Calculate a comprehensive match score between the page and search criteria."""
        scores = {
            'title_match': 0.0,
            'ingredients_match': 0.0,
            'content_match': 0.0
        }

        # Title matching (30% of total score)
        if search_data.get('title') and page_data['title']:
            search_title = self.clean_text(search_data['title'])
            if search_title in page_data['title']:
                scores['title_match'] = 1.0
            else:
                # Use TF-IDF similarity for partial matches
                title_vectors = self.tfidf.fit_transform([search_title, page_data['title']])
                scores['title_match'] = cosine_similarity(title_vectors[0:1], title_vectors[1:2])[0][0]

        # Ingredients matching (40% of total score)
        if search_data.get('ingredients'):
            found_ingredients = 0
            search_ingredients = [self.clean_text(ing) for ing in search_data['ingredients']]

            for search_ing in search_ingredients:
                for page_ing in page_data['ingredients_list']:
                    if search_ing in page_ing:
                        found_ingredients += 1
                        break

            if search_ingredients:
                scores['ingredients_match'] = found_ingredients / len(search_ingredients)

        # Content relevance matching (30% of total score)
        all_search_terms = [
            search_data.get('title', ''),
            *search_data.get('ingredients', [])
        ]
        all_search_terms = ' '.join([self.clean_text(term) for term in all_search_terms if term])

        if all_search_terms and page_data['main_content']:
            content_vectors = self.tfidf.fit_transform([all_search_terms, page_data['main_content']])
            scores['content_match'] = cosine_similarity(content_vectors[0:1], content_vectors[1:2])[0][0]

        # Calculate weighted final score
        final_score = (
                scores['title_match'] * 0.3 +
                scores['ingredients_match'] * 0.4 +
                scores['content_match'] * 0.3
        )

        return final_score, scores

    def process_page(self, url, search_data, threshold=0.4):
        """Process a single page and determine if it matches the search criteria."""
        try:
            response = requests.get(url, timeout=10, headers={'User-Agent': 'Recipe-Bot/1.0'})
            if response.status_code != 200:
                return False, 0.0, {}

            soup = BeautifulSoup(response.text, 'html.parser')
            page_data = self.extract_structured_data(soup)

            final_score, detailed_scores = self.calculate_match_score(page_data, search_data)

            matches = final_score >= threshold
            return matches, final_score, detailed_scores

        except Exception as e:
            print(f"Error processing {url}: {e}")
            return False, 0.0, {}, {}

    def find_matching_recipes(self, recipes, search_data, threshold=0.4):
        """
        Find all matching recipes from a list of recipes with URLs, titles, and keywords.
        Args:
            recipes: List of dictionaries with 'url', 'title', and 'keywords'.
            search_data: Data to match against the recipes.
            threshold: Minimum score for a recipe to be considered a match.
        Returns:
            List of matches with detailed information.
        """
        matches = []
        for recipe in recipes:
            url = recipe['url']

            # Process the page and get match results
            matches_criteria, score, detailed_scores = self.process_page(url, search_data, threshold)

            if matches_criteria:
                matches.append({
                    'url': url,
                    'title': recipe['title'],
                    'imageURL': recipe['main_image_url'],
                    'score': score,
                    'detailed_scores': detailed_scores
                })

        # Sort matches by score in descending order
        matches.sort(key=lambda x: x['score'], reverse=True)
        return matches
    
class HybridRecipeMatcher:
    def __init__(self):
        self.full_matcher = RecipeMatcher()
        self.lightweight_matcher = LightweightMatcher()

    def is_valid(self, value):
        # Check if value is not None, NaN, or an empty string
        if value is None or (isinstance(value, float) and math.isnan(value)) or value == "":
            return False
        return True
    
    def search(self, recipe_features, search_data, lightweight=True, threshold=0.4):
        """Hybrid search using lightweight when possible, full matching when needed"""
        
        # Check if we have keywords for all URLs
        #urls_with_keywords = {feature['url'] for feature in recipe_features if self.is_valid(feature['keywords'])}
        urls_to_check = {feature['url'] for feature in recipe_features if 'url' in feature}

        if lightweight:
            urls_with_keywords = urls_to_check
            urls_needing_full_match = []
        else:
            urls_with_keywords = []
            urls_needing_full_match = urls_to_check
        
        results = []
        
        # Use lightweight matcher for URLs with keywords
        if urls_with_keywords:
            # Extract URLs from recipe_features and check membership
            known_urls = [feature['url'] for feature in recipe_features if feature['url'] in urls_with_keywords]
            filtered_features = [feature for feature in recipe_features if feature['url'] in known_urls]
            
            # Pass the filtered features to the lightweight matcher
            lightweight_results = self.lightweight_matcher.search(filtered_features, search_data)
            results.extend(lightweight_results)
        
        # Use full matcher for URLs without keywords
        if urls_needing_full_match:
            unknown_urls = [feature for feature in recipe_features if feature['url'] in urls_needing_full_match]
            #unknown_urls = list(urls_needing_full_match)
            full_results = self.full_matcher.find_matching_recipes(
                unknown_urls, 
                search_data, 
                threshold
            )
        
            results.extend(full_results)
        
        # Sort combined results by score
        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:20]  # Return top 20 matches
    
class LightweightMatcher:
    def search(self, urls_data, query_data):
        """
        urls_data: List of dictionaries containing url, title, and keywords
        query_data: Dictionary containing 'title' and/or 'ingredients'
        """
        search_terms = set()
        detailed_score = {
            'title_match': 0.0,
            'ingredients_match': 0.0,
            'content_match': 0.0
        }
        
        # Get search terms from title and ingredients
        if 'title' in query_data:
            search_terms.update(re.findall(r'\w+', query_data['title'].lower()))
        if 'ingredients' in query_data:
            for ing in query_data['ingredients']:
                search_terms.update(re.findall(r'\w+', ing.lower()))
        
        # Score each recipe based on keyword matches
        scores = []
        for row in urls_data:
            keywords = set(row['keywords'].split(','))
            if len(search_terms) == 0:
                match_score = 0.0  # or handle this case as needed
            else:
                match_score = len(search_terms & keywords) / len(search_terms)
            scores.append({
                'url': row['url'],
                'title': row['title'],
                'imageURL': row['main_image_url'],
                'score': match_score,
                'detailed_scores': detailed_score
            })
        
        # Return top matches
        matches = sorted(
            [s for s in scores if s['score'] > 0], 
            key=lambda x: x['score'], 
            reverse=True
        )[:20]

        # Return just the URLs if there are matches
        return matches
    
class LightweightMatcher2:
    def __init__(self):
        # Define word importance weights
        self.title_weight = 0.4
        self.ingredients_weight = 0.4
        self.keywords_weight = 0.2
        
    def tokenize(self, text):
        """Normalize and tokenize text, handling common cooking terms."""
        if not isinstance(text, str):
            return set()
        # Convert to lowercase and split into words
        words = set(re.findall(r'\w+', text.lower()))
        return words
    
    def calculate_detailed_scores(self, recipe_data, query_terms, query_ingredients):
        """Calculate detailed matching scores for different aspects."""
        # Title matching
        recipe_title_terms = self.tokenize(recipe_data.get('title', ''))
        title_match = len(recipe_title_terms & query_terms) / max(len(query_terms), 1)
        
        # Ingredients matching
        recipe_keywords = set(recipe_data.get('keywords', '').split(','))
        ingredients_match = len(recipe_keywords & query_ingredients) / max(len(query_ingredients), 1)
        
        # Overall content matching
        content_match = len(recipe_keywords & query_terms) / max(len(query_terms), 1)
        
        return {
            'title_match': round(title_match * 100, 2),
            'ingredients_match': round(ingredients_match * 100, 2),
            'content_match': round(content_match * 100, 2)
        }
    
    def search(self, urls_data, query_data):
        """Enhanced search with weighted scoring and detailed matching."""
        # Extract and process query terms
        query_title_terms = self.tokenize(query_data.get('title', ''))
        query_ingredients = set()
        for ing in query_data.get('ingredients', []):
            query_ingredients.update(self.tokenize(ing))
        
        all_query_terms = query_title_terms | query_ingredients
        
        scores = []
        for recipe in urls_data:
            if not recipe.get('keywords'):
                continue
                
            # Calculate detailed scores
            detailed_scores = self.calculate_detailed_scores(
                recipe, query_title_terms, query_ingredients
            )
            
            # Calculate weighted final score
            final_score = (
                detailed_scores['title_match'] * self.title_weight +
                detailed_scores['ingredients_match'] * self.ingredients_weight +
                detailed_scores['content_match'] * self.keywords_weight
            ) / 100.0
            
            # Only include if there's some match
            if final_score > 0:
                scores.append({
                    'url': recipe['url'],
                    'title': recipe['title'],
                    'score': round(final_score, 3),
                    'detailed_scores': detailed_scores
                })
        
        # Sort and return top matches
        matches = sorted(scores, key=lambda x: x['score'], reverse=True)[:20]
        return matches

    def adjust_weights(self, title_weight=0.4, ingredients_weight=0.4, keywords_weight=0.2):
        """Adjust the importance of different matching criteria."""
        total = title_weight + ingredients_weight + keywords_weight
        self.title_weight = title_weight / total
        self.ingredients_weight = ingredients_weight / total
        self.keywords_weight = keywords_weight / total