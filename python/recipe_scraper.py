import time
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import pandas as pd
import re
import json
from joblib import load
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from dotenv import load_dotenv
import logging

load_dotenv()

DISCORD_WEBHOOK_URL = os.getenv('DISCORD_WEBHOOK_URL')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def prepare_features(features):
    # Create a DataFrame with a single row
    df = pd.DataFrame(features, index=[0])
    prepared_features = df.drop(['url', 'title', 'keywords', 'main_image_url'], axis=1)

    # Return DataFrame with columns in correct order
    return prepared_features


def is_recipe_site(features, url):
    if re.search(r'/recipe(s?)/', url, re.IGNORECASE):
        return True

    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(SCRIPT_DIR, 'trained_pipeline.joblib')
    loaded_pipeline = load(model_path)
    prepared_features = prepare_features(features)
    value = loaded_pipeline.predict(prepared_features)
    if value[0] == 0:
        return False
    else:
        return True


class RecipeCrawler:
    def __init__(self):
        self.cooking_verbs = {'bake', 'boil', 'broil', 'chop', 'cook', 'dice', 'fry', 'grate', 'grill', 'mince', 'mix',
                              'peel', 'roast', 'simmer', 'slice', 'stir', 'whisk'}

        self.measurement_terms = {'cup', 'tablespoon', 'teaspoon', 'gram', 'ounce', 'pound', 'ml', 'g', 'kg', 'oz',
                                  'lb', 'pinch', 'dash'}

        self.nutrition_terms = {'calories', 'protein', 'fat', 'carbohydrates', 'fiber', 'sugar', 'sodium'}

        self.headers = {
            'User-Agent': 'Recipe-Collector-Bot/1.0 (Educational Purpose)'
        }

        self.visited_urls = set()
        self.features_data = []
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

    def extract_keywords(self, soup):
        # Get text content
        text = soup.get_text(separator=' ', strip=True)

        # Get title (more weight to title terms)
        title = ""
        if soup.title:
            title = soup.title.string

        # Extract ingredients if available
        ingredients = []
        ingredient_section = soup.find_all(['ul', 'div'], class_=re.compile(r'ingredient', re.I))
        for section in ingredient_section:
            ingredients.extend(item.get_text() for item in section.find_all('li'))

        # Fit TF-IDF on the content
        tfidf_matrix = self.tfidf.fit_transform([text])
        feature_names = self.tfidf.get_feature_names_out()

        # Get top terms
        important_terms = []
        for idx in tfidf_matrix[0].nonzero()[1]:
            important_terms.append(feature_names[idx])

        # Add title words and ingredients as additional keywords
        title_words = set(re.findall(r'\w+', title.lower()))
        ingredient_words = set(word.lower() for ing in ingredients
                               for word in re.findall(r'\w+', ing))

        # Combine all keywords
        all_keywords = set(important_terms) | title_words | ingredient_words

        return {
            'title': title,
            'keywords': ','.join(all_keywords),
        }

    def extract_features(self, soup, url):
        """Extract relevant features from a webpage."""
        # Get text content
        text_content = soup.get_text(separator=' ', strip=True).lower()

        # Get all text within list items for better ratio calculation
        list_items_text = ' '.join(li.get_text(strip=True) for li in soup.find_all(['li']))

        # Extract title
        title = ""
        if soup.title:
            title = soup.title.string.strip()

        # Find main image URL
        main_image_url = ""
        # Try schema.org recipe image first
        schema = soup.find('script', {'type': 'application/ld+json'})
        if schema:
            try:
                schema_data = json.loads(schema.string)
                if isinstance(schema_data, dict):
                    if schema_data.get('@type') == 'Recipe':
                        main_image_url = schema_data.get('image', '')
                        if isinstance(main_image_url, list):
                            main_image_url = main_image_url[0] if main_image_url else ""
            except Exception as e:
                pass

        # If no schema image, try other methods
        if not main_image_url:
            # Try meta og:image
            og_image = soup.find('meta', {'property': 'og:image'})
            if og_image:
                main_image_url = og_image.get('content', '')

            # If still no image, try to find the largest image in the content
            if not main_image_url:
                images = soup.find_all('img')
                largest_area = 0
                for img in images:
                    width = img.get('width')
                    height = img.get('height')
                    if width and height:
                        try:
                            area = int(width) * int(height)
                            if area > largest_area:
                                largest_area = area
                                main_image_url = img.get('src', '')
                        except:
                            continue

        # Extract features
        features = {
            'url': url,
            'title': title,
            'keywords': self.extract_keywords(soup)['keywords'],
            'main_image_url': main_image_url,
            'cooking_verb_count': sum(text_content.count(verb) for verb in self.cooking_verbs),
            'measurement_term_count': sum(text_content.count(term) for term in self.measurement_terms),
            'nutrition_term_count': sum(text_content.count(term) for term in self.nutrition_terms),
            'number_count': len(re.findall(r'\d+(?:\.\d+)?', text_content)),
            'time_mentions': len(re.findall(r'\d+\s*(?:minute|hour|min|hr)', text_content)),
            'temperature_mentions': len(re.findall(r'\d+\s*(?:degrees?|°|fahrenheit|celsius|f\b|c\b)', text_content)),
            'list_count': len(soup.find_all(['ul', 'ol'])),
            'image_count': len(soup.find_all('img')),
            'total_text_length': len(text_content),
            'has_schema_recipe': 1 if soup.find('script', {'type': 'application/ld+json'}) else 0,
            'recipe_class_indicators': len(re.findall(r'recipe|ingredient|instruction|method|direction',
                                                      str(soup.find_all(['class', 'id'])))),
            'list_text_ratio': len(list_items_text) / (len(text_content) if len(text_content) > 0 else 1),
            'has_print_button': 1 if soup.find('a', text=re.compile(r'print|save', re.I)) else 0,
            'has_servings': 1 if re.search(r'serves?|servings?|yield', text_content) else 0,
            'title_contains_recipe': 1 if 'recipe' in title.lower() else 0,
            'meta_description_contains_recipe': 1 if soup.find('meta', attrs={'name': 'description'}) and 'recipe' in
                                                     soup.find('meta', attrs={'name': 'description'})[
                                                         'content'].lower() else 0,
            'category_mentions': len(re.findall(r'dessert|appetizer|main course|breakfast|dinner', text_content)),
            'link_to_text_ratio': len(soup.find_all('a', href=True)) / (len(text_content) + 1),
            'url_is_generic': 1 if re.search(r'/home|/categories|/recipes$', url) else 0
        }

        return features

    def crawl_page(self, url, visited_urls, external=False):
        """Crawl a single page and extract features."""
        if external:
            if url in self.visited_urls:
                return None, []

        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            if not external: self.visited_urls.add(url)

            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                text_content = soup.get_text(separator=' ', strip=True).lower()

                features = None

                # Get all links and words only if needed
                total_links = len(soup.find_all('a', href=True))
                total_words = len(text_content.split())

                # Avoid crawling pages that are mostly links
                if total_words == 0 or (total_links / total_words) > 0.5:
                    features = None  # Skip pages with too many links
                elif url not in visited_urls:
                    features = self.extract_features(soup, url)

                # Find all links on the page
                links = []
                for link in soup.find_all('a', href=True):
                    next_url = urljoin(url, link['href'])
                    if next_url.startswith('http'):
                        links.append(next_url)

                return features, links

        except Exception as e:
            print(f"Error crawling {url}: {str(e)}")
            return None, []

        return None, []

    def score_recipe_page(self, row):
        """Calculate recipe score based on weighted features."""
        score = 0

        # Strong indicators
        score += row['has_schema_recipe'] * 20
        score += row['title_contains_recipe'] * 15
        score += row['meta_description_contains_recipe'] * 10

        # Content indicators
        score += min(row['measurement_term_count'] * 2, 30)  # Cap at 30 points
        score += min(row['cooking_verb_count'] * 1.5, 20)
        score += min(row['list_count'] * 3, 15)
        score += min(row['image_count'] * 0.5, 10)

        # Ratio indicators
        score += row['list_text_ratio'] * 10  # Higher ratio = more likely recipe

        # Negative indicators
        if row['total_text_length'] > 10000:
            score -= 10  # Very long pages less likely to be recipes

        return score

    def classify_recipe(self, features, threshold=50):
        """Classify a single page based on features."""
        score = self.score_recipe_page(features)
        is_recipe = 1 if score >= threshold else 0
        features['recipe_score'] = score
        features['is_recipe'] = is_recipe
        return features

    def crawl_sites(self, start_urls, visited_urls, train=True, max_pages=200, max_depth=20, delay=2):
        urls_to_visit = [(url, 0) for url in start_urls]  # Start with depth 0

        # For progress tracking
        total_urls = len(start_urls)
        processed_count = 0
        last_reported_percentage = 0

        # Discord reporting function
        def send_progress_update(percent):
            try:
                message = f"🔄 Recipe crawler progress: {percent}% URLs checked)"
                requests.post(DISCORD_WEBHOOK_URL, json={"content": message})
                print(f"Progress update sent to Discord: {percent}%")
            except Exception as e:
                print(f"Failed to send Discord progress update: {e}")

        # Initial update
        send_progress_update(0)

        while urls_to_visit and len(self.visited_urls) < max_pages:
            url, depth = urls_to_visit.pop(0)

            if depth > max_depth or url in self.visited_urls:
                continue

            print(f"Visiting: {url} (Depth: {depth})")
            features, new_urls = self.crawl_page(url, visited_urls)

            if features:
                if train:
                    features = self.classify_recipe(features)
                    self.features_data.append(features)
                elif is_recipe_site(features, url):
                    self.features_data.append(features)

            self.visited_urls.add(url)  # Mark as visited
            urls_to_visit.extend([(new_url, depth + 1) for new_url in new_urls if new_url not in self.visited_urls])

            # Update progress tracking
            processed_count += 1

            # Send Discord update every 20% progress
            if processed_count % 100 == 0:
                send_progress_update(processed_count)

            time.sleep(delay)  # Add delay between requests

        # Final update
        send_progress_update(100)

        return pd.DataFrame(self.features_data).set_index('url', drop=False)
