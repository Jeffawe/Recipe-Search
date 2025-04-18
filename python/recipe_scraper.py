import time
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import pandas as pd
import re
import json
from joblib import load
import os
from dotenv import load_dotenv
import logging
import psycopg2
from psycopg2.extras import execute_batch
import signal
import pickle
from datetime import datetime

from app import send_discord_message

# Load environment variables
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


def get_db_connection():
    """Create and return a new database connection."""
    return psycopg2.connect(os.environ["DATABASE_URL"])


def prepare_features(features):
    # Create a DataFrame with a single row
    df = pd.DataFrame(features, index=[0])
    prepared_features = df.drop(['url', 'title', 'keywords', 'main_image_url', 'abandon_value'], axis=1)
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
    return value[0] == 1


class RecipeCrawler:
    def __init__(self, batch_size=100, checkpoint_interval=1000):
        # Cooking and measurement terms
        self.cooking_verbs = {'bake', 'boil', 'broil', 'chop', 'cook', 'dice', 'fry', 'grate', 'grill', 'mince', 'mix',
                              'peel', 'roast', 'simmer', 'slice', 'stir', 'whisk'}

        self.measurement_terms = {'cup', 'tablespoon', 'teaspoon', 'gram', 'ounce', 'pound', 'ml', 'g', 'kg', 'oz',
                                  'lb', 'pinch', 'dash'}

        self.nutrition_terms = {'calories', 'protein', 'fat', 'carbohydrates', 'fiber', 'sugar', 'sodium'}

        self.headers = {
            'User-Agent': 'Recipe-Collector-Bot/1.0 (Educational Purpose)'
        }

        # Batch processing parameters
        self.batch_size = batch_size
        self.checkpoint_interval = checkpoint_interval

        # State tracking
        self.visited_urls = set()
        self.feature_batch = []
        self.batch_counter = 0
        self.total_processed = 0

        # For clean shutdown
        self.running = True
        signal.signal(signal.SIGINT, self.handle_shutdown)
        signal.signal(signal.SIGTERM, self.handle_shutdown)

        # Checkpoint file
        self.checkpoint_file = 'crawler_checkpoint.pkl'

        # Import TF-IDF later to save memory when not needed
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.tfidf = TfidfVectorizer(
            stop_words='english',
            ngram_range=(1, 2),
            max_features=5000
        )

    def handle_shutdown(self, sig, frame):
        """Handle graceful shutdown with CTRL+C"""
        logger.info("Shutdown signal received, finishing current batch...")
        self.running = False
        self.save_checkpoint()

    def clean_text(self, text):
        """Clean and normalize text."""
        if not text:
            return ""
        text = re.sub(r'[^\w\s]', ' ', text.lower())
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
        title_words = set(re.findall(r'\w+', title.lower())) if title else set()
        ingredient_words = set(word.lower() for ing in ingredients
                               for word in re.findall(r'\w+', ing))

        # Combine all keywords
        all_keywords = set(important_terms) | title_words | ingredient_words

        return {
            'title': title or "",
            'keywords': ','.join(list(all_keywords)[:100]),  # Limit keywords to avoid DB issues
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
            'title': title[:255] if title else "",  # Limit title length for DB
            'keywords': self.extract_keywords(soup)['keywords'][:1000] if soup else "",  # Limit keywords length
            'main_image_url': main_image_url[:500] if main_image_url else "",  # Limit URL length
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
            'title_contains_recipe': 1 if title and 'recipe' in title.lower() else 0,
            'meta_description_contains_recipe': 1 if soup.find('meta', attrs={'name': 'description'}) and 'recipe' in
                                                     soup.find('meta', attrs={'name': 'description'})[
                                                         'content'].lower() else 0,
            'category_mentions': len(re.findall(r'dessert|appetizer|main course|breakfast|dinner', text_content)),
            'link_to_text_ratio': len(soup.find_all('a', href=True)) / (len(text_content) + 1),
            'url_is_generic': 1 if re.search(r'/home|/categories|/recipes$', url) else 0,
            'abandon_value': 0  # Default value for abandon_value
        }

        return features

    def crawl_page(self, url, visited_urls, external=False):
        """Crawl a single page and extract features."""
        if external:
            if url in self.visited_urls:
                return None, []

        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            if not external:
                self.visited_urls.add(url)

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
            logger.error(f"Error crawling {url}: {str(e)}")
            return None, []

        return None, []

    def save_batch_to_db(self):
        """Save the current batch of features to the database."""
        if not self.feature_batch:
            return

        try:
            conn = get_db_connection()
            with conn.cursor() as cur:
                # Use execute_batch for better performance
                columns = list(self.feature_batch[0].keys())
                values = [[row[col] for col in columns] for row in self.feature_batch]

                # Create parameterized query
                placeholders = ', '.join(['%s'] * len(columns))
                columns_str = ', '.join(columns)

                # Create the ON CONFLICT portion dynamically
                update_str = ', '.join([f"{col} = EXCLUDED.{col}" for col in columns if col != 'url'])

                query = f"""
                    INSERT INTO recipes ({columns_str}) 
                    VALUES ({placeholders})
                    ON CONFLICT (url) DO UPDATE SET {update_str}
                """

                # Execute batch insert
                execute_batch(cur, query, values, page_size=100)
                conn.commit()

            logger.info(f"Saved batch of {len(self.feature_batch)} recipes to database")
            send_discord_message(f"Saved batch of {len(self.feature_batch)} recipes to database")
            self.feature_batch = []  # Clear the batch after saving
        except Exception as e:
            logger.error(f"Error saving batch to database: {str(e)}")
            send_discord_message(f"Error saving batch to database: {str(e)}")
            # Don't clear the batch to allow for retry
        finally:
            if conn:
                conn.close()

    def save_checkpoint(self):
        """Save crawler state to a checkpoint file."""
        checkpoint = {
            'visited_urls': self.visited_urls,
            'total_processed': self.total_processed,
            'timestamp': datetime.now().isoformat()
        }

        try:
            with open(self.checkpoint_file, 'wb') as f:
                pickle.dump(checkpoint, f)
            logger.info(f"Checkpoint saved: {self.total_processed} URLs processed")

            # Send notification to Discord
            try:
                requests.post(DISCORD_WEBHOOK_URL, json={
                    "content": f"📊 Crawler checkpoint saved: {self.total_processed} URLs processed"
                })
            except Exception as e:
                logger.error(f"Failed to send Discord checkpoint notification: {e}")

        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")

    def load_checkpoint(self):
        """Load crawler state from checkpoint file if it exists."""
        if os.path.exists(self.checkpoint_file):
            try:
                with open(self.checkpoint_file, 'rb') as f:
                    checkpoint = pickle.load(f)
                self.visited_urls = checkpoint['visited_urls']
                self.total_processed = checkpoint['total_processed']
                logger.info(f"Loaded checkpoint: {self.total_processed} URLs previously processed")
                return True
            except Exception as e:
                logger.error(f"Failed to load checkpoint: {e}")
        return False

    def send_progress_update(self, message):
        """Send a progress update to Discord."""
        try:
            requests.post(DISCORD_WEBHOOK_URL, json={"content": message})
            logger.info(f"Progress update sent to Discord: {message}")
        except Exception as e:
            logger.error(f"Failed to send Discord progress update: {e}")

    def get_domain(self, url):
        """Extract domain from URL for domain-based rate limiting."""
        try:
            from urllib.parse import urlparse
            parsed_url = urlparse(url)
            return parsed_url.netloc
        except Exception:
            return url

    def crawl_sites(self, start_urls, visited_urls, train=False, max_pages=1000000, max_depth=10, delay=1):
        """
        Crawl websites from start_urls and find recipe pages.

        Parameters:
        - start_urls: List of URLs to start crawling from
        - visited_urls: URLs already processed in previous runs
        - train: Whether to classify recipes for training
        - max_pages: Maximum number of pages to crawl
        - max_depth: Maximum crawl depth
        - delay: Base delay between requests
        """
        # Load checkpoint if available
        checkpoint_loaded = self.load_checkpoint()

        # Initialize the queue with start URLs
        urls_to_visit = [(url, 0) for url in start_urls if url not in self.visited_urls]  # Start with depth 0

        # Skip URLs we've already visited if checkpoint loaded
        if checkpoint_loaded:
            urls_to_visit = [(url, depth) for url, depth in urls_to_visit if url not in self.visited_urls]

        # For domain-based rate limiting
        domain_last_access = {}

        # For progress tracking
        start_time = time.time()
        last_report_time = start_time

        # Send initial update
        self.send_progress_update(f"🚀 Starting crawler with {len(urls_to_visit)} URLs in queue")

        try:
            while urls_to_visit and len(self.visited_urls) < max_pages and self.running:
                # Get next URL and its depth
                url, depth = urls_to_visit.pop(0)

                # Skip if too deep or already visited
                if depth > max_depth or url in self.visited_urls:
                    continue

                # Domain-based rate limiting to be nice to servers
                current_domain = self.get_domain(url)
                if current_domain in domain_last_access:
                    time_since_last = time.time() - domain_last_access[current_domain]
                    if time_since_last < delay * 2:  # Add extra delay for same domain
                        sleep_time = delay * 2 - time_since_last
                        if sleep_time > 0:
                            time.sleep(sleep_time)

                domain_last_access[current_domain] = time.time()

                # Process URL
                logger.info(f"Visiting: {url} (Depth: {depth}, Queue: {len(urls_to_visit)})")
                features, new_urls = self.crawl_page(url, visited_urls)

                # If we found valid features
                if features:
                    if train:
                        # For training data collection
                        self.feature_batch.append(features)
                    elif is_recipe_site(features, url):
                        # For production - only save recipes
                        self.feature_batch.append(features)
                        logger.info(f"Found recipe: {url}")

                # Mark as visited and add new URLs to queue
                self.visited_urls.add(url)
                self.total_processed += 1

                # Add new URLs to the queue (avoiding already visited ones)
                filtered_urls = [(u, depth + 1) for u in new_urls
                                 if u not in self.visited_urls and
                                 u not in [url for url, _ in urls_to_visit]]
                urls_to_visit.extend(filtered_urls[:100])  # Limit number of URLs added per page

                # Save batch if it reaches batch size
                self.batch_counter += 1
                if len(self.feature_batch) >= self.batch_size:
                    self.save_batch_to_db()

                # Create checkpoint at regular intervals
                if self.total_processed % self.checkpoint_interval == 0:
                    self.save_checkpoint()

                # Report progress at regular intervals
                current_time = time.time()
                if current_time - last_report_time > 300:  # Report every 5 minutes
                    elapsed = current_time - start_time
                    rate = self.total_processed / elapsed if elapsed > 0 else 0
                    estimated_remaining = (max_pages - self.total_processed) / rate if rate > 0 else "unknown"

                    progress_msg = (
                        f"🔄 Progress: {self.total_processed}/{max_pages} pages processed\n"
                        f"⏱️ Rate: {rate:.2f} pages/second\n"
                        f"⏳ Est. remaining time: {estimated_remaining if isinstance(estimated_remaining, str) else f'{estimated_remaining / 3600:.1f} hours'}\n"
                        f"🗂️ Queue size: {len(urls_to_visit)}"
                    )
                    self.send_progress_update(progress_msg)
                    last_report_time = current_time

                # Basic rate limiting
                time.sleep(delay)

        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received, saving progress...")
        except Exception as e:
            logger.error(f"Error during crawling: {str(e)}")
            self.send_progress_update(f"❌ Crawler error: {str(e)}")
        finally:
            # Save any remaining recipes
            if self.feature_batch:
                self.save_batch_to_db()

            # Save final checkpoint
            self.save_checkpoint()

            # Final report
            total_time = time.time() - start_time
            rate = self.total_processed / total_time if total_time > 0 else 0
            final_msg = (
                f"✅ Crawler finished: {self.total_processed} URLs processed\n"
                f"⏱️ Total time: {total_time / 3600:.2f} hours\n"
                f"📊 Average rate: {rate:.2f} pages/second"
            )
            self.send_progress_update(final_msg)

        return self.total_processed