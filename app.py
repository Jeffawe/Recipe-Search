import os
from flask import Flask, request, jsonify
from datetime import datetime
from google.oauth2 import service_account
from googleapiclient.discovery import build
import logging
from functools import wraps
import traceback
from typing import List
from python.recipe_matcher import HybridRecipeMatcher
from python.recipe_scraper import RecipeCrawler
from dotenv import load_dotenv
import psycopg2
from psycopg2.extras import RealDictCursor

# Load environment variables from .env file
load_dotenv()

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

app = Flask(__name__)

conn = psycopg2.connect(os.environ["DATABASE_URL"])

with conn.cursor() as cur:
    cur.execute("SELECT now()")
    res = cur.fetchall()
    conn.commit()
    print(res)

# Google Sheets Configuration
SCOPES = ['https://www.googleapis.com/auth/spreadsheets.readonly']
SERVICE_ACCOUNT_FILE = './data/recipesocial.json'  # Update this path
SPREADSHEET_ID = os.getenv('SPREADSHEET_ID')
RANGE_NAME = 'URL!A1:A'

def create_tables():
    with conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS recipes (
                url VARCHAR PRIMARY KEY,
                title VARCHAR,
                keywords TEXT,
                main_image_url VARCHAR,
                cooking_verb_count INT,
                measurement_term_count INT,
                nutrition_term_count INT,
                number_count INT,
                time_mentions INT,
                temperature_mentions INT,
                list_count INT,
                image_count INT,
                total_text_length INT,
                has_schema_recipe INT,
                has_print_button INT,
                has_servings INT,
                title_contains_recipe INT,
                meta_description_contains_recipe INT,
                recipe_class_indicators INT,
                category_mentions INT,
                list_text_ratio FLOAT,
                link_to_text_ratio FLOAT,
                abandon_value FLOAT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        conn.commit()

def get_google_sheets_urls() -> List[str]:
    """Fetch URLs from Google Sheets."""
    try:
        credentials = service_account.Credentials.from_service_account_file(
            SERVICE_ACCOUNT_FILE, scopes=SCOPES)

        service = build('sheets', 'v4', credentials=credentials)
        sheet = service.spreadsheets()
        result = sheet.values().get(spreadsheetId=SPREADSHEET_ID,
                                    range=RANGE_NAME).execute()
        values = result.get('values', [])

        if not values:
            logger.warning('No data found in Google Sheet')
            return []

        # Flatten the list and remove empty strings
        urls = [url[0] for url in values if url and url[0].strip()]
        return urls

    except Exception as e:
        logger.error(f"Error fetching URLs from Google Sheets: {str(e)}")
        raise


def load_data() -> List[dict]:
    """Load data from database."""
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        try:
            cur.execute("SELECT * FROM recipes")
            rows = cur.fetchall()
            return rows  # Returns a list of dictionaries, each row is a recipe
        except Exception as e:
            logger.error(f"Error loading from database: {str(e)}")
            raise

def error_handler(f):
    """Decorator for consistent error handling."""

    @wraps(f)
    def wrapper(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in {f.__name__}: {str(e)}\n{traceback.format_exc()}")
            return jsonify({
                'error': 'Internal server error',
                'message': str(e)
            }), 500

    return wrapper


@app.route('/')
def home():
    return 'Welcome to Recipe Search!'

@app.route('/create_table', methods=['POST'])
@error_handler
def create_table():
    """Create the table if it does not exist."""
    try:
        with conn.cursor() as cur:
            create_table_query = """
                CREATE TABLE IF NOT EXISTS recipes (
                    url VARCHAR PRIMARY KEY,
                    title VARCHAR,
                    keywords TEXT,
                    main_image_url VARCHAR,
                    cooking_verb_count INTEGER,
                    measurement_term_count INTEGER,
                    nutrition_term_count INTEGER,
                    number_count INTEGER,
                    time_mentions INTEGER,
                    temperature_mentions INTEGER,
                    list_count INTEGER,
                    image_count INTEGER,
                    total_text_length INTEGER,
                    has_schema_recipe INTEGER,
                    has_print_button INTEGER,
                    has_servings INTEGER,
                    title_contains_recipe INTEGER,
                    meta_description_contains_recipe INTEGER,
                    recipe_class_indicators INTEGER,
                    category_mentions INTEGER,
                    list_text_ratio FLOAT,
                    link_to_text_ratio FLOAT,
                    abandon_value FLOAT,
                    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
                )
            """
            cur.execute(create_table_query)
            conn.commit()
        return jsonify({"status": "success", "message": "Table created successfully."})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/drop_table', methods=['POST'])
@error_handler
def drop_table():
    """Drop the table if it exists."""
    try:
        with conn.cursor() as cur:
            drop_table_query = "DROP TABLE IF EXISTS recipes"
            cur.execute(drop_table_query)
            conn.commit()
        return jsonify({"status": "success", "message": "Table dropped successfully."})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/health', methods=['GET'])
@error_handler
def health_check():
    """Health check endpoint."""
    try:
        # Open a new connection and create a cursor
        with conn.cursor() as cur:
            # Perform a simple query to check if the connection is active
            cur.execute("SELECT COUNT(*) FROM recipes")
            recipe_count = cur.fetchone()[0]  # Fetch the count result

        return jsonify({
            "status": "healthy",
            "database_records": recipe_count,
            "database_connection": "active"
        })

    except Exception as e:
        return jsonify({
            "status": "unhealthy",
            "error": str(e)
        }), 503


@app.route('/search/recipe', methods=['POST'])
@error_handler
def search():
    """Search endpoint."""
    data = request.get_json()

    if not data or 'search_data' not in data:
        return jsonify({'error': 'search_data field is required'}), 400

    try:
        recipe_features = load_data()
        matcher = HybridRecipeMatcher()
        threshold_value = data.get('threshold', 0.3)

        if not isinstance(threshold_value, (int, float)) or not 0 <= threshold_value <= 1:
            return jsonify({'error': 'threshold must be a number between 0 and 1'}), 400

        results = matcher.search(recipe_features, data['search_data'], threshold=threshold_value)

        return jsonify({
            "results": results,
            "metadata": {
                "threshold": threshold_value,
                "total_matches": len(results),
                "timestamp": datetime.now().isoformat()
            }
        })

    except Exception as e:
        logger.error(f"Search error: {str(e)}")
        raise


@app.route('/search/create', methods=['POST'])
@error_handler
def generate_recipes():
    """Generate recipes from URLs in Google Sheet."""
    try:
        # Fetch URLs from Google Sheets
        urls = get_google_sheets_urls()

        if not urls:
            return jsonify({
                'error': 'No URLs found in Google Sheet',
                'message': 'Please add URLs to the specified Google Sheet'
            }), 400

        # Crawl recipes
        visited_urls = [recipe['url'] for recipe in load_data()]
        crawler = RecipeCrawler()
        df = crawler.crawl_sites(urls, visited_urls, False, max_pages=500)

        # Upsert data into database (replace ORM insert with raw SQL)
        for _, row in df.iterrows():
            try:
                with conn.cursor() as cur:
                    cur.execute("""
                               INSERT INTO recipes (url, title, keywords, main_image_url, cooking_verb_count,
                                                    measurement_term_count, nutrition_term_count, number_count,
                                                    time_mentions, temperature_mentions, list_count, image_count,
                                                    total_text_length, has_schema_recipe, has_print_button, has_servings,
                                                    title_contains_recipe, meta_description_contains_recipe,
                                                    recipe_class_indicators, list_text_ratio, link_to_text_ratio,
                                                    category_mentions, abandon_value)
                               VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                               ON CONFLICT (url) 
                               DO UPDATE SET title = EXCLUDED.title,
                                             keywords = EXCLUDED.keywords,
                                             main_image_url = EXCLUDED.main_image_url,
                                             cooking_verb_count = EXCLUDED.cooking_verb_count,
                                             measurement_term_count = EXCLUDED.measurement_term_count,
                                             nutrition_term_count = EXCLUDED.nutrition_term_count,
                                             number_count = EXCLUDED.number_count,
                                             time_mentions = EXCLUDED.time_mentions,
                                             temperature_mentions = EXCLUDED.temperature_mentions,
                                             list_count = EXCLUDED.list_count,
                                             image_count = EXCLUDED.image_count,
                                             total_text_length = EXCLUDED.total_text_length,
                                             has_schema_recipe = EXCLUDED.has_schema_recipe,
                                             has_print_button = EXCLUDED.has_print_button,
                                             has_servings = EXCLUDED.has_servings,
                                             title_contains_recipe = EXCLUDED.title_contains_recipe,
                                             meta_description_contains_recipe = EXCLUDED.meta_description_contains_recipe,
                                             recipe_class_indicators = EXCLUDED.recipe_class_indicators,
                                             list_text_ratio = EXCLUDED.list_text_ratio,
                                             link_to_text_ratio = EXCLUDED.link_to_text_ratio,
                                             category_mentions = EXCLUDED.category_mentions,
                                             abandon_value = EXCLUDED.abandon_value
                           """, tuple(row[col] for col in df.columns))
                conn.commit()

            except Exception as e:
                logger.error(f"Error processing URL {row['url']}: {e}")
                conn.rollback()

        return jsonify({
            "status": "success",
            "timestamp": datetime.now().isoformat()
        })

    except Exception as e:
        logger.error(f"Recipe generation error: {str(e)}")
        raise
    finally:
        logger.debug("Finished")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)