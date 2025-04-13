import os
from flask import jsonify
from datetime import datetime
from google.oauth2 import service_account
from googleapiclient.discovery import build
import logging
from typing import List
from python.recipe_scraper import RecipeCrawler
from dotenv import load_dotenv
import psycopg2
from psycopg2.extras import RealDictCursor
import requests

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

SCOPES = ['https://www.googleapis.com/auth/spreadsheets.readonly']
SERVICE_ACCOUNT_FILE = './data/recipesocial.json'
SPREADSHEET_ID = os.getenv('SPREADSHEET_ID')
RANGE_NAME = 'URL!A1:A'

DISCORD_WEBHOOK_URL = os.getenv('DISCORD_WEBHOOK_URL')

def get_db_connection():
    """Create and return a new database connection."""
    return psycopg2.connect(os.environ["DATABASE_URL"])

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

def send_discord_message(content):
    try:
        requests.post(DISCORD_WEBHOOK_URL, json={"content": content})
    except Exception as e:
        logger.error(f"Failed to send Discord message: {e}")

def load_data() -> List[dict]:
    """Load data from database."""
    conn = get_db_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT * FROM recipes")
            return cur.fetchall()
    except Exception as e:
        logger.error(f"Error loading from database: {str(e)}")
        raise
    finally:
        conn.close()

def generate_recipes():
    """Generate recipes from URLs in Google Sheet."""
    try:
        send_discord_message("🚀 Recipe generation has started...")
        urls = get_google_sheets_urls()
        if not urls:
            return jsonify({
                'error': 'No URLs found in Google Sheet',
                'message': 'Please add URLs to the specified Google Sheet'
            }), 400

        visited_urls = [recipe['url'] for recipe in load_data()]
        crawler = RecipeCrawler()
        df = crawler.crawl_sites(urls, visited_urls, False, max_pages=1000000)

        for _, row in df.iterrows():
            conn = get_db_connection()
            try:
                with conn.cursor() as cur:
                    cur.execute("""
                        INSERT INTO recipes (                   
                            url, title, keywords, main_image_url, cooking_verb_count,
                            measurement_term_count, nutrition_term_count, number_count,
                            time_mentions, temperature_mentions, list_count, image_count,
                            total_text_length, has_schema_recipe, has_print_button, has_servings,
                            title_contains_recipe, meta_description_contains_recipe,
                            recipe_class_indicators, list_text_ratio, link_to_text_ratio,
                            category_mentions, abandon_value
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (url) DO UPDATE SET
                            title = EXCLUDED.title,
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
                send_discord_message(f"❌ Error during recipe generation: {str(e)}")
                conn.rollback()
            finally:
                conn.close()
                send_discord_message(f"✅ Recipe generation finished successfully at {datetime.now().isoformat()}")

        return jsonify({"status": "success", "timestamp": datetime.now().isoformat()})
    except Exception as e:
        logger.error(f"Recipe generation error: {str(e)}")
        raise

# Add this at the end of your existing file
if __name__ == "__main__":
    generate_recipes()