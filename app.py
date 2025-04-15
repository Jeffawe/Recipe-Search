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
from dotenv import load_dotenv
import psycopg2
from psycopg2.extras import RealDictCursor
from math import ceil
from functools import lru_cache
import requests
import subprocess

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

def get_db_connection():
    """Create and return a new database connection."""
    return psycopg2.connect(os.environ["DATABASE_URL"])

# Google Sheets Configuration
SCOPES = ['https://www.googleapis.com/auth/spreadsheets.readonly']
SERVICE_ACCOUNT_FILE = './data/recipesocial.json'
SPREADSHEET_ID = os.getenv('SPREADSHEET_ID')
RANGE_NAME = 'URL!A1:A'

DISCORD_WEBHOOK_URL = os.getenv('DISCORD_WEBHOOK_URL')

def send_discord_message(content):
    try:
        requests.post(DISCORD_WEBHOOK_URL, json={"content": content})
    except Exception as e:
        logger.error(f"Failed to send Discord message: {e}")

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


@app.route('/increment/abandon_value', methods=['POST'])
@error_handler
def increment_abandon_value():
    """Increment the abandon_value by 1 for a specific URL."""
    data = request.get_json()

    if not data or 'url' not in data:
        return jsonify({
            "status": "error",
            "message": "Missing required field: url"
        }), 400

    url = data.get('url')

    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            # First check if the URL exists
            cur.execute("SELECT url, abandon_value FROM recipes WHERE url = %s", (url,))
            existing = cur.fetchone()

            if not existing:
                return jsonify({
                    "status": "error",
                    "message": f"URL not found: {url}"
                }), 404

            # Increment the abandon_value by 1
            cur.execute("""
                UPDATE recipes
                SET abandon_value = COALESCE(abandon_value, 0) + 1
                WHERE url = %s
                RETURNING url, abandon_value
            """, (url,))

            result = cur.fetchone()
            conn.commit()

            return jsonify({
                "status": "success",
                "url": result[0],
                "abandon_value": result[1]
            })

    except Exception as e:
        conn.rollback()
        logger.error(f"Error incrementing abandon_value: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500
    finally:
        conn.close()

@app.route('/create_table', methods=['POST'])
@error_handler
def create_table():
    """Create the table if it does not exist."""
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
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
            """)
            conn.commit()
        return jsonify({"status": "success", "message": "Table created successfully."})
    except Exception as e:
        conn.rollback()
        return jsonify({"status": "error", "message": str(e)}), 500
    finally:
        conn.close()


@app.route('/drop_table', methods=['POST'])
@error_handler
def drop_table():
    """Drop the table if it exists."""
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("DROP TABLE IF EXISTS recipes")
            conn.commit()
            return jsonify({"status": "success", "message": "Table dropped successfully."})
    except Exception as e:
        conn.rollback()
        return jsonify({"status": "error", "message": str(e)}), 500
    finally:
        conn.close()

@app.route('/health', methods=['GET'])
@error_handler
def health_check():
    """Health check endpoint."""
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM recipes")
            recipe_count = cur.fetchone()[0]
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
    finally:
        conn.close()

@app.route('/')
def home():
    return 'Welcome to Recipe Search!'

@lru_cache(maxsize=1)
def get_recipe_features():
    return load_data()

@app.route('/search/recipe', methods=['POST'])
@error_handler
def search():
    """Search endpoint."""
    data = request.get_json()

    if not data or 'search_data' not in data:
        return jsonify({'error': 'search_data field is required'}), 400

    try:
        recipe_features = get_recipe_features()
        matcher = HybridRecipeMatcher()
        threshold_value = data.get('threshold', 0.3)

        if not isinstance(threshold_value, (int, float)) or not 0 <= threshold_value <= 1:
            return jsonify({'error': 'threshold must be a number between 0 and 1'}), 400

        # Get pagination parameters
        page = int(data.get('page', 1))
        limit = int(data.get('limit', 10))

        if page < 1 or limit < 1:
            return jsonify({'error': 'Page and limit must be positive integers'}), 400


        results = matcher.search(recipe_features, data['search_data'], threshold=threshold_value)

        # Pagination logic
        total_results = len(results)
        total_pages = ceil(total_results / limit)
        start_idx = (page - 1) * limit
        end_idx = start_idx + limit
        paginated_results = results[start_idx:end_idx]

        return jsonify({
            "results": paginated_results,
            "metadata": {
                "page": page,
                "limit": limit,
                "total_results": total_results,
                "total_pages": total_pages,
                "threshold": threshold_value,
                "timestamp": datetime.now().isoformat()
            }
        })

    except Exception as e:
        logger.error(f"Search error: {str(e)}")
        raise


@app.route('/search/create', methods=['POST'])
@error_handler
def start_scraper():
    try:
        # Path to the batch manager script
        script_path = os.path.join(os.path.dirname(__file__), 'batch_manager.py')

        # Open files for output redirection
        stdout_file = open('batch_manager_output.log', 'a')
        stderr_file = open('batch_manager_error.log', 'a')

        # Use subprocess.Popen with detachment settings
        process = subprocess.Popen(
            ['python3', script_path, '--mode=run'],  # This will run pending batches
            stdout=stdout_file,
            stderr=stderr_file,
            close_fds=True,
            start_new_session=True  # This is key for detaching
        )

        # Log the process ID for tracking
        app.logger.info(f"Batch manager started with PID: {process.pid}")
        return jsonify({'status': 'Batch manager started in background', 'pid': process.pid}), 202
    except Exception as e:
        app.logger.error(f"Failed to start batch manager: {str(e)}")
        return jsonify({'status': 'error', 'message': f"Failed to start batch manager: {str(e)}"}), 500


@app.route('/search/create_batches', methods=['POST'])
@error_handler
def create_batches():
    try:
        # Path to the batch manager script
        script_path = os.path.join(os.path.dirname(__file__), 'batch_manager.py')

        # Get batch size from request or use default
        batch_size = request.json.get('batch_size', 20)

        # Use subprocess.run for synchronous execution
        result = subprocess.run(
            ['python3', script_path, '--mode=create', f'--batch-size={batch_size}'],
            capture_output=True,
            text=True
        )

        if result.returncode == 0:
            return jsonify({'status': 'Batches created successfully', 'output': result.stdout}), 200
        else:
            return jsonify({'status': 'error', 'message': result.stderr}), 500
    except Exception as e:
        app.logger.error(f"Failed to create batches: {str(e)}")
        return jsonify({'status': 'error', 'message': f"Failed to create batches: {str(e)}"}), 500


@app.route('/search/test_discord', methods=['GET'])
@error_handler
def test_discord():
    try:
        webhook_url = os.getenv('DISCORD_WEBHOOK_URL')
        if not webhook_url:
            return jsonify({'status': 'error', 'message': 'DISCORD_WEBHOOK_URL not configured'}), 500

        response = requests.post(webhook_url, json={"content": "🧪 Testing Discord webhook from web application"})
        return jsonify({
            'status': 'success',
            'discord_response_code': response.status_code,
            'discord_response': response.text
        }), 200
    except Exception as e:
        return jsonify({'status': 'error', 'message': f"Discord test failed: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)