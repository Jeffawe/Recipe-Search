import shutil
from flask import Flask, request, jsonify
import pandas as pd
from python.recipe_matcher import HybridRecipeMatcher
from python.recipe_scraper import RecipeCrawler
import logging
from functools import wraps
import traceback
from typing import Optional, TypedDict
import os
from datetime import datetime
from platform import system

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

## Define the type structure for our cache
class DataCache(TypedDict):
    data: Optional[pd.DataFrame]
    last_updated: Optional[datetime]

# Initialize the cache with proper typing
df_cache: DataCache = {
    'data': None,
    'last_updated': None
}

def load_data(force_reload: bool = False) -> pd.DataFrame:
    """Load data from CSV with caching mechanism."""
    current_time = datetime.now()
    csv_path = os.path.join('./data', 'labelledData.csv')

    if (df_cache['data'] is None or force_reload or
        (df_cache['last_updated'] and (current_time - df_cache['last_updated']).seconds > 3600)):
        try:
            df_cache['data'] = pd.read_csv(
                csv_path,
                na_filter=False,
                encoding='utf-8',
                encoding_errors='ignore'
            )
            df_cache['last_updated'] = current_time
            logger.info("Data reloaded from CSV")
        except Exception as e:
            logger.error(f"Error loading CSV: {str(e)}")
            raise

    return df_cache['data']

def update_data_file(new_file_path: str, current_file_path: str) -> bool:
    """
    Updates the current data file using symlink on Linux and copy on Windows.
    Returns True if successful, False if failed.
    """
    try:
        # Remove existing file/symlink if it exists
        if os.path.exists(current_file_path):
            if os.path.islink(current_file_path):
                os.unlink(current_file_path)
            else:
                os.remove(current_file_path)

        # Use symlink on Linux, copy on Windows
        if system().lower() == 'windows':
            shutil.copy2(new_file_path, current_file_path)
            logger.info("Windows system: Created file copy")
        else:
            # Create relative symlink from within data directory
            relative_target = os.path.basename(new_file_path)
            os.symlink(relative_target, current_file_path)
            logger.info("Unix system: Created symlink")
        return True

    except Exception as e:
        logger.error(f"Failed to update data file: {e}")
        # Last resort: try to copy if everything else fails
        try:
            shutil.copy2(new_file_path, current_file_path)
            logger.info("Fallback: Created file copy after error")
            return True
        except Exception as copy_error:
            logger.error(f"Failed to create backup copy: {copy_error}")
            return False


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

@app.route('/health', methods=['GET'])
@error_handler
def health_check():
    """Enhanced health check endpoint."""
    try:
        # Test database access
        df = load_data()
        return jsonify({
            "status": "healthy",
            "database_records": len(df),
            "last_updated": df_cache['last_updated'].isoformat() if df_cache['last_updated'] else None
        })
    except Exception as e:
        return jsonify({
            "status": "unhealthy",
            "error": str(e)
        }), 503


@app.route('/search/recipe', methods=['POST'])
@error_handler
def search():
    """Enhanced search endpoint with input validation and error handling."""
    data = request.get_json()

    # Input validation
    if not data:
        return jsonify({'error': 'No JSON data provided'}), 400

    if 'search_data' not in data or not data['search_data']:
        return jsonify({'error': 'search_data field is required'}), 400

    try:
        df = load_data()
        recipe_features = df[['url', 'title', 'keywords']].to_dict('records')

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
        logger.error(f"Search error: {str(e)}\n{traceback.format_exc()}")
        return jsonify({'error': 'Search operation failed', 'message': str(e)}), 500


@app.route('/search/create', methods=['POST'])
@error_handler
def generate_csv():
    """Enhanced CSV generation endpoint with progress tracking."""
    try:
        crawler = RecipeCrawler()
        urls = [
            'https://www.allrecipes.com',
            'https://www.foodnetwork.com/recipes',
            'https://www.simplyrecipes.com',
            'https://damndelicious.net',
            'https://www.beefitswhatsfordinner.com/recipes'
        ]

        # Ensure data directory exists
        data_dir = './data'
        os.makedirs(data_dir, exist_ok=True)

        # Create timestamped filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        new_filename = f'labelledData.csv'
        new_file_path = os.path.join(data_dir, new_filename)
        current_file_path = os.path.join(data_dir, 'labelledData.csv')

        # Crawl and save to new file
        df = crawler.crawl_sites(urls, [], False, max_pages=500)
        df.to_csv(new_file_path, index=False)

        # Update the current data file
        if not update_data_file(new_file_path, current_file_path):
            return jsonify({
                'error': 'Failed to update data file',
                'message': 'Could not create symlink or copy'
            }), 500

        # Force reload of data cache
        load_data(force_reload=True)

        return jsonify({
            "status": "success",
            "filename": new_filename,
            "record_count": len(df),
            "timestamp": datetime.now().isoformat()
        })

    except Exception as e:
        logger.error(f"CSV generation error: {str(e)}\n{traceback.format_exc()}")
        return jsonify({
            'error': 'CSV generation failed',
            'message': str(e)
        }), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)