import os
import time
import random
import requests
import argparse
import psycopg2
import logging
from dotenv import load_dotenv
from datetime import datetime
import json
from python.recipe_scraper import RecipeCrawler
from typing import List
from psycopg2.extras import RealDictCursor

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('batch_manager.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)
DISCORD_WEBHOOK_URL = os.getenv('DISCORD_WEBHOOK_URL')


def get_db_connection():
    """Create and return a new database connection."""
    return psycopg2.connect(os.environ["DATABASE_URL"])


def get_google_sheets_urls():
    """Fetch URLs from Google Sheets. This is a placeholder - use your existing function."""
    from google.oauth2 import service_account
    from googleapiclient.discovery import build

    SCOPES = ['https://www.googleapis.com/auth/spreadsheets.readonly']
    SERVICE_ACCOUNT_FILE = './data/recipesocial.json'
    SPREADSHEET_ID = os.getenv('SPREADSHEET_ID')
    RANGE_NAME = 'URL!A1:A'

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
    """Send a message to Discord webhook."""
    try:
        requests.post(DISCORD_WEBHOOK_URL, json={"content": content})
        logger.info(f"Discord message sent: {content[:50]}...")
    except Exception as e:
        logger.error(f"Failed to send Discord message: {e}")


def get_visited_urls_from_db():
    """Get all URLs already processed from the database."""
    conn = None
    try:
        conn = get_db_connection()
        with conn.cursor() as cur:
            cur.execute("SELECT url FROM recipes")
            return {row[0] for row in cur.fetchall()}
    except Exception as e:
        logger.error(f"Error fetching visited URLs: {str(e)}")
        return set()
    finally:
        if conn:
            conn.close()


def save_batch_state(batch_id, urls, status="pending"):
    """Save batch state to a JSON file."""
    batch_data = {
        "batch_id": batch_id,
        "urls": urls,
        "status": status,
        "timestamp": datetime.now().isoformat()
    }

    os.makedirs("batches", exist_ok=True)
    with open(f"batches/batch_{batch_id}.json", "w") as f:
        json.dump(batch_data, f)


def load_pending_batches():
    """Load any pending batches from the batches directory."""
    pending_batches = []
    if not os.path.exists("batches"):
        return pending_batches

    for filename in os.listdir("batches"):
        if filename.startswith("batch_") and filename.endswith(".json"):
            try:
                with open(f"batches/{filename}", "r") as f:
                    batch_data = json.load(f)
                    if batch_data["status"] == "pending":
                        pending_batches.append(batch_data)
            except Exception as e:
                logger.error(f"Error loading batch {filename}: {e}")

    return pending_batches

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

def mark_batch_complete(batch_id):
    """Mark a batch as complete."""
    batch_file = f"batches/batch_{batch_id}.json"
    if os.path.exists(batch_file):
        try:
            with open(batch_file, "r") as f:
                batch_data = json.load(f)

            batch_data["status"] = "completed"
            batch_data["completed_at"] = datetime.now().isoformat()

            with open(batch_file, "w") as f:
                json.dump(batch_data, f)

            logger.info(f"Marked batch {batch_id} as completed")
        except Exception as e:
            logger.error(f"Error marking batch {batch_id} as complete: {e}")


def create_batches(start_urls, batch_size=20, shuffle=True):
    """Create batches of URLs for processing."""
    if shuffle:
        # Shuffle to diversify domains in each batch
        random.shuffle(start_urls)

    # Create batches
    return [start_urls[i:i + batch_size] for i in range(0, len(start_urls), batch_size)]


def process_batch(batch_id, urls, max_pages=5000, max_depth=5):
    """Process a single batch of URLs."""
    logger.info(f"Starting batch {batch_id} with {len(urls)} seed URLs")
    send_discord_message(f"🚀 Starting batch {batch_id} with {len(urls)} seed URLs")

    # Get already visited URLs from database
    visited_urls = [recipe['url'] for recipe in load_data()]
    logger.info(f"Found {len(visited_urls)} already visited URLs in database")

    # Create crawler instance with batch-specific settings
    crawler = RecipeCrawler(
        batch_size=100,  # Save to DB every 100 recipes
        checkpoint_interval=500  # Create checkpoint every 500 pages
    )

    # Start crawling
    try:
        processed_count = crawler.crawl_sites(
            start_urls=urls,
            visited_urls=visited_urls,
            train=False,
            max_pages=max_pages,
            max_depth=max_depth,
            delay=1  # Base delay between requests
        )

        logger.info(f"Batch {batch_id} completed: processed {processed_count} URLs")
        send_discord_message(f"✅ Batch {batch_id} completed: processed {processed_count} URLs")

        # Mark batch as complete
        mark_batch_complete(batch_id)

        return processed_count
    except Exception as e:
        logger.error(f"Error processing batch {batch_id}: {e}")
        send_discord_message(f"❌ Error in batch {batch_id}: {str(e)}")
        return 0


def main():
    parser = argparse.ArgumentParser(description="Recipe Crawler Batch Manager")
    parser.add_argument('--mode', choices=['create', 'run'], required=True,
                        help='Create new batches or run pending batches')
    parser.add_argument('--batch-size', type=int, default=20,
                        help='Number of seed URLs per batch (for create mode)')
    parser.add_argument('--batch-id', type=int, help='Specific batch ID to run (for run mode)')
    parser.add_argument('--max-pages', type=int, default=100000,
                        help='Maximum pages to crawl per batch')
    parser.add_argument('--max-depth', type=int, default=5,
                        help='Maximum crawl depth')

    args = parser.parse_args()
    print(args)

    if args.mode == 'create':
        # Create new batches from Google Sheets
        send_discord_message("📋 Creating new batches from Google Sheets...")

        try:
            # Get URLs from Google Sheets
            start_urls = get_google_sheets_urls()
            if not start_urls:
                logger.error("No URLs found in Google Sheet")
                send_discord_message("❌ No URLs found in Google Sheet")
                return

            # Create batches
            url_batches = create_batches(start_urls, args.batch_size, shuffle=True)

            # Save batches
            for i, batch_urls in enumerate(url_batches):
                batch_id = int(time.time()) + i  # Use timestamp + index as batch ID
                save_batch_state(batch_id, batch_urls)
                logger.info(f"Created batch {batch_id} with {len(batch_urls)} URLs")

            send_discord_message(f"✅ Created {len(url_batches)} batches, each with up to {args.batch_size} URLs")

        except Exception as e:
            logger.error(f"Error creating batches: {e}")
            send_discord_message(f"❌ Error creating batches: {str(e)}")

    elif args.mode == 'run':
            # Run pending batches or a specific batch
            if args.batch_id:
                # Run specific batch
                batch_file = f"batches/batch_{args.batch_id}.json"
                if not os.path.exists(batch_file):
                    logger.error(f"Batch {args.batch_id} not found")
                    send_discord_message(f"❌ Batch {args.batch_id} not found")
                    return

                try:
                    with open(batch_file, "r") as f:
                        batch_data = json.load(f)

                    if batch_data["status"] == "completed":
                        logger.info(f"Batch {args.batch_id} is already marked as completed")
                        send_discord_message(f"ℹ️ Batch {args.batch_id} is already marked as completed")
                        return

                    # Process the batch
                    process_batch(
                        batch_id=args.batch_id,
                        urls=batch_data["urls"],
                        max_pages=args.max_pages,
                        max_depth=args.max_depth
                    )

                except Exception as e:
                    logger.error(f"Error running batch {args.batch_id}: {e}")
                    send_discord_message(f"❌ Error running batch {args.batch_id}: {str(e)}")

            else:
                # Run all pending batches
                pending_batches = load_pending_batches()

                if not pending_batches:
                    logger.info("No pending batches found")
                    send_discord_message("ℹ️ No pending batches found. Use --mode=create to create new batches.")
                    return

                logger.info(f"Found {len(pending_batches)} pending batches")
                send_discord_message(f"🔄 Found {len(pending_batches)} pending batches, starting execution...")

                for batch_data in pending_batches:
                    process_batch(
                        batch_id=batch_data["batch_id"],
                        urls=batch_data["urls"],
                        max_pages=args.max_pages,
                        max_depth=args.max_depth
                    )
                    # Add small delay between batches
                    time.sleep(5)

if __name__ == "__main__":
    main()