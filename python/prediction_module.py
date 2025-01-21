from recipe_scraper import RecipeCrawler

# Initialize crawler
crawler = RecipeCrawler()

# Define starting URLs
urls = [
    'https://www.allrecipes.com',
    'https://www.foodnetwork.com/recipes',
    'https://www.simplyrecipes.com',
    'https://damndelicious.net',
    'https://www.beefitswhatsfordinner.com/recipes'
]

# Crawl and get DataFrame
df = crawler.crawl_sites(urls, [], False, max_pages=500)

# Convert to a csv file
df.to_csv('labelledData.csv', index=False)