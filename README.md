# Recipe-Search

**Recipe-Search** is the backend service for the [Recipe Social](https://recipe-social.vercel.app) website. It is built with **Flask** and integrates a custom trained **machine learning model** and **natural language processing** techniques to search and retrieve recipes from the internet.

This backend is designed to:

- Use an ML model (trained using **XGBoost**) to intelligently scrape recipe content from various websites.
- Apply **NLP (with spaCy)** to enhance the extracted recipe data.
- Store the processed recipes in a **Supabase** database for easy access.
- Allow users to search for certain recipes using Natural Language Processing techniques.

---

## Project Overview

The Flask server provides API endpoints that interact with a trained ML model and NLP pipeline to:

- Scrape recipe information from web pages.
- Parse and enhance the scraped content using natural language processing.
- Save the finalised recipe data into a Supabase Postgresql database.
- Allow for easy query-based recipe matching to easily find recipes 

---

## Machine Learning and NLP

- **ML Model:**
  - **Algorithm:** [XGBoost](https://xgboost.ai/)
  - **Training:** Custom-trained to detect valid recipe pages based on web page content.
  - **Role:** To decide whether a page contains a recipe worth extracting.

- **Natural Language Processing:**
  - **Library:** [spaCy](https://spacy.io/)
  - **Purpose:** To process and clean the text, extract key entities (like ingredients and steps), and allow for easy query-based recipe matching.

---

## Technologies Used

- **Flask** — Lightweight backend framework for serving API endpoints.
- **Python** — Core programming language.
- **XGBoost** — ML model training and predictions.
- **spaCy** — NLP for text processing and enhancement.
- **BeautifulSoup / Requests** — Web scraping and parsing libraries.
- **Supabase** — Database storage and management.

---

## Key Features

- **Smart Scraping:**  
  Only web pages with verified, model-approved recipes are scraped and stored.

- **NLP-Enhanced Structuring and Matching:**  
  Recipes are properly matched with queries from the frontend or the Node-based backend using NLP techniques to produce accurate results.

- **Automated Storage:**  
  Successfully extracted recipes are automatically inserted into the Supabase database.

- **Extensible Design:**  
  The backend is structured to allow easy model updates, additional scraping targets, and future feature expansion.

---

## Contact

For any queries, suggestions, or improvements, feel free to open a discussion or reach out!

