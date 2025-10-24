# A-word-cloud-generator-based-on-API-calls
This project is a tool that scrapes a modern Chinese poet’s works, extracts high‑frequency imagery, and uses an API to assign emotion‑related colors before rendering a stylized, name‑shaped word cloud. It combines web scraping, text cleaning and segmentation, synonym merging, AI‑assisted color selection, and image generation into a simple pipeline.

#Key features:

1. Fetches poems from shiku.org using the poet’s pinyin page, cleans and segments Chinese text with jieba.
2. Merges common imagery synonyms and counts frequencies; saves the full text and frequency data.
3. Calls DashScope (Qwen) to map each imagery word to one of a predefined Chinese color terms, then recolors the cloud.
4. Generates a high‑resolution word cloud masked to the poet’s Chinese name, with adjustable fonts and styling.
5. Retries API calls with exponential backoff and saves outputs (PNG and JSON) for reproducibility.

#Requirements:

1. Python packages: requests, beautifulsoup4, jieba, wordcloud, pypinyin, matplotlib, Pillow, numpy, openai (DashScope compatible), tenacity.
2. An environment variable DASHSCOPE_API_KEY and access to the DashScope compatible endpoint.
3. A Chinese font (e.g., simkai.ttf) available to render characters; adjust paths for your system.

#Quick start:

1. Install dependencies and set DASHSCOPE_API_KEY.
2. Ensure simkai.ttf is available and update output/project directories as needed.
3. Run the script, input a modern Chinese poet’s name, and find the generated colored word cloud PNG and the imagery‑to‑color JSON in the output folders.
