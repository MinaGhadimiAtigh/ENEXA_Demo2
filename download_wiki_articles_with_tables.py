# Improvements made to the original script:
# 1. Url is included in the resulting output file
# 2. Headings are matched with the extracted paragraph.
# 3. Tables are extracted as well as paragraphs.
# 4. The script uses structural information and handles different types of content better.
from bs4 import *
import requests
import re
import argparse
import json
import os

def get_wiki_urls(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        urls = json.load(f)
    return urls

def download_wiki_articles(urls, output_dir):
    for url in urls:
        print(f'Downloading {url}')
        page_name = url.split('/')[-1] # we get the page name from the url
        page = requests.get(url)
        page_content = BeautifulSoup(page.text,'html.parser').select('body')[0]
        page_text = [{"url": url}]
        
        current_heading = "Introduction"
        section_content = []
        
        for tag in page_content.find_all(['h1', 'h2', 'h3', 'p', 'table']):
            if tag.name in ['h1', 'h2', 'h3']:
                if section_content:
                    page_text.append({
                        "heading": current_heading,
                        "paragraphs": section_content
                    })
                    section_content = []
                current_heading = tag.get_text(strip=True)

            elif tag.name == "p":
                text = tag.get_text(strip=True)
                text = re.sub(r'\[\d+\]', '', text)
                text = text.replace('\\', '').replace('[citation needed]', '')
                if text:
                    section_content.append(text)

            elif tag.name == "table":
                table_data = []
                rows = tag.find_all("tr")
                headers = [th.get_text(strip=True) for th in rows[0].find_all("th")]
                
                for row in rows[1:]:
                    cells = row.find_all(["td", "th"])
                    row_data = [cell.get_text(strip=True) for cell in cells]
                    if headers and len(headers) == len(row_data):
                        table_data.append(dict(zip(headers, row_data)))
                    else:
                        table_data.append(row_data)  # fallback: list of values
                
                if table_data:
                    page_text.append({
                        "heading": current_heading,
                        "table": table_data
                    })
        #print(page_text)
        with open(f'{output_dir}/{page_name}.json', 'w', encoding='utf-8') as f:
            json.dump(page_text, f, ensure_ascii=False, indent=4)

def main():
    parser = argparse.ArgumentParser(description='Download Wikipedia Articles')
    parser.add_argument('-f', '--url_file', type=str, default= 'ENEXA_Demo2/input_files/one_url.json', help='Json file with a list of Wikipedia URLs')
    parser.add_argument('-o', '--output_dir', type=str, default='ENEXA_Demo2/wiki_downloads_with_tables_adidas', help='Output directory to set for the downloaded articles')
    print('This updated script downloads text and tables from Wikipedia articles. Given a list of URLs, downloaded text and tables are ssved in JSON format: List of dictionaries with page url, headings and paragraphs, and tables.')
    args = parser.parse_args()
    urls = get_wiki_urls(args.url_file)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    download_wiki_articles(urls, args.output_dir)
    print('Done!')
    
if __name__ == '__main__':
    main()