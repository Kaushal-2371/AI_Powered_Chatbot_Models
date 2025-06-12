# This is a script to filter the arXiv dataset for Computer Science papers. You can use it according to your needs.
# It reads a JSONL file containing arXiv metadata, filters for papers in the Computer Science category, and saves the titles and abstracts to a TXT file.

import json

input_file = 'arxiv-metadata-oai-snapshot.json'  # orignal data
output_file = 'arxiv_cs_subset_8000.txt'  # saving 8000 papers out of 367968
max_papers = 8000  # Limit

papers = []
count = 0

print("Processing...")

# Open the file and read line by line
with open(input_file, 'r', encoding='utf-8') as f_in:
    for line in f_in:
        if count >= max_papers:
            break
        paper = json.loads(line)
        
        # Only choose computer science papers
        if 'cs' in paper['categories']:
            text = f"Title: {paper['title']}\nAuthors: {paper['authors']}\nAbstract: {paper['abstract']}"
            papers.append(text)
            count += 1

print(f"Selected {len(papers)} papers.")

# Save to output file
with open(output_file, 'w', encoding='utf-8') as f_out:
    f_out.write('\n\n'.join(papers))

print("Saved selected papers")
