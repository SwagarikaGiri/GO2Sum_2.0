# generate_go_doc.py

import os
import re
import random

def parse_go_obo(obo_file_path):
    go_dict = {}
    with open(obo_file_path, 'r', encoding='utf-8') as f:
        current_go = None
        current_def = None

        for line in f:
            if line.startswith("[Term]"):
                current_go = None
                current_def = None

            elif line.startswith("id: GO:"):
                current_go = line.strip().split("id: ")[1]

            elif line.startswith("def:"):
                match = re.search(r'"(.*?)"', line)
                if match:
                    current_def = match.group(1)

            elif line.strip() == "" and current_go and current_def:
                go_dict[current_go] = current_def

    return go_dict

def remove_evidence_code(text):
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'\[|\]', '', text)
    return text.strip()

def get_go_description(go_terms, go_definitions):
    descriptions = []
    for term in go_terms:
        if term in go_definitions:
            descriptions.append(go_definitions[term])
    random.shuffle(descriptions)
    return ' '.join(descriptions)
