# dataset.py

import torch
from torch.utils.data import Dataset
import pandas as pd
from generate_go_doc import parse_go_obo, get_go_description

class GODocDataset(Dataset):
    def __init__(self, tsv_path, obo_path):
        self.data = pd.read_csv(tsv_path, sep='\t')
        self.go_definitions = parse_go_obo(obo_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        go_ids = row['Gene Ontology IDs'].split('; ')
        go_doc = get_go_description(go_ids, self.go_definitions)
        function_cc = row['Function [CC]']
        return {'input': go_doc, 'target': function_cc}
