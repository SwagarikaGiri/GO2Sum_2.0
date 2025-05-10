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
        go_doc = get_go_description(go_ids, self.go_definitions).strip()
        function_cc = str(row['Function [CC]']).strip()

        if not go_doc or not function_cc or function_cc.lower() == "nan":
            return {'input': None, 'target': None}

        return {'input': go_doc, 'target': function_cc}
