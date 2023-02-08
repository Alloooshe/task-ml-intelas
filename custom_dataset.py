import torch 
import json

class LedgerDataset(torch.utils.data.Dataset):
    source_lang = "source"
    target_lang = "target"
    prefix = "translate source to target: "
    
    def __init__(self, path, tokenizer):
        self.path = path 
        self.tokenizer = tokenizer 
        self.data = self.load_data()
        self.encodings = self.preprocess()        
        
    
    def load_data(self): 
        with open(self.path) as json_data:
            return json.load(json_data)

    def __len__(self):
        return len(self.data)

    
    def __getitem__(self, idx):
       item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
       item["translation"] = self.data[idx]["translation"]
       return item
    
    def preprocess(self):
        inputs = [LedgerDataset.prefix + example["translation"][LedgerDataset.source_lang] for example in self.data]
        targets = [example["translation"][LedgerDataset.target_lang] for example in self.data]
        model_inputs = self.tokenizer(inputs, text_target=targets, max_length=32, truncation=True)
        return model_inputs

from transformers import AutoTokenizer

if __name__ =="__main__" : 
    tokenizer = AutoTokenizer.from_pretrained("t5-base")

    data = LedgerDataset("data.json",tokenizer)
    print(len(data))
    print(data[0])