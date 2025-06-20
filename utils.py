import os
import random
import torch
import numpy as np
from Bio.Blast.Applications import NcbiblastpCommandline
from tqdm import tqdm
import os
import pandas as pd
import json

def set_seed(seed=1000):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
def mkdir(path):
    path = path.strip()
    path = path.rstrip("\\")
    is_exists = os.path.exists(path)
    if not is_exists:
        os.makedirs(path)
from Bio.Blast import NCBIWWW, NCBIXML
def integer_label_protein(id, data):
    #esm2 is the fold that stores ESM-2 embedding vectors
    encoding=torch.load(f'esm2/{data}/{str(id)}.pt')['mean_representations'][33]
    return encoding
def integer_label_peptide(id, data):
    # esm2 is the fold that stores ESM-2 embedding vectors
    encoding=torch.load(f'esm2/{data}/{str(id)}.pt')['mean_representations'][33]
    return encoding

