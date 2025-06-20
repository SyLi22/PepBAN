import pandas as pd
import torch.utils.data as data
import torch
import numpy as np
from utils import integer_label_protein, integer_label_peptide


class PepPIDataset(data.Dataset):
    def __init__(self, list_IDs, df, data, vis=False, KPGT=False, concat=False):
        self.list_IDs = list_IDs
        self.df = df
        self.data=data
        self.vis = vis
        self.KPGT=KPGT
        self.concat=concat

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        index = self.list_IDs[index]
        pep = self.df.iloc[index]['Peptide']
        pep = integer_label_peptide(pep, self.data)
        pro = self.df.iloc[index]['Protein']
        pro = integer_label_protein(pro, self.data)
        y = self.df.iloc[index]["Y"]
        return pep, pro, y


class MultiDataLoader(object):
    def __init__(self, dataloaders, n_batches):
        if n_batches <= 0:
            raise ValueError("n_batches should be > 0")
        self._dataloaders = dataloaders
        self._n_batches = np.maximum(1, n_batches)
        self._init_iterators()

    def _init_iterators(self):
        self._iterators = [iter(dl) for dl in self._dataloaders]

    def _get_nexts(self):
        def _get_next_dl_batch(di, dl):
            try:
                batch = next(dl)
            except StopIteration:
                new_dl = iter(self._dataloaders[di])
                self._iterators[di] = new_dl
                batch = next(new_dl)
            return batch

        return [_get_next_dl_batch(di, dl) for di, dl in enumerate(self._iterators)]

    def __iter__(self):
        for _ in range(self._n_batches):
            yield self._get_nexts()
        self._init_iterators()

    def __len__(self):
        return self._n_batches
