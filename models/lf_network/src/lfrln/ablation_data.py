"""Epoch/index-addressed noise, shared exactly between ablation branches."""
from torch.utils.data import Dataset
from .datasets import MixedLFNoiseDataset


class EpochNoiseReplay(Dataset):
    def __init__(self,dataset,*,seed,**noise_options):
        self.base_seed=int(seed);self.epoch=0
        self.noisy=MixedLFNoiseDataset(dataset,deterministic=True,seed=seed,**noise_options)

    def set_epoch(self,epoch):
        self.epoch=int(epoch)
        if self.epoch<0:raise ValueError('Epoch must be nonnegative')
        self.noisy.seed=self.base_seed+self.epoch*len(self)

    def __len__(self):return len(self.noisy)

    def __getitem__(self,index):
        row=self.noisy[index]
        row['ablation_sample_index']=int(index)
        row['ablation_noise_seed']=self.noisy.seed+int(index)
        return row
