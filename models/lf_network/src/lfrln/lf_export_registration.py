"""Geometry from LF_reconstruct_compiled.m, with MATLAB imwarp edge fill.

The existing noise-profile registration is retained for historical checkpoint
reproducibility. This corrected operator is for new native-sensor forward
models and new calibration exports; never apply it to already exported LF.
"""
import torch
from torch.nn import functional as F
from .emccd import LFNoiseRegistration


class LFExportRegistration(LFNoiseRegistration):
    def forward(self,value):
        if value.shape[-2:]!=self.sensor_shape:
            raise ValueError(f'Expected native MATLAB matrix {self.sensor_shape}, not registered LF')
        shape=value.shape[:-2]
        v=value.float().reshape(-1,1,*self.sensor_shape)
        v=torch.rot90(v[...,1:1003,:].flip(-1),1,(-2,-1))
        grid=self.rotation_grid.expand(v.shape[0],-1,-1,-1)
        v=F.grid_sample(v,grid,mode='bilinear',padding_mode='zeros',align_corners=False)
        # MATLAB imwarp uses FillValues for interpolation queries outside the
        # pixel-center range [1,1002], unlike grid_sample's partial edge blend.
        valid=(grid.abs()<=1-1/1002).all(-1)
        v=v*valid[:,None]
        v=(v[:,:,self.indices,:]*self.weights[None,None,:,:,None]).sum(3)
        v=(v[...,self.indices]*self.weights[None,None,None,:,:]).sum(-1)
        return v[...,2:-10,5:-7].reshape(*shape,*self.output_shape)
