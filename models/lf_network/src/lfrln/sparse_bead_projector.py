"""Exact active-depth evaluation of the finite-bead forward projector.

This only skips zero source planes, retaining one extra plane on each side for
position derivatives. It does not truncate the LF PSF or change optical support.
"""
import torch
from torch.nn import functional as F
from .bead_calibration import LocalBeadProjector
from .physics_rearrange import space_to_view_depth,view_depth_to_space


class SparseDepthBeadProjector(LocalBeadProjector):
    def __call__(self,volume):
        if volume.shape[1:]!=(self.config.depth,self.local_size,self.local_size):
            raise ValueError('Source must fit the local finite-bead volume')
        outputs=[];depth=self.config.depth;views=self.config.nnum**2
        weights=self.weight.reshape(views,depth,*self.weight.shape[1:])
        for sample in volume:
            # Support selection is discrete, but the linear operator and its
            # position derivative are exact within the retained support. This
            # operator is for compact moving-sphere sources, not arbitrary
            # trainable network volumes with zero planes that may become active.
            active=sample.detach().abs().sum((-2,-1)).nonzero().flatten()
            if not len(active):raise ValueError('A finite bead must contain fluorescence')
            first=max(int(active[0])-1,0);last=min(int(active[-1])+2,depth)
            selected=weights[:,first:last].reshape(views*(last-first),*self.weight.shape[1:]).contiguous()
            source=space_to_view_depth(sample[None,first:last],self.config.nnum).to(selected.dtype)
            pattern=F.conv_transpose2d(source,selected)
            outputs.append(view_depth_to_space(pattern,channels=1,block_size=self.config.nnum).float()*self.gain)
        return torch.cat(outputs)
