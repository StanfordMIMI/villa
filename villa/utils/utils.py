import torch
import numpy as np
import random
import collections
from torch._six import string_classes


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def collate_grouped(batch, key=None):
    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, collections.abc.Mapping):
        return elem_type(
            {key: collate_grouped([d[key] for d in batch], key=key) for key in elem}
        )
    elif isinstance(elem, torch.Tensor):
        out = None
        if torch.utils.data.get_worker_info() is not None:
            numel = sum(x.numel() for x in batch)
            storage = elem.storage()._new_shared(numel)
            out = elem.new(storage)
        if key in ["img", "attr_labels"]:
            out = out.view(-1, list(elem.size())[-1])
            return torch.cat(batch, 0, out=out)
        else:
            return torch.stack(batch, 0, out=out)
    elif isinstance(elem, string_classes):
        return batch
