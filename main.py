import torch
from torch import optim
import numpy as np

import NNet as NN
import runtime_parser as rp
import file_writer as fw


print("Running %s model with:\nlr = %.2f\nmini_batch_size: %d\nTolerance: %d\n" % 
	("ovr" if rp.ovr else "5 classes",rp.lr,rp.size,rp.tolerance))


fw.create_dir()

if not rp.ovr:
	NN.run_full_net(5,rp.lr,rp.ovr)
else:
	NN.run_ovr_nets(2,rp.lr,rp.ovr)
