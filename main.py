import NNet as NN
import runtime_parser as rp
import file_writer as fw
import run_model 


#TODO: gradient noise? (low)

if rp.verbose:
	print("Running %s model with:\nLearning Rate = %s\nLearning_Rate_Decay = %s\nDepth = %d\nWidth = %s \nBatch_size: %d\nTolerance: %d\nIterations: %d\n" 	% ("ovr" if rp.ovr else "5 classes",str(rp.lr),str(rp.lrd),rp.depth,str(rp.width),rp.size,rp.tolerance,rp.iterations))

fw.create_dir()

if not rp.ovr:
	run_model.run_full_net(5,rp.lr,rp.lrd,rp.width,rp.depth,rp.size,rp.ovr,rp.iterations)
	
else:
	run_model.run_ovr_nets(2,rp.lr,rp.lrd,rp.width,rp.depth,rp.size,rp.ovr,rp.iterations)


