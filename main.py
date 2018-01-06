import NNet as NN
import runtime_parser as rp
import file_writer as fw
import run_model 


def main():
	print("Running %s model with:\nLearning Rate = %s\nMomentum = %s\nDepth = %d\nWidth = %s \nBatch_size: %d"
		% ("ovr" if rp.ovr else "5 classes",str(rp.lr),str(rp.m),rp.depth,str(rp.width),rp.size))

	fw.create_dir()


	if not rp.ovr:
		run_model.run_full_net(5,rp.lr,rp.width,rp.depth,rp.size,rp.ovr)
		
	else:
		run_model.run_ovr_nets(2,rp.lr,rp.width,rp.depth,rp.size,rp.ovr)


if __name__ == '__main__':
	main()
