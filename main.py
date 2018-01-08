import NNet as NN
import runtime_parser as rp
import file_writer as fw
import run_model 


def main():
	
	print("Running model with:\nLearning Rate = %s\nMomentum = %s\nDepth = %d\nWidth = %s \nBatch_size: %d"
		% (str(rp.lr),str(rp.m),rp.depth,str(rp.width),rp.size))

	fw.create_dir()
	run_model.run(5,rp.lr,rp.width,rp.depth,rp.size)
		

if __name__ == '__main__':
	main()
