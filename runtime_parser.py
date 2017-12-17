import sys



def process_runtime_arguments():

    if len(sys.argv) > 1:
        if sys.argv[1] == "--help":
                print("Usage: main.py")
                print("-lr learning_rate for the model")
                print("-ovr use ovr architecture")
                print("-t number of epochs that the model can train without improvement (early stopping)")
                print("-dth depth of the network architecture")
                print("-wth width reduction rate of the network architecture")
                print("-size the size of the mini-batches used for the Gradient Descent")
                print("-iter  how many models do you want to run with the same architecture (this are NOT epochs)")
                print("-lrd the rate in which the learning_rate decreases")
                print ("-v will show results on the standard output")
                print("NOTE: The default runtime parameters are: main.py -dth 3 -wthv 0.5 -lr 0.001 -lrd 0.9 -size 128 -t 10 -iter 10")
                sys.exit(1)

    argvs = []
    for i in range(len(sys.argv)):
        argvs.append(sys.argv[i])

    return argvs

argvs = process_runtime_arguments()

# Get the the values of the runtime parameters
lr = float(argvs[argvs.index("-lr") + 1]) if "-lr" in argvs else 0.001
ovr = True if "-ovr" in argvs else False
size = int(argvs[argvs.index("-size") + 1]) if "-size" in argvs else 128
tolerance = int(argvs[argvs.index("-t") + 1]) if "-t" in argvs else 10
iterations = int(argvs[argvs.index("-iter") + 1]) if "-iter" in argvs else 10
verbose = True if "-v" in argvs else False
lrd =  float(argvs[argvs.index("-lrd") + 1]) if "-lrd" in argvs else 0.9
width = float(argvs[argvs.index("-wth") + 1]) if "-wth" in argvs else 0.5
depth = int(argvs[argvs.index("-dth") + 1]) if "-dth" in argvs else 3


