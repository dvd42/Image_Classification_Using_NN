import sys



def process_runtime_arguments():

    if len(sys.argv) > 1:
        if sys.argv[1] == "--help":
                print("Usage: main.py")
                print("-lr learning_rate for the model")
                print("-ovr use ovr architecture")
                print("-dth depth of the network architecture")
                print("-wth width reduction rate of the network architecture")
                print("-size the size of the mini-batches used for the Gradient Descent")
                print("-m momentum value")
                print ("-v will show results on the standard output")
                print("NOTE: The default runtime parameters are: main.py -dth 3 -wth 0.5 -lr 0.001 -size 128")
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
verbose = True if "-v" in argvs else False
width = float(argvs[argvs.index("-wth") + 1]) if "-wth" in argvs else 1
depth = int(argvs[argvs.index("-dth") + 1]) if "-dth" in argvs else 2
m = float(argvs[argvs.index("-m") + 1]) if "-m" in argvs else 0.9


