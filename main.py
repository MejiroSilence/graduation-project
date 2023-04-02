import argparse

from run import train

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--agentHiddenDim', default=32)
    parser.add_argument('--actorLR', default=0.001)
    parser.add_argument('--criticHiddenDim', default=32)
    parser.add_argument('--criticLR', default=0.001)
    parser.add_argument('--epsilon', default=0.1)
    parser.add_argument('--gamma', default=0.9)
    parser.add_argument('--tau', default=0.01)
    parser.add_argument('--epoch', default=1000)
    args=parser.parse_args()
    train(args)
