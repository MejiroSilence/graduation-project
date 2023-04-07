import argparse

from run import train

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--agentHiddenDim', default=128,type=int)
    parser.add_argument('--actorLR', default=0.0005,type=float)
    parser.add_argument('--criticHiddenDim', default=128,type=int)
    parser.add_argument('--mixerHiddenDim', default=128,type=int)
    parser.add_argument('--criticLR', default=0.0005,type=float)
    parser.add_argument('--epsilon', default=0.1,type=float)
    parser.add_argument('--gamma', default=0.99,type=float)
    parser.add_argument('--tau', default=0.01,type=float)
    parser.add_argument('--epoch', default=100000,type=int)
    parser.add_argument('--evalEp',default=100,type=int)
    parser.add_argument('--batchSize',default=128,type=int)
    parser.add_argument('--epochEpisodes',default=16,type=int)
    parser.add_argument('--sampleSize',default=64,type=int)
    parser.add_argument('--entropy',default=0.09,type=float)
    parser.add_argument('--td_lambda',default=0.6,type=float)
    args=parser.parse_args()
    train(args)
