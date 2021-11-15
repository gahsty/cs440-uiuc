import argparse
import queue

import numpy as np


def main(args):
    det=np.array([1,2,3,4])
    train=np.array([2,3,4,5])
    print(np.power(det,2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CS440 MP5 Classify')
    args = parser.parse_args()
    main(args)