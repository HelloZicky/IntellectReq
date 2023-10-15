# coding=utf-8
import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--tables', default="", type=str, required=True)
    parser.add_argument('--dataset', default="movielens", type=str, required=True)
    parser.add_argument('--outputs', default="", type=str)
    parser.add_argument('--model', default='din', type=str, help='model')
    parser.add_argument('--ini_checkpoint', default='', type=str, help='initial checkpoint')
    parser.add_argument('--batch_size', dest='batch_size', default=32, type=int)
    parser.add_argument('--parallel', dest='num_workers', default=1, type=int, metavar='N',
                        help='number of data loading workers (default: 16)')
    parser.add_argument('--lr', default=0.01, type=float, help='initial learning rate')

    args = parser.parse_args()
    return args
