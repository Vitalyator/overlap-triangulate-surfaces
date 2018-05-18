import numpy as np
import os
from sklearn.neighbors import NearestNeighbors
from pyntcloud import PyntCloud
from datetime import datetime
import argparse


def search_nearest_neighbors(set_a, set_b):
    pass


def icp(model_set_points_cloud, data_set_points_cloud):
    pass


def argument_parse():
    argument_parser = argparse.ArgumentParser(description='Script combine two points cloud in 3 dimensional space '
                                                          'with used algorithm ICP on .ply-files')
    argument_parser.add_argument('-m', type=str, default='', help='input path to .ply-file with model set for compare, '
                                                                  '(default used random points of cloud')
    argument_parser.add_argument('-s', type=str, default='', help='input path to .ply-file with data set, '
                                                                  '(default used random points of cloud')
    argument_parser.add_argument('-o', type=str, default='/tmp/', help='output path to result')
    args = argument_parser.parse_args()
    return args


def extract_points_cloud(path_to_file):
    if not os.path.isfile(path_to_file):
        points = 2 * np.random.random_sample((100, 3)) - 1
        print(points, type(points))
        return points
    points = np.empty(0, dtype='float32')
    n_vertices = 0
    with open(path_to_file, 'rb') as ply_file:
        t_start = datetime.now()
        for line in ply_file.readlines():
            if b'vertices' in line:
                n_vertices = int(line.decode().split()[-1])
                continue
            print(line)
            if b'v' in line:
                x, y, z = line.decode().split()[1:]
                points = np.append(points, np.array([[x, y, z]], dtype='float32'))
    points = points.reshape((n_vertices, 3))
    t_finish = datetime.now()
    # print(points)
    print(t_finish - t_start)
    return points


def main():
    args = argument_parse()
    model_set_points_cloud = extract_points_cloud(args.m)
    data_set_points_cloud = extract_points_cloud(args.s)
    transrofmation_points = icp(model_set_points_cloud, data_set_points_cloud)
    icp(None, None)


if __name__ == '__main__':
    main()
