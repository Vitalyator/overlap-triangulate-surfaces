import numpy as np
import os
from sklearn.neighbors import NearestNeighbors
# from pyntcloud import PyntCloud
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from mpl_toolkits.mplot3d import Axes3D
from collections import defaultdict
from datetime import datetime
import argparse


def randrange(n, vmin, vmax):
    '''
    Helper function to make an array of random numbers having shape (n, )
    with each number distributed Uniform(vmin, vmax).
    '''
    return (vmax - vmin)*np.random.rand(n) + vmin


def draw_set_points_clouds(model_set_points_cloud, data_set_points_cloud):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(model_set_points_cloud[:, 0], model_set_points_cloud[:, 1], model_set_points_cloud[:, 2],
               c='r', marker='o')
    ax.scatter(data_set_points_cloud[:, 0], data_set_points_cloud[:, 1], data_set_points_cloud[:, 2],
               c='b', marker='+')
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    plt.show()


def make_points_ellipsoid(center=[0.5, 0.5, 0.5], a=0.4, b=0.2, c=0.2):
    theta = np.linspace(-np.pi / 2, np.pi / 2, num=20)
    thi = np.linspace(0, 2 * np.pi, num=30)
    points = np.empty(0)
    for ang1 in theta:
        for ang2 in thi:
            x = center[0] + a * np.sin(ang2) * np.cos(ang1)
            y = center[1] + b * np.sin(ang2) * np.sin(ang1)
            z = center[2] + c * np.cos(ang2)
            points = np.append(points, np.array([[x, y, z]]))
    points = np.unique(points, axis=0)
    return points.reshape((-1, 3))


def draw_points_cloud(points, color='r', marker='o'):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=color, marker=marker)
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    plt.show()


def search_nearest_neighbors(data_set, model_set):
    """

    :param model_set:
    :param data_set:
    :return:
    >>> sample_array = np.array([[1.0, 1.0, 0.0], [2.0, 2.0, 0.0], [3.0, 3.0, 0.0]])
    >>> dest_array = np.array([[2.0, 1.5, 0.0], [3.0, 2.5, 0.0],[1.0, 0.0, 0.0]])
    >>> search_nearest_neighbors(sample_array, dest_array)
    (array([0.5, 0.5, 1. ]), array([1, 2, 0]))
    """
    neighbors = NearestNeighbors(n_neighbors=1)
    neighbors.fit(data_set)
    distances, indices = neighbors.kneighbors(model_set, return_distance=True)
    return distances.ravel(), indices.ravel()


def search_optimal_transform(data, model):
    assert data.shape == model.shape

    m = data.shape[1]

    centroid_data = np.mean(data, axis=0)
    centroid_model = np.mean(model, axis=0)

    normalize_data = data - centroid_data
    normalize_model = model - centroid_model

    H = np.dot(normalize_data.T, normalize_model)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)

    if np.linalg.det(R) < 0:
        Vt[m-1, :] *= -1
        R = np.dot(Vt.T, U.t)

    t = centroid_model.T - np.dot(R, centroid_data.T)

    T = np.identity(m+1)
    T[:m, :m] = R
    T[:m, m] = t

    return T, R, t


def search_optimal_transform_with_normals(data, model, normals_model, indices):
    """

    :param data:
    :param model:
    :param indices:
    :return:


    """
    assert data.shape == model.shape

    m = data.shape[1]

    model = model[indices]
    b = np.dot(normals_model.T, model) - np.dot(normals_model.T, data)
    a = np.cross(data, normals_model)
    A = np.hstack(a, normals_model)
    U, S, Vt = np.linalg.svd(A)
    S_inverse = np.linalg.inv(S)
    A_pse_inverse = np.dot(np.dot(Vt.T, S_inverse), U.T)
    x_opt = np.dot(A_pse_inverse, b)
    return x_opt


def p_to_p_min(data, model, indices):
    errors = []
    for i in range(data.shape[0]):
        error = pow(np.linalg.norm(data[i] - model[indices[i]]), 2)
        errors.append(error)
    return sum(errors)


def icp(data_set_points, model_set_points, model_normals, init_pose=None, max_iterations=20, tolerance=0.001):
    assert model_set_points.shape == data_set_points.shape

    m = model_set_points.shape[1]

    model = np.ones((m + 1, model_set_points.shape[0]))
    data = np.ones((m + 1, model_set_points.shape[0]))
    model[:m, :] = np.copy(model_set_points.T)
    data[:m, :] = np.copy(data_set_points.T)

    if init_pose is not None:
        data = np.dot(init_pose, data)

    for i in range(max_iterations):
        distances, indices = search_nearest_neighbors(data, model)

        error = p_to_p_min(data, model, indices)
        if error < tolerance:
            break

        T, _, __ = search_optimal_transform(data[:m, :].T, model[:m, :].T)
        data = np.dot(T, data)
        draw_set_points_clouds(model[:m, :].T, data[:m, :].T)

    T, _, __ = search_optimal_transform(data_set_points, data[:m, :].T)

    return T, i


def argument_parse():
    argument_parser = argparse.ArgumentParser(description='Script combine two points cloud in 3 dimensional space '
                                                          'with used algorithm ICP on .ply-files')
    argument_parser.add_argument('-m', type=str, default='', help='input path to .ply-file with model set for compare, '
                                                                  '(default used random points of cloud')
    argument_parser.add_argument('-s', type=str, default='', help='input path to .ply-file with data set, '
                                                                  '(default used random points of cloud')
    argument_parser.add_argument('-o', type=str, default='/tmp/compare_results/', help='output path to result')
    args = argument_parser.parse_args()
    return args


def extract_points_cloud(path_to_file, copy_first_set=None):
    points = np.empty(0)
    faces = np.empty(0)
    if copy_first_set is not None:
        points = make_points_ellipsoid(center=[0.5, 0.5, 0.4])
    elif not os.path.isfile(path_to_file):
        points = make_points_ellipsoid()
    else:
        with open(path_to_file, 'rb') as ply_file:
            t_start = datetime.now()
            for line in ply_file.readlines():
                if b'vertices' or b'faces' in line:
                    continue
                # print(line)
                if b'v' in line:
                    x, y, z = line.decode().split()[1:]
                    points = np.append(points, np.array([[x, y, z]], dtype='float32'))
                if b'f' in line.split():
                    p_ind_1, p_ind_2, p_ind_3 = line.decode().split()[1:]
                    faces = np.append(faces, np.array([[int(p_ind_1) - 1, int(p_ind_2) - 1, int(p_ind_3) - 1]], dtype='int'))
        t_finish = datetime.now()
        print(t_finish - t_start)
    return points.reshape((-1, 3)), faces.reshape((-1, 3))


def get_normal(p1, p2, p3):
    v1 = p1 - p2
    v2 = p2 - p3
    normal = np.cross(v1, v2)
    return normal


def generate_normals(model_set_points_cloud, model_faces=[]):
    closest_normals = defaultdict(list)
    normals_point = np.empty(0)
    if len(model_faces) != 0:
        for face in model_faces:
            normal = get_normal(model_set_points_cloud[face[0]], model_set_points_cloud[face[1]], model_set_points_cloud[face[2]])
            closest_normals[face[0]].append(normal)
            closest_normals[face[1]].append(normal)
            closest_normals[face[2]].append(normal)
        for i in range(model_set_points_cloud.shape[0]):
            normals = np.array([closest_normals[i]])
            normals_point = np.append(normals_point, np.mean(normals, axis=1))
    else:
        neighbors = NearestNeighbors(n_neighbors=5)
        neighbors.fit(model_set_points_cloud)
        indices = neighbors.kneighbors(model_set_points_cloud, return_distance=False)
        for neighbors_points in indices:
            print(neighbors_points)
            sub_set = model_set_points_cloud[neighbors_points]
            rad = np.linalg.norm(sub_set, axis=1)
            zen = np.arccos(sub_set[:, -1] / rad)
            azi = np.arctan2(sub_set[:, 1], sub_set[:, 0])

            tris = mtri.Triangulation(zen, azi)
            # TODO complete calculate normals
            print(tris.triangles)
    return normals_point.reshape((-1, 3))


def main():
    args = argument_parse()
    os.makedirs(args.o, exist_ok=True)
    model_set_points_cloud, model_faces = extract_points_cloud(args.m)
    model_normals = generate_normals(model_set_points_cloud, model_faces)
    data_set_points_cloud, _ = extract_points_cloud(args.s, copy_first_set=model_set_points_cloud)
    draw_set_points_clouds(model_set_points_cloud, data_set_points_cloud)
    transform_matrix, number_iteration = icp(model_set_points_cloud, data_set_points_cloud, model_normals)
    print(transform_matrix, number_iteration)


if __name__ == '__main__':
    main()
