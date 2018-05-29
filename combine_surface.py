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
from sympy import diff, symbols, sin, cos


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
    theta = np.linspace(-np.pi / 2, np.pi / 2, num=40)
    thi = np.linspace(0, 2 * np.pi, num=40)
    points = np.empty(0)
    normals = np.empty(0)
    for u in theta:
        for v in thi:
            x = center[0] + a * np.sin(v) * np.cos(u)
            y = center[1] + b * np.sin(v) * np.sin(u)
            z = center[2] + c * np.cos(v)
            points = np.append(points, np.array([[x, y, z]]))
            dx_u = -a * np.sin(u) * np.sin(v)
            dx_v = a * np.cos(u) * np.cos(v)
            dy_u = b * np.sin(v) * np.cos(u)
            dy_v = b * np.sin(u) * np.cos(v)
            dz_u = 0
            dz_v = -c * np.sin(v)
            normal = np.cross(np.array([[dx_u, dy_u, dz_u]]), np.array([[dx_v, dy_v, dz_v]]))
            normal = normal / np.linalg.norm(normal) if np.linalg.norm(normal) != 0 else normal
            normals = np.append(normals, normal)
    return points.reshape((-1, 3)), normals.reshape((-1, 3))


def draw_points_cloud(points, normals=None, color='r', marker='o'):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    if points.shape[0] > 1000:
        step = int(points.shape[0] / 1000)
    ax.scatter(points[::step, 0], points[::step, 1], points[::step, 2], c=color, marker=marker, s=3)
    if normals is not None:
        normals = points + normals / 100
        for i in range(0, points.shape[0], step):
            line = np.stack((points[i], normals[i]), axis=-1)
            ax.plot(line[0], line[1], line[2], 'g-')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

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


def calculate_m_opt(x_opt):
    alpha, betta, gamma, t_x, t_y, t_z = tuple(x_opt)
    theta_ang = np.sum(x_opt[:3])
    if theta_ang < 0.00001:
        T = np.array([[1, -gamma, betta, t_x], [gamma, 1, -alpha, t_y], [-betta, alpha, 1, t_z], [0, 0, 0, 1]])
    else:
        a_11 = np.cos(gamma) * np.cos(betta)
        a_12 = -np.sin(gamma) * np.cos(alpha) + np.cos(gamma) * np.sin(betta) * np.sin(alpha)
        a_13 = np.sin(gamma) * np.sin(alpha) + np.cos(gamma) * np.sin(betta) * np.cos(alpha)
        a_21 = np.sin(gamma) * np.cos(betta)
        a_22 = np.cos(gamma) * np.cos(alpha) + np.sin(gamma) * np.sin(betta) * np.sin(alpha)
        a_23 = -np.cos(gamma) * np.sin(alpha) + np.sin(gamma) * np.sin(betta) * np.cos(alpha)
        a_31 = -np.sin(betta)
        a_32 = np.cos(betta) * np.sin(alpha)
        a_33 = np.cos(betta) * np.cos(alpha)
        T = np.array([[a_11, a_12, a_13, 0],
                      [a_21, a_22, a_23, 0],
                      [a_31, a_32, a_33, 0],
                      [0, 0, 0, 1]])
    return T.reshape(4, 4)


def search_optimal_transform_with_normals(data, model, normals_model, indices):
    """

    :param normals_model:
    :param data:
    :param model:
    :param indices:
    :return:


    """
    assert data.shape == model.shape

    m = data.shape[1]

    model = model[indices]
    normals_model[indices]

    b = np.diag(np.dot(normals_model, model.T)) - np.diag(np.dot(normals_model, data.T))
    a = np.cross(data, normals_model)
    A = np.hstack((a, normals_model))
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    matr_s = np.diagflat(S)
    S_inverse = np.linalg.inv(matr_s)
    A_pse_inverse = np.dot(np.dot(Vt, S_inverse), U.T)
    x_opt = np.dot(A_pse_inverse, b)
    M = calculate_m_opt(x_opt)
    return M


def p_to_p_min(data, model, indices):
    error = np.sum(np.linalg.norm(data - model[indices], axis=1) ** 2)
    return error


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
        distances, indices = search_nearest_neighbors(data[:m, :].T, model[:m, :].T)

        error = p_to_p_min(data[:m, :].T, model[:m, :].T, indices)
        if error < tolerance:
            break

        # T, _, __ = search_optimal_transform(data[:m, :].T, model[:m, :].T)
        T = search_optimal_transform_with_normals(data[:m, :].T, model[:m, :].T, model_normals, indices)
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


def extract_points_cloud(path_to_file):
    points = np.empty(0)
    faces = np.empty(0, dtype='int')
    n_points = 0
    n_faces = 0
    with open(path_to_file, 'rb') as ply_file:
        t_start = datetime.now()
        for line in ply_file.readlines():
            # print(line)
            if b'vertices' in line:
                n_points = int(line.split()[-1].decode())
                continue
            if b'faces' in line:
                n_faces = int(line.split()[-1].decode())
                continue
            if b'v' in line.split():
                x, y, z = line.decode().split()[1:]
                points = np.append(points, np.array([[x, y, z]], dtype='float32'))
                continue
            if b'f' in line.split():
                p_ind_1, p_ind_2, p_ind_3 = line.decode().split()[1:]
                faces = np.append(faces, np.array([[int(p_ind_1) - 1, int(p_ind_2) - 1, int(p_ind_3) - 1]]))
    t_finish = datetime.now()
    print(t_finish - t_start)
    points = points + np.abs(np.min(points))
    return points.reshape(n_points, 3), faces.reshape(n_faces, 3)


def get_normal(p1, p2, p3):
    v1 = p1 - p2
    v2 = p2 - p3
    normal = np.cross(v1, v2)
    return normal


def generate_normals(model_set, model_faces=[]):
    closest_normals = defaultdict(list)
    normals_point = np.empty(0)
    if len(model_faces) == 0:
        print("Couldn\'t calculate normals, have\'t faces")
        return normals_point
    for face in model_faces:
        normal = get_normal(model_set[face[0]], model_set[face[1]], model_set[face[2]])
        closest_normals[face[0]].append(normal)
        closest_normals[face[1]].append(normal)
        closest_normals[face[2]].append(normal)
    for i in range(model_set.shape[0]):
        normals = np.array([closest_normals[i]])
        mean_normal = np.mean(normals, axis=1)
        mean_normal = mean_normal / np.linalg.norm(mean_normal) if np.linalg.norm(mean_normal) != 0 else mean_normal
        normals_point = np.append(normals_point, mean_normal)
    return normals_point.reshape((-1, 3))


def main():
    args = argument_parse()
    os.makedirs(args.o, exist_ok=True)
    model_set = None
    data_set = None
    model_normals = None
    if os.path.isfile(args.m) and os.path.isfile(args.s):
        model_set, model_faces = extract_points_cloud(args.m)
        model_normals = generate_normals(model_set, model_faces)
        draw_points_cloud(model_set, normals=model_normals, color='r', marker='o')
        data_set, _ = extract_points_cloud(args.s)
    else:
        model_set, model_normals = make_points_ellipsoid()
        # draw_points_cloud(model_set, normals=model_normals, color='r', marker='o')
        data_set, _ = make_points_ellipsoid(center=[0.5, 0.5, 0.4])
    draw_set_points_clouds(model_set, data_set)
    transform_matrix, number_iteration = icp(model_set, data_set, model_normals)
    print(transform_matrix, number_iteration)


if __name__ == '__main__':
    main()
