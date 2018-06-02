import numpy as np
import os
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from collections import defaultdict
from datetime import datetime
import argparse


BUNNY = 'bunny'
ELLIPSOID = 'ellipsoid'
SINUSOID = 'sinusoid'
PATH_TO_FILE_BUNNY = '/home/vitaliy/media/bunny.ply'
ROTATE = 'translation_rotate'


def draw_set_points_clouds(model_set_points_cloud, data_set_points_cloud, output_path, file_name='sample_plot'):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(model_set_points_cloud[:, 0], model_set_points_cloud[:, 1], model_set_points_cloud[:, 2],
               c='r', marker='o')
    ax.scatter(data_set_points_cloud[:, 0], data_set_points_cloud[:, 1], data_set_points_cloud[:, 2],
               c='b', marker='+')
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.savefig(os.path.join(output_path, file_name))
    plt.show()


def make_points_sinusoid():
    pass


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
            dx_u = -a * np.sin(u) * np.sin(v)
            dx_v = a * np.cos(u) * np.cos(v)
            dy_u = b * np.sin(v) * np.cos(u)
            dy_v = b * np.sin(u) * np.cos(v)
            dz_u = 0
            dz_v = -c * np.sin(v)
            normal = np.cross(np.array([[dx_u, dy_u, dz_u]]), np.array([[dx_v, dy_v, dz_v]]))
            if np.linalg.norm(normal) == 0:
                continue
            points = np.append(points, np.array([[x, y, z]]))
            normal = normal / np.linalg.norm(normal)
            normals = np.append(normals, normal)
    return points.reshape((-1, 3)), normals.reshape((-1, 3))


def draw_points_cloud(points, output_path, normals=None, color='r', marker='o'):
    fig = plt.figure()
    ax = Axes3D(fig)
    step = 1
    while len(points[::step]) > 1600:
        step += 1
    ax.scatter(points[::step, 0], points[::step, 1], points[::step, 2], c=color, marker=marker, s=3)
    if normals is not None:
        normals = points + normals / 100
        for i in range(0, points.shape[0], step):
            line = np.stack((points[i], normals[i]), axis=-1)
            ax.plot(line[0], line[1], line[2], 'g-')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.savefig(os.path.join(output_path, 'model'))
    plt.show()


def draw_analyze_graphic(p_plane_results, p_point_results, output_path, name_graphic='plot', sum_values=False):
    fig = plt.figure()
    if sum_values:
        summed_values = []
        for i in range(1, len(p_plane_results) + 1):
            summed_values.append(sum(p_plane_results[:i]))
        p_plane_results = summed_values
        summed_values = []
        for i in range(1, len(p_point_results) + 1):
            summed_values.append(sum(p_point_results[:i]))
        p_point_results = summed_values
    ax1 = fig.add_subplot(111)
    ax1.set(xlabel='iterations', ylabel='error (euclid distance)',
            title=name_graphic, xticks=range(0, max(len(p_point_results), len(p_plane_results)) + 1))
    ax1.plot(range(len(p_plane_results)), p_plane_results, color='red', marker='o', linestyle='dashed',
             linewidth=2, markersize=5, label='point_to_plane_minimization')
    ax1.plot(range(len(p_point_results)), p_point_results, color='blue', marker='+', linestyle='dashed',
             linewidth=2, markersize=10, label='point_to_point_minimization')
    ax1.legend()
    ax1.grid()
    plt.savefig(os.path.join(output_path, name_graphic))
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


def p_to_point_min_func(data, model, normals_model, indices):
    assert data.shape == model.shape
    model = model[indices]
    m = data.shape[1]

    centroid_data = np.mean(data, axis=0)
    centroid_model = np.mean(model, axis=0)

    normalize_data = data - centroid_data
    normalize_model = model - centroid_model

    H = np.dot(normalize_data.T, normalize_model)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)

    if np.linalg.det(R) < 0:
        Vt[m - 1, :] *= -1
        R = np.dot(Vt.T, U.t)

    t = centroid_model.T - np.dot(R, centroid_data.T)

    T = np.identity(m + 1)
    T[:m, :m] = R
    T[:m, m] = t

    return T


def translation_matrix(alpha, betta, gamma):
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


def calculate_m_opt(x_opt):
    alpha, betta, gamma, t_x, t_y, t_z = tuple(x_opt)
    theta_ang = np.sum(x_opt[:3])
    if theta_ang < 0.00001:
        T = np.array([[1, -gamma, betta, t_x], [gamma, 1, -alpha, t_y], [-betta, alpha, 1, t_z], [0, 0, 0, 1]])
    else:
        T = translation_matrix(alpha, betta, gamma)
    return T.reshape(4, 4)


def p_to_plane_min_func(data, model, normals_model, indices):
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
    normals_model = normals_model[indices]

    b = scalar_vectors(model, normals_model) - scalar_vectors(data, normals_model,)
    a = np.cross(data, normals_model)
    matrix_A = np.hstack((a, normals_model))
    U, sigma, Vt = np.linalg.svd(matrix_A, full_matrices=False)
    matrix_sigma = np.diagflat(sigma)
    sigma_inverse = np.linalg.inv(matrix_sigma)
    pse_inverse_matrix_A = np.dot(np.dot(Vt.T, sigma_inverse), U.T)
    x_opt = np.dot(pse_inverse_matrix_A, b)
    M = calculate_m_opt(x_opt)
    return M


def error_metric_p_to_point(data, model, normals_model, indices):
    error = np.sum(np.linalg.norm(data - model[indices], axis=1) ** 2)
    return error


def error_metric_p_to_plane(data, model, normals_model, indices):
    error = scalar_vectors(data - model[indices], normals_model[indices])
    error = sum(error ** 2)
    return error


def scalar_vectors(vectors, normals):
    scalars = np.empty(vectors.shape[0])
    for i in range(vectors.shape[0]):
        scalars[i] = np.vdot(vectors[i], normals[i])
    return scalars


def icp(data_set_points, model_set_points, normals_model, init_pose, output_path, max_iterations=20, tolerance=0.00001,
        p_to_plane=False):
    assert model_set_points.shape == data_set_points.shape
    mean_distance_errors = []
    errors = []
    time_spend = []
    m = model_set_points.shape[1]
    minimize_function = p_to_plane_min_func if p_to_plane else p_to_point_min_func
    error_metric = error_metric_p_to_plane if p_to_plane else error_metric_p_to_point
    error_name = 'point_to_plane' if p_to_plane else 'point_to_point'

    model = np.ones((m + 1, model_set_points.shape[0]))
    data = np.ones((m + 1, model_set_points.shape[0]))
    normals = np.zeros((m + 1, normals_model.shape[0]))
    model[:m, :] = np.copy(model_set_points.T)
    data[:m, :] = np.copy(data_set_points.T)
    normals[:m, :] = np.copy(normals_model.T)

    if init_pose is not None:
        data = np.dot(init_pose, data)
    for i in range(max_iterations):
        distances, indices = search_nearest_neighbors(data[:m, :].T, model[:m, :].T)
        mean_distance_errors.append(np.mean(distances))
        error = error_metric(data[:m, :].T, model[:m, :].T, normals[:m, :].T, indices)
        errors.append(error)
        if error < tolerance:
            break

        start = datetime.now()
        transformation = minimize_function(data[:m, :].T, model[:m, :].T, normals[:m, :].T, indices)
        time_spend.append((datetime.now() - start).total_seconds())
        data = np.dot(transformation, data)
        draw_set_points_clouds(model[:m, :].T, data[:m, :].T, output_path, file_name='%s_iteration_%d' % (error_name, i))

    return mean_distance_errors, errors, time_spend


def argument_parse():
    argument_parser = argparse.ArgumentParser(description='Script combine two points cloud in 3 dimensional space '
                                                          'with used algorithm ICP on .ply-files')
    argument_parser.add_argument('--sample_figure',  choices=['ellipsoid, sinusoid, bunny'], default='ellipsoid',
                                 help='Choose sample figure from list for work')
    argument_parser.add_argument('--modification', choices=['translation', 'translation_rotate'], default='translation',
                                 help='apply modification for data set points cloud')
    argument_parser.add_argument('--noise', action='store_true',
                                 help='add noise for data set points cloud')
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


def add_noise(data_set):
    return data_set


def modification_points(type_mod):
    matrix_transportation = np.identity(4)
    if type_mod == ROTATE:
        matrix_transportation = translation_matrix(np.pi / 6, 0, 0)
    matrix_transportation[:, -2:-1] = -0.1
    return matrix_transportation


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
    if args.sample_figure == BUNNY:
        model_set, model_faces = extract_points_cloud(PATH_TO_FILE_BUNNY)
        model_normals = generate_normals(model_set, model_faces)
    else:
        generation_function = make_points_ellipsoid if args.sample_figure == ELLIPSOID else make_points_sinusoid
        model_set, model_normals = generation_function()
    # draw_points_cloud(model_set, args.o, color='r', marker='o')
    data_set = model_set.copy()
    if args.noise:
        data_set = add_noise(data_set)
    init_pose = modification_points(args.modification)
    draw_set_points_clouds(model_set, data_set, args.o, file_name='model_and_data_set')
    p_plane_results = icp(model_set, data_set, model_normals, init_pose, args.o, p_to_plane=True)
    p_point_results = icp(model_set, data_set, model_normals, init_pose, args.o, p_to_plane=False)
    draw_analyze_graphic(p_plane_results[0], p_point_results[0], output_path=args.o,
                         name_graphic='mean distance corresponding points')
    draw_analyze_graphic(p_plane_results[1], p_point_results[1], output_path=args.o, name_graphic='metric_error')
    draw_analyze_graphic(p_plane_results[2], p_point_results[2], output_path=args.o,
                         name_graphic='time_spend', sum_values=True)


if __name__ == '__main__':
    main()
