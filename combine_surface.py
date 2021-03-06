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
PATH_TO_FILE_BUNNY = '/home/vitaliy/media/bunny.ply'
ROTATE = 'rotate'
TRANSLATE = 'translation'
ACCURACY = 0.00001
TRANSLATE_VALUE = [0, 0, -0.08]
OPT_ANGLE = 0.00001


def make_points_surface():
    """
    Генерирует точки и нормали (к точкам) поверхонсти, заданный функцией
    sqrt(x ** 2 * y ** 2) = z на сетке значений x = [0,1], y = [0,1], с шагом 0.015
    :return: Массив координат точек и векторов нормалей
    """
    x = np.arange(-10, 10, 0.3)
    y = np.arange(-10, 10, 0.3)
    xgrid, ygrid = np.meshgrid(x, y)
    zgrid = np.sqrt(xgrid ** 2 + ygrid ** 2)
    df_x = xgrid / np.sqrt(xgrid ** 2 + ygrid ** 2)
    df_y = ygrid / np.sqrt(xgrid ** 2 + ygrid ** 2)
    df_z = np.empty(xgrid.shape)
    df_z[:] = -1
    normals = np.array([df_x.ravel(), df_y.ravel(), df_z.ravel()]).T
    norms = np.linalg.norm(normals, axis=1)
    for i in range(normals.shape[0]):
        normals[i] = normals[i] / norms[i]
    points = np.array([xgrid.ravel(), ygrid.ravel(), zgrid.ravel()]).T
    points = points + np.abs(np.min(points))
    points = points / np.max(points)
    return points.reshape((-1, 3)), normals.reshape((-1, 3))


def make_points_paraboloid(a=0.2, b=0.1):
    """
    Генерирует точки и нормали (к точкам) гиперболического параболоида, заданный функцией
    x ** 2 / a ** 2 - y ** / b ** 2 = 2 * z
    :param a: 1ый параметр
    :param b: 2ой параметр
    :return: Массив координат точек и векторов нормалей
    """
    x = np.arange(-10, 10, 0.3)
    y = np.arange(-10, 10, 0.3)
    xgrid, ygrid = np.meshgrid(x, y)
    zgrid = xgrid ** 2 / (a ** 2 * 4) - ygrid ** 2 / (b ** 2 * 4)
    df_x = xgrid / (a ** 2 * 2)
    df_y = -ygrid / (b ** 2 * 2)
    df_z = np.empty(xgrid.shape)
    df_z[:] = -1
    normals = np.array([df_x.ravel(), df_y.ravel(), df_z.ravel()]).T
    norms = np.linalg.norm(normals, axis=1)
    for i in range(normals.shape[0]):
        normals[i] = normals[i] / norms[i]
    points = np.array([xgrid.ravel(), ygrid.ravel(), zgrid.ravel()]).T
    points = points + np.abs(np.min(points))
    points = points / np.max(points)
    return points.reshape((-1, 3)), normals.reshape((-1, 3))


def make_points_ellipsoid(center=[0.5, 0.5, 0.5], a=0.4, b=0.2, c=0.2):
    """
    Генерирует точки и нормали (к точкам) эллипсоида, заданный параметрически
    x = x_0 + a*sin(v)*cos(u)
    y = y_0 + b*sin(v)*sin(u)
    z = z_0 + c*cos(v)
    :param center: Центр эллипсоида в 3-ой системе координат M_0{x_0, y_0, z_0}
    :param a: Длина 1ой полуоси
    :param b: Длина 2ой полуоси
    :param c: Длина 3ей полуоси
    :return: Массив координат точек и векторов нормалей к точкам
    """
    theta = np.linspace(-np.pi / 2, np.pi / 2, num=60)
    thi = np.linspace(0, 2 * np.pi, num=60)
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
            if v > np.pi:
                normal = -normal
            if np.linalg.norm(normal) == 0:
                continue
            points = np.append(points, np.array([[x, y, z]]))
            normal = normal / np.linalg.norm(normal)
            normals = np.append(normals, normal)
    return points.reshape((-1, 3)), normals.reshape((-1, 3))


def draw_set_points_clouds(model_set_points_cloud, data_set_points_cloud, output_path, file_name='sample_plot'):
    """
    Рисует график размещения исходной и целевой модели в 3мерной системе координат.
    Сохраняет изображение в файл в формате *.png. Отображение моделей производится точками
    :param model_set_points_cloud: Целевая массив точек (N x 3)
    :param data_set_points_cloud: Исходная массив точек (N x 3)
    :param output_path: Путь для сохранения графика
    :param file_name: Название графика
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(model_set_points_cloud[:, 0], model_set_points_cloud[:, 1], model_set_points_cloud[:, 2],
               c='r', marker='o', label='destination model')
    ax.scatter(data_set_points_cloud[:, 0], data_set_points_cloud[:, 1], data_set_points_cloud[:, 2],
               c='b', marker='+', label='source model')
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    ax.legend()
    plt.savefig(os.path.join(output_path, file_name))
    # plt.show()
    plt.close()


def draw_points_cloud(points, output_path, normals=None, color='r', marker='o'):
    """
    Рисует график модели в 3мерной системе координат.
    Сохраняет изображение в файл в формате *.png. Отображение моделей производится точками
    :param points: Массив точек
    :param output_path: Путь для сохранения графика
    :param normals: Режим отображения нормалей, опционально
    :param color: Цвет точек
    :param marker: Маркер точек
    """
    fig = plt.figure()
    ax = Axes3D(fig)
    step = 1
    while len(points[::step]) > 1600:
        step += 1
    ax.scatter(points[::step, 0], points[::step, 1], points[::step, 2], c=color, marker=marker, s=3)
    if normals is not None:
        normals = points + normals / 1200
        for i in range(0, points.shape[0], step):
            line = np.stack((points[i], normals[i]), axis=-1)
            ax.plot(line[0], line[1], line[2], 'g-')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.savefig(os.path.join(output_path, 'model'))
    # plt.show()
    plt.close()


def draw_analyze_graphic(p_plane_results, p_point_results, output_path, name_graphic='plot',
                         label='error (euclid distance)', accuracy=None):
    """
    Рисует анализирующий график значений за каждую итерацию алгоритма.
    Сохраняет изображение в файл в формате *.png.
    :param accuracy: Граничная ошибка
    :param p_plane_results: Результаты выполнения метода "точка-плоскость" по каждой итерации
    :param p_point_results: Результаты выполнения метода "точка-точка" по каждой итерации
    :param output_path: Путь для сохранения графика
    :param name_graphic: Название графика
    :param label: Подпись к оси Ох
    """
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.set(xlabel='iterations', ylabel=label,
            title=name_graphic, xticks=range(0, max(len(p_point_results), len(p_plane_results)) + 1))
    ax1.plot(range(len(p_plane_results)), p_plane_results, color='red', marker='o', linestyle='dashed',
             linewidth=2, markersize=5, label='point_to_plane_minimization')
    ax1.plot(range(len(p_point_results)), p_point_results, color='blue', marker='+', linestyle='dashed',
             linewidth=2, markersize=10, label='point_to_point_minimization')
    if accuracy is not None:
        max_iterations = max(len(p_plane_results), len(p_point_results))
        ax1.plot(range(max_iterations), [accuracy] * max_iterations, color='green')
        plt.text(-0.5, accuracy, r'eps = %f' % accuracy)
    ax1.legend()
    ax1.grid()
    plt.savefig(os.path.join(output_path, name_graphic))
    plt.close()


def search_nearest_neighbors(data_set, model_set):
    """
    Алгоритм поиска ближайшего соседа (точек)
    :param model_set: Целевой массив точек (N x 3)
    :param data_set: Исходный массив точек (N x 3)
    :return: Расстояния и указатели между ближайшими точками
    >>> sample_array = np.array([[1.0, 1.0, 0.0], [2.0, 2.0, 0.0], [3.0, 3.0, 0.0]])
    >>> dest_array = np.array([[2.0, 1.5, 0.0], [3.0, 2.5, 0.0],[1.0, 0.0, 0.0]])
    >>> search_nearest_neighbors(sample_array, dest_array)
    (array([0.5, 0.5, 1. ]), array([1, 2, 0]))
    """
    neighbors = NearestNeighbors(n_neighbors=1, metric='euclidean')
    neighbors.fit(data_set)
    distances, indices = neighbors.kneighbors(model_set, return_distance=True)
    return distances.ravel(), indices.ravel()


def p_to_point_min_func(data, model, normals_model, indices):
    """
    Алгоритм минимизации средне-квадратичной ошибки методом "точка-точка"
    :param data: Исходный массив точек (N x 3)
    :param model: Целевой массив точек (N x 3)
    :param normals_model: Нормали к точкам из целевого массива (N x 3)
    :param indices: Указатели на ближайшие точки
    :return: Матрица преобразования
    """
    assert data.shape == model.shape
    data = data[indices]
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
        R = np.dot(Vt.T, U.T)

    t = centroid_model.T - np.dot(R, centroid_data.T)

    T = np.identity(m + 1)
    T[:m, :m] = R
    T[:m, m] = t

    return T


def translation_matrix(alpha, betta, gamma):
    """
    Формирование матрицы преобразования по заданным углам
    :param alpha: Угол поворота вдоль оси oX
    :param betta: Угол поворота вдоль оси oY
    :param gamma: Угол поворота вдоль оси oZ
    :return: Матрица поворота (4 x 4)
    """
    a_11 = np.cos(gamma) * np.cos(betta)
    a_12 = -np.sin(gamma) * np.cos(alpha) + np.cos(gamma) * np.sin(betta) * np.sin(alpha)
    a_13 = np.sin(gamma) * np.sin(alpha) + np.cos(gamma) * np.sin(betta) * np.cos(alpha)
    a_21 = np.sin(gamma) * np.cos(betta)
    a_22 = np.cos(gamma) * np.cos(alpha) + np.sin(gamma) * np.sin(betta) * np.sin(alpha)
    a_23 = -np.cos(gamma) * np.sin(alpha) + np.sin(gamma) * np.sin(betta) * np.cos(alpha)
    a_31 = -np.sin(betta)
    a_32 = np.cos(betta) * np.sin(alpha)
    a_33 = np.cos(betta) * np.cos(alpha)
    translation = np.array([[a_11, a_12, a_13, 0],
                            [a_21, a_22, a_23, 0],
                            [a_31, a_32, a_33, 0],
                            [0, 0, 0, 1]])
    return translation.reshape(4, 4)


def calculate_m_opt(x_opt):
    """
    Расчет матрицы преобразования по оптимальным параметрам, в случае небольшого угла поворота (< Pi / 6),
    расчет матрицы преобразования упрощается
    :param x_opt: Оптимальные параметры (6)
    :return: Матрица преобразования (4 x 4)
    """
    alpha, betta, gamma, t_x, t_y, t_z = tuple(x_opt)
    theta_ang = np.sum(x_opt[:3])
    if theta_ang < OPT_ANGLE:
        translation = np.array([[1, -gamma, betta, t_x], [gamma, 1, -alpha, t_y], [-betta, alpha, 1, t_z], [0, 0, 0, 1]])
    else:
        translation = translation_matrix(alpha, betta, gamma)
        translation[:-1, -1] = x_opt[3:]
    return translation.reshape(4, 4)


def p_to_plane_min_func(data, model, normals_model, indices):
    """
    Алгоритм минимизации средне-квадратичной ошибки методом "точка-плоскость"
    :param data: Исходный массив точек (N x 3)
    :param model: Целевой массив точек (N x 3)
    :param indices: Указатели на ближайшие точки
    :param normals_model: Нормали к точкам из целевого массива (N x 3)
    :return: Матрица преобразования (4 x 4)
    """
    assert data.shape == model.shape

    m = data.shape[1]

    data = data[indices]

    b = scalar_vectors(model, normals_model) - scalar_vectors(data, normals_model)
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
    """
    Подсчет средне-квадратичной ошибки метода "точка-точка"
    :param data: Исходный массив точек
    :param model: Целевой массив точек
    :param normals_model: Нормали к точкам из целевого массива
    :param indices: Указатели к ближайшим точкам
    :return:
    """
    error = np.sum(np.linalg.norm(data[indices] - model, axis=1) ** 2)
    return error


def error_metric_p_to_plane(data, model, normals_model, indices):
    """
    Подсчет средне-квадратичной ошибки метода "точка-плоскость"
    :param data: Исходный массив точек
    :param model: Целевой массив точек
    :param normals_model: Нормали к точкам из целевого массива
    :param indices: Указатели к ближайшим точкам
    :return:
    """
    error = scalar_vectors(data[indices] - model, normals_model)
    error = sum(error ** 2)
    return error


def scalar_vectors(vectors, normals):
    """
    Скалярное умножение векторов и нормалей
    :param vectors: Массив векторов (N x 3)
    :param normals: Массив нормалей (N x 3)
    :return: Массив скалярных произведений (N)
    """
    scalars = np.empty(vectors.shape[0])
    for i in range(vectors.shape[0]):
        scalars[i] = np.vdot(vectors[i], normals[i])
    return scalars


def icp(model_set_points, data_set_points, normals_model, init_pose, tolerance, output_path, max_iterations=30,
        p_to_plane=False):
    """
    Итеративный алгоритм ближайших точек с применением методов наименьшего квадрата: "точка-точка", "точка-плоскость",
    который совмещает два облака точек путем расчета оптимального преобразования для исходного набора точек
    :param data_set_points: Исходный набор точек, массив точек (N x 3)
    :param model_set_points: Целевой набор точек, массив точек (N x 3)
    :param normals_model: Векторы нормалей к точкам из целевого набора, массив векторов (N x 3)
    :param init_pose: Предварительное преобразование исходного набора точек (смещение, поворот)
    :param tolerance: Допустимое значение ошибки, в случае, если ошибка меньше или равна указанного значения,
    алгоритм заканчивает работу
    :param output_path: Путь для сохранения изображений, совершенные на каждом шаге итерации
    :param max_iterations: Максимальное количество итераций
    :param p_to_plane: Режим использования метода минимизации, в случае его активации,
     применяется метод "точка-плоскость", иначе "точка-точка"
    :return: По каждой итерации возвращается среднее расстояние между ближайшими точками,
    средне-квадратичная ошибка и затраченное время.
    """
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
    prev_error = 0
    draw_set_points_clouds(model[:m, :].T, data[:m, :].T, output_path, file_name='model_and_data_set')
    start = datetime.now()
    for i in range(max_iterations):
        distances, indices = search_nearest_neighbors(data[:m, :].T, model[:m, :].T)

        mean_distance = np.mean(distances)
        mean_distance_errors.append(mean_distance)
        error = error_metric(data[:m, :].T, model[:m, :].T, normals[:m, :].T, indices)
        errors.append(error)
        if mean_distance == 0 or abs(prev_error - mean_distance) <= tolerance / 100:
            break
        prev_error = mean_distance

        transformation = minimize_function(data[:m, :].T, model[:m, :].T, normals[:m, :].T, indices)
        data = np.dot(transformation, data)
        draw_set_points_clouds(model[:m, :].T, data[:m, :].T, output_path,
                               file_name='%s_iteration_%d' % (error_name, i))
        time_spend.append((datetime.now() - start).total_seconds())

    return mean_distance_errors, errors, time_spend


def extract_points_cloud(path_to_file):
    """
    Извлекает координаты точек и полигоны из *.ply файла. Размещает координаты в квадрате [0,1]
    :param path_to_file: Путь к ply-файлу
    :return: Массив точек (N x 3) и полигонов (M x 3) в формате списка точек, образующих полигон
    """
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
    """
    Добавление шума к исходному массиву точек. Случайно выбирается N/100 точек для добавления шума
    :param data_set: Исходный массив точек (N x 3)
    :return: Зашумленный массив точек (N x 3)
    """
    noised_data_set = data_set.copy()
    index = np.random.choice(range(noised_data_set.shape[0]), int(noised_data_set.shape[0] / 100))
    for i in index:
        for j, coordinate in enumerate(noised_data_set[i]):
            noise_value = (coordinate / 10) * np.random.random_sample()
            if i % 2 == 0:
                noise_value = -noise_value
            noised_data_set[i][j] += noise_value
    distances, indices = search_nearest_neighbors(data_set, noised_data_set)
    mean_distance = np.mean(distances) + ACCURACY
    return noised_data_set, mean_distance


def modification_points(type_mod):
    """
    Формирует матрицу преобразования для предварительного изменения положения исходного массива точек
    :param type_mod: Тип преобразования, в случае 'translate_rotate' формируется матрица перемещения и поворота,
    по умолчанию, только перемещенения
    :return: Матрица преобразования (4 x 4)
    """
    matrix_translation = np.identity(4)
    if type_mod == ROTATE:
        matrix_translation = translation_matrix(np.pi / 2, np.pi, np.pi)
    elif type_mod == TRANSLATE:
        matrix_translation[:-1, -1] = TRANSLATE_VALUE
    else:
        matrix_translation = translation_matrix(np.pi / 170, 0, 0)
        matrix_translation[:-1, -1] = TRANSLATE_VALUE
    return matrix_translation


def get_normal(p1, p2, p3):
    """
    Векторное произведение векторов по трем точкам
    :param p1: 1ая точка
    :param p2: 2ая точка
    :param p3: 3я точка
    :return: Вектор нормали
    """
    v1 = p1 - p2
    v2 = p2 - p3
    normal = np.cross(v1, v2)
    return normal


def generate_normals(model_set, model_faces):
    """
    Генерирует нормали точек по прилежащим к ним полигонам
    :param model_set: Целевой массив точек (N x 3)
    :param model_faces: Массив списка указателей на точки, образующие полигон (M x 3)
    :return: Массив нормалей к точкам из целевого массива (N x 3)
    """
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


def extract_sets(sample_figure, output_path):
    """
    Формирует массивы точек для целевой и исходной модели. Выбор модели зависит от входного аргумента,
    где на выбор прилагается следующие модели: эллипсоид, поверхность синусоида, отсканированный кролик,
    по умолчанию эллипсоид
    :param sample_figure: Модель из списка
    :param output_path: Путь до файла для сохранения изображения
    :return: Массивы (N x 3)  иссходных, целевых точек и нормалей (N x 3) к точкам из целевого массива
    """
    if sample_figure == BUNNY:
        model_set, model_faces = extract_points_cloud(PATH_TO_FILE_BUNNY)
        model_normals = generate_normals(model_set, model_faces)
    else:
        generation_function = make_points_ellipsoid if sample_figure == ELLIPSOID else make_points_paraboloid
        model_set, model_normals = generation_function()
    draw_points_cloud(model_set, output_path, color='r', marker='o', normals=None)
    data_set = model_set.copy()
    return model_set, model_normals, data_set


def argument_parse():
    argument_parser = argparse.ArgumentParser(description='Algorithm combine two points cloud in 3 dimensional space '
                                                          'with used algorithm ICP and analyze results')
    argument_parser.add_argument('--sample_figure', choices=['ellipsoid', 'surface', 'bunny'], default='ellipsoid',
                                 help='Choose sample figure from list for work')
    argument_parser.add_argument('--modification', choices=['translation', 'rotate', 'translation_rotate'], default='translation',
                                 help='apply modification for data set points cloud')
    argument_parser.add_argument('--noise', action='store_true',
                                 help='add noise for data set points cloud')
    argument_parser.add_argument('-o', type=str, default='/tmp/compare_results/', help='output path to result')
    args = argument_parser.parse_args()
    return args


def main():
    args = argument_parse()
    directory_name = args.sample_figure + '_' + args.modification
    if args.noise:
        directory_name += '_noised'
    output_path = os.path.join(args.o, directory_name)
    os.makedirs(output_path, exist_ok=True)
    tolerance = ACCURACY
    model_set, model_normals, data_set = extract_sets(args.sample_figure, output_path)
    if args.noise:
        data_set, tolerance = add_noise(data_set)
    init_pose = modification_points(args.modification)
    p_plane_results = icp(model_set, data_set, model_normals, init_pose, tolerance, output_path, p_to_plane=True)
    p_point_results = icp(model_set, data_set, model_normals, init_pose, tolerance, output_path, p_to_plane=False)
    draw_analyze_graphic(p_plane_results[0], p_point_results[0], output_path=output_path,
                         name_graphic='mean distance corresponding points', accuracy=tolerance)
    draw_analyze_graphic(p_plane_results[1], p_point_results[1], output_path=output_path, name_graphic='metric_error')
    draw_analyze_graphic(p_plane_results[2], p_point_results[2], output_path=output_path,
                         name_graphic='time_spend', label='time (seconds)')


if __name__ == '__main__':
    main()
