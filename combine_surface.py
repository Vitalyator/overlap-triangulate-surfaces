import numpy as np
import pandas as pd
from pyntcloud import PyntCloud
from datetime import datetime

def icp(a_set, b_set):
    pass


def main():
    points = pd.DataFrame(columns=['x', 'y', 'z'])
    with open('/home/vitaliy/media/bunny.ply', 'rb') as ply_file:
        t_start = datetime.now()
        # contains = ply_file.readline().split()
        # while b'f' not in contains:
        #     print(contains)
        #     if b'vertices' in contains:
        #         n_vertices = int(contains[-1].decode())
        #     if b'v' in contains:
        #         x, y, z = tuple(contains[1:])
        #         append_df = pd.DataFrame([[x.decode(), y.decode(), z.decode()]], columns=['x', 'y', 'z'])
        #         points = points.append(append_df, ignore_index=True)
        #     contains = ply_file.readline().split()
        for line in ply_file.readlines():
            if b'vertices' in line:
                n_vertices = int(line.decode().split()[-1])
                continue
            print(line)
            if b'v' in line:
                x, y, z = line.decode().split()[1:]
                append_df = pd.DataFrame([[x, y, z]], columns=['x', 'y', 'z'])
                points = points.append(append_df, ignore_index=True)
    t_finish = datetime.now()
    print(points)
    print(t_finish - t_start)
    icp(None, None)


if __name__ == '__main__':
    main()
