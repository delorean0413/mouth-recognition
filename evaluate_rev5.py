import numpy as np
import pickle
import csv
import datetime
import datetime
import matplotlib.pyplot as plt


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def save_as_csv(name, data):
    now = datetime.datetime.now()
    filename = "{0}_{1:%Y%m%d-%H%M%S}.csv".format(name, now)
    with open(filename, 'w') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerows(data)


def visualize_vec(data):
    plt.figure()
    for i, frame in enumerate(data):
        for j, point in enumerate(frame):
            plt.quiver(i, j, point[0], point[1], angles='xy', scale=2.0, scale_units='xy', linewidth=.5)
    plt.xlabel('frame')
    plt.ylabel('index of feature point')
    plt.grid()
    plt.draw()
    plt.show()


def main():
    data = []

    # 登録済みデータ
    old = np.array(load_pickle("sample_updown_10.pickle"))
    # old = np.array([[1,1],[1,1]])#[2,2],[4,4]
    #old = np.array([ [[1,1],[1,1]], [[2,2],[2,2]] ])

    # 認証データ
    new = np.array(load_pickle("sample_updown_19.pickle"))
    # new = np.array([[0,0],[1,1],[0,0]])#[0,0],[0,0],[2,2],[4,4]
    #new = np.array([ [[0,0],[0,0]], [[1,1],[1,1]], [[2,2], [2,2]], [[3,3],[3,3]] ])

    # 3次元配列(フレーム数,1フレーム内の特徴点の数,動きベクトルの次元数)
    print("old.shape", old.shape)
    print("new.shape", new.shape)
    assert old.ndim == new.ndim, "登録データと認証対象データの次元が異なる"

    if len(old) > len(new):
        max, min = old, new
    else:
        max, min = new, old

    # 相互相関(時系列をそろえる(時間の頭をそろえる)ために行う)
    for i in range(0, len(max)-len(min)+1):
        dist = 0
        for j in range(0, len(min)):
            dist += np.linalg.norm(max[i + j] - min[j])  # ベクトル間の距離
        data.append(dist)

    n = np.argmin(data)
    print(n)
    
    # 認証処理(時系列をそろえた結果を用いてそれぞれの特徴点ごとに距離を比較)
    auth = []
    for k in range(0, len(min)):
        auth.append(np.linalg.norm(max[n] - min[k], axis=1))
        n += 1

    # 最大値で割って [0, 1] に正規化
    auth = np.array(auth)
    max_value_of_auth = auth.max()
    auth /= max_value_of_auth

    #オプティカルフローを数値で表現
    data_norm = []
    dist_norm = 0
    for no in range(0, len(old)):
        data_norm.append(np.linalg.norm(old[no], axis = 1))  # ベクトル間の距離
        
    print(data_norm[0])

    # ベクトルの可視化
    visualize_vec(old)
    #visualize_vec(new)
    # 認証結果の1フレーム目の結果表示
    print(auth[0])

    # ファイル出力
    save_as_csv('outputs/output_true', auth)
    save_as_csv('outputs/vec_true', old)
    save_as_csv('outputs/vec_challenge', new)
    save_as_csv('outputs/old_norm',data_norm)


if __name__ == "__main__":
    main()
