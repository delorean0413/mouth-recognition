import numpy as np
import pickle
import csv
def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def main():
    data = []

    # 登録済みデータ
    old = np.array(load_pickle("vec_true.pickle"))
    #old = np.array([[1,1],[1,1]])#[2,2],[4,4]
    #old = np.array([ [[1,1],[1,1]], [[2,2],[2,2]] ])

    # 認証データ
    new = np.array(load_pickle("vec_challenge.pickle"))
    #new = np.array([[0,0],[1,1],[0,0]])#[0,0],[0,0],[2,2],[4,4]
    #new = np.array([ [[0,0],[0,0]], [[1,1],[1,1]], [[2,2], [2,2]], [[3,3],[3,3]] ])
    
    #3次元配列(フレーム数,1フレーム内の特徴点の数,動きベクトルの次元数)
    print("old.shape", old.shape)
    print("new.shape", new.shape)
    assert old.ndim == new.ndim, "登録データと認証対象データの次元が異なる"

    if len(old) > len(new):
        max, min = old, new
    else:
        max, min = new, old
    #print("max")
    #print(max[0])
    #print("min")
    #print(min)

    #相互相関(時系列をそろえる(時間の頭をそろえる)ために行う) 
    for i in range(0, len(max)-len(min)+1):
        dist = 0
        for j in range(0, len(min)):
            dist += np.linalg.norm(max[i + j] - min[j]) # ベクトル間の距離
        data.append(dist)

    n = np.argmin(data)
    print(n)

    #認証処理(時系列をそろえた結果を用いてそれぞれの特徴点ごとに距離を比較)
    auth = []
    for k in range(0, len(min)):
        auth.append(np.linalg.norm(max[n] - min[k], axis=1))
    
    #認証結果の1フレーム目の結果表示
    print(auth[0])

    f = open('output_prestudy.csv','w')
    writer = csv.writer(f, lineterminator='\n')

    #csvlist = []
    #csvlist.append(auth)

    writer.writerows(auth)

    f.close()
    
    f = open('vec_true_prestudy.csv','w')
    writer = csv.writer(f, lineterminator='\n')

    #csvlist = []
    #csvlist.append(auth)

    writer.writerows(old)

    f.close()

    f = open('vec_challenge_prestudy.csv','w')
    writer = csv.writer(f, lineterminator='\n')

    #csvlist = []
    #csvlist.append(auth)

    writer.writerows(new)

    f.close()

if __name__ == "__main__":
    main()