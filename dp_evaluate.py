import numpy as np
import pickle
import csv
import datetime
import matplotlib.pyplot as plt
import sys


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
            plt.quiver(i, j, point[0], point[1], angles='xy',
                       scale=2.0, scale_units='xy', linewidth=.5)
    plt.xlabel('frame')
    plt.ylabel('index of feature point')
    plt.grid()
    plt.draw()
    plt.show()


def main():
    ZurePenalty = 1  # 1文字ずれたことへのペナルティ
    AwazuPenalty = 5  # 1文字不一致へのペナルティ
    Distance = 0  # 2つの文字列の不一致度
    LengthA = 0  # Aの長さ
    LengthB = 0  # Bの長さ

    dtemp1 = 0
    dtemp2 = 0
    dtemp3 = 0

    LenAB = 0

    # 登録済みデータ
    old = np.array(load_pickle(sys.argv[1]))

    # 認証データ
    new = np.array(load_pickle(sys.argv[2]))

    # 3次元配列(フレーム数,1フレーム内の特徴点の数,動きベクトルの次元数)
    print("old.shape", old.shape)
    print("new.shape", new.shape)
    assert old.ndim == new.ndim, "登録データと認証対象データの次元が異なる"

    if len(old) > len(new):
        max, min = old, new
    else:
        max, min = new, old

    """
    # 相互相関(時系列をそろえる(時間の頭をそろえる)ために行う)
    for i in range(0, len(max)-len(min)+1):
        dist = 0
        for j in range(0, len(min)):
            dist += np.linalg.norm(max[i + j] - min[j])  # ベクトル間の距離
        data.append(dist)

    n = np.argmin(data)
    print(n)
    """

    # DPマッチング
    print("Input StringA", max.shape)
    print("Input StringB", min.shape)

    LengthA = len(max)
    LengthB = len(min)
    print(LengthA)
    print(LengthB)
    # 一致結果バッファ  int MissMatch[64][64];
    #print(sum(len(v) for v in min))
    Length_elem = sum(len(v) for v in min[0])
    print("length_elem/2",Length_elem/2)
    MissMatch = [[0 for i in range(LengthB+1)] for j in range(LengthA+1)] #サンプル動作
    #MissMatch = [[[0 for i in range(int(Length_elem/2)+1)] for j in range(LengthB+1)]for k in range(LengthA+1)]

    # 総当たりで一致の確認
    for i, mA in enumerate(max):  # range(0,LengthA)  for(i = 0; i < LengthA; i++)
        # print(i+1)
        #print('i', i)
        for j, mI in enumerate(min):  # for(j = 0; j < LengthB; j++)
            
            #diff = np.linalg.norm(mA-mI) # 距離の定義　サンプル動作
            diff = 0
            for k in range(0, np.min([len(mA), len(mI)])):
                #print("mA(k):", np.linalg.norm(mA[k]))
                #print("mI(k):", np.linalg.norm(mI[k]))
                if np.min([np.linalg.norm(mA[k]), np.linalg.norm(mI[k])]) < 1e-3:
                    diff += 0
                else:
                    diff += np.dot(mA[k],mI[k]) / (np.linalg.norm(mA[k]) * np.linalg.norm(mI[k]))

            MissMatch[i][j] = diff / len(mI)            # [-1, 1]
            MissMatch[i][j] = 2 - (MissMatch[i][j] + 1) # [0, 2]
            """
            if(mA == mI):
                MissMatch[i][j] = 0
            else:
                MissMatch[i][j] = 1
            """
    #print("mA:",mA.shape)
    #print("mI:",mI.shape)
    #print("i",i)
    #print("j",j)
    
    print("\n")

    # 各経路点の到達コスト
    Cost = [[0 for i in range(LengthB+1)] for j in range(LengthA+1)]
    # 最短距離経路はどこから来たか 0:斜め 1:i増え 2:j増え
    From = [[0 for i in range(LengthB+1)] for j in range(LengthA+1)]
    # コスト計算
    Cost[0][0] = MissMatch[0][0] * AwazuPenalty
    From[0][0] = 0

    # i側の縁
    for i in range(1, LengthA):
        Cost[i][0] = Cost[i-1][0] + ZurePenalty + \
            MissMatch[i][0] * AwazuPenalty
        From[i][0] = 1

    # j側の縁
    for j in range(1, LengthB):
        Cost[0][j] = Cost[0][j-1] + ZurePenalty + \
            MissMatch[0][j] * AwazuPenalty
        From[0][j] = 2

    # 中間部
    for i in range(1, LengthA):
        for j in range(1, LengthB):
            # 斜めで来た場合のコスト
            dtemp1 = Cost[i-1][j-1] + MissMatch[i][j] * AwazuPenalty  
            # i増えで来た場合のコスト 
            dtemp2 = Cost[i-1][j] + MissMatch[i][j] * AwazuPenalty + ZurePenalty
            # j増えで来た場合のコスト 
            dtemp3 = Cost[i][j-1] + MissMatch[i][j] * AwazuPenalty + ZurePenalty

            #print(i, ": ", dtemp1, dtemp2, dtemp3)

            if(dtemp1 <= dtemp2 and dtemp1 <= dtemp3):
                Cost[i][j] = dtemp1
                From[i][j] = 0
            elif (dtemp2 <= dtemp3):
                Cost[i][j] = dtemp2
                From[i][j] = 1
            else:
                Cost[i][j] = dtemp3
                From[i][j] = 2
    
    Distance = Cost[LengthA-1][LengthB-1]  # DPマッチングの不一致度はこれ。以降は結果観察のための整形手続き

    # ゴールからスタートへ逆にたどる
    LenAB = LengthA + LengthB
    s = LengthA-1
    g = LengthB-1

    ResultA = [[0 for i in range(LengthB+1)]
               for j in range(LengthA+1)]
    ResultB = [[0 for i in range(LengthB+1)]
               for j in range(LengthA+1)]

    for k in range(LenAB, 0, -1):  
        ResultA.append(max[s])  # ResultA[k] = max[s]
        ResultB.append(min[g])  # ResultB[k] = min[s]

        if(From[s][g] == 0):
            s -= 1
            g -= 1
        elif(From[s][g] == 1):
            s -= 1
        elif(From[s][g] == 2):
            g -= 1
        else:
            print("Error")
        if(s >= 0 and g >= 0):
            break
    LenAB -= k

    for i in range(0, LenAB):
        ResultA[i] = ResultA[i+k+1]
        ResultB[i] = ResultB[i+k+1]

    ResultA[LenAB] = ResultB[LenAB] = '\0'

    print("===Matching Result===")
    print("Difference = ", Distance)

    for i in range(0, LengthA):
        print(i+1, ": ", end="")
        for j in range(0, LengthB):
            if(From[i][j] == 0):
                print("\\", end=" ")
            elif(From[i][j] == 1):
                print("|", end=" ")
            elif(From[i][j] == 2):
                print("-", end=" ")
        print("")

    #print("A: %s",ResultA)
    #print("B: %s",ResultB)

    # DP

    """
    # 認証処理(時系列をそろえた結果を用いてそれぞれの特徴点ごとに距離を比較)
    auth = []
    for k in range(0, len(min)):
        auth.append(np.linalg.norm(max[n] - min[k], axis=1))
        n += 1

    # 最大値で割って [0, 1] に正規化
    auth = np.array(auth)
    max_value_of_auth = auth.max()
    auth /= max_value_of_auth

    # ベクトルの可視化
    visualize_vec(old)

    # 認証結果の1フレーム目の結果表示
    print(auth[0])

    # ファイル出力
    save_as_csv('outputs/output_true', auth)
    save_as_csv('outputs/vec_true', old)
    save_as_csv('outputs/vec_challenge', new)
    """


if __name__ == "__main__":
    main()
