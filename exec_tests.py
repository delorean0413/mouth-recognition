"""
    全データの比較を実行するプヨグラム
"""
import os
import re
import csv


def main():
    N = 13 #「あ」「ん」、21　実験2 20
    data_dir = "./data/"
    output_dir = "./results/"
    results = [[-1 for j in range(N)] for i in range(N)]

    for i in range(1, N + 1):
        true_file = data_dir + "anan_true_{}.pickle".format(i)#vec_true_{}.pickle
        for j in range(1, N + 1):
            test_file = data_dir + "anan_challenge_{}.pickle".format(j)#vec_test_{}.pickle
            output_file = output_dir + "result_anan_{}_{}.txt".format(i, j)
            command = "python ./dp_evaluate.py {} {} > {}".format(
                true_file, test_file, output_file)
            print("running : ", command)
            os.system(command)

            with open(output_file) as f:
                for line in f.readlines():
                    m = re.search(r"Difference = *([0-9\.]+)", line)
                    if m:
                        diff = m.group(1)
                        break

            results[i-1][j-1] = diff
            print(results)

    with open(output_dir + "anan.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["", *["challenge_{}".format(i) for i in range(1, N + 1)]])
        for index, row in enumerate(results):
            writer.writerow(["true_{}".format(index+1), *row])


if __name__ == "__main__":
    main()
