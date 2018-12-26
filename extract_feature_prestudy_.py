from scipy import ndimage
import cv2
import time
import numpy as np
import threading
import time
import pickle
import sys
import copy  # snakeで使用？
import math
import sympy.geometry as sg


class DumpTh:
    def __init__(self, interval=5):
        self.interval = interval
        self.data = []
        self.timer = threading.Timer(interval, self.save)
        self.timer.start()

    def save(self):
        with open("vec_challenge_prestudy.pickle", "wb") as fp:
            pickle.dump(self.data, fp)
        self.timer = threading.Timer(self.interval, self.save)
        self.timer.start()

    def append(self, elm):
        self.data.append(elm)

    def stop(self):
        self.timer.cancel()


#"""
#パラメータ
gamma = 10

#口唇の上点で使用
test = cv2.imread("frame.png", cv2.IMREAD_COLOR)#BGRなので気をつける "./Img/input.jpg"

gray_test = cv2.imread("frame.png",cv2.IMREAD_GRAYSCALE)
height = test.shape[0]
width = test.shape[1] 
#画像の書き出し
cv2.imwrite('test.png', test)
cv2.imwrite('test.png',gray_test)

N = 2000 #頂点数

v = np.zeros((N,2))
start_v = np.zeros((N,2))
vec_g = np.zeros(2)
for i in range(0,N):
    if(i<N/4):
        v[i] = [height/(N/4)*i,0]
    elif(i<2*N/4):
        v[i] = [height-1,width/(N/4)*(i-N/4)]
    elif(i<3*N/4):
        v[i] = [height-1-height/(N/4)*(i-2*N/4),width-1]
    else:
        v[i] = [0,width-1 - width/(N/4)*(i-3*N/4)]

start_v = copy.deepcopy(v)
display = copy.deepcopy(test)

"""
#"""
#パラメータ
gamma = 10
width = 0
height = 0

def EpsEx(vec0,pix):#gray
    global width,height
    value = 0
    x = int(vec0[0])
    y = int(vec0[1])

    if(x+1 >= height or y+1 >= width):
        return float('inf') 
    else:
        I = [abs(int(pix[x+1,y]) - int(pix[x,y])) ,abs(int(pix[x,y+1])-int(pix[x,y]))]
        value = -gamma*np.linalg.norm(I)**2
#         print("Ex:"+str(value))
        return value

def Energy(vec0,pix):
    value = 0
    value = EpsEx(vec0,pix)
#     print("Result:"+str(value))
    return value

#"""


def distance(point_a, point_b):
    """
    2点間の距離を返す
    Args:
        point_a: 始点
        point_b: 終点
    """
    side_a = np.asarray(point_a[0])
    side_b = np.asarray(point_b[0])
    side_vec = side_a - side_b
    side_vec = np.array(side_vec, dtype='float64')
    side = np.linalg.norm(side_vec)
    return side


def cross(a, b):
    """
    線分の交点を返す
    """
    line_a = sg.Line(sg.Point(a[0][0], a[0][1]), sg.Point(a[1][0], a[1][1]))
    line_b = sg.Line(sg.Point(b[0][0], b[0][1]), sg.Point(b[1][0], b[1][1]))
    result = line_a.intersection(line_b)
    return result


def score(prevPts):
    # prevPts = [0:左, 1:右, 2:下, 3:上]
    # 左右の線分
    a = [prevPts[0][0], prevPts[1][0]]
    # 上下の線分
    b = [prevPts[2][0], prevPts[3][0]]

    crosspoint = cross(a, b)
    #print("crosspoint", crosspoint)
    updistance = distance(prevPts[3], crosspoint)
    downdistance = distance(prevPts[2], crosspoint)
    leftdistance = distance(prevPts[0], crosspoint)
    rightdistance = distance(prevPts[1], crosspoint)
    rad_1 = math.atan(rightdistance/updistance)
    distance1 = math.degrees(rad_1) * 2

    rad_2 = math.atan(rightdistance/downdistance)
    distance2 = math.degrees(rad_2) * 2

    side3 = distance(prevPts[0], prevPts[3])
    rad_3 = math.cos(leftdistance/side3)
    distance3 = math.degrees(rad_3)

    side4 = distance(prevPts[0], prevPts[2])
    rad_4 = math.cos(downdistance/side4)
    distance4 = math.degrees(rad_4)

    distance5 = distance(prevPts[2], prevPts[3])
    distance6 = distance(prevPts[0], prevPts[1])

    similarity = distance1*distance2*distance3*distance4*distance5*distance6
    return similarity


def main():
    global width, height
    # コマンドライン引数
    if len(sys.argv) != 2:
        print("Usage: {} <video file|device index>".format(sys.argv[0]))
        exit(-1)

    video_src = sys.argv[1]
    if video_src.isdigit():
        video_src = int(video_src)

    cap = cv2.VideoCapture(video_src)

    # カスケード分類器の特徴量を取得する
    cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    mouth_cascade = cv2.CascadeClassifier('haarcascade_mcs_mouth.xml')

    faces = []
    gray = []
    while(len(faces) == 0):  # 顔が検出されるまで
        ret, frame = cap.read()
        if ret == False:
            print("End of File")
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        value_image = hsv[:, :, 2]
        #cv2.imshow("value image",value_image)
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        a_image = lab[:, :, 1]
        faces = cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5)  # minneighbors:顔の検出精度パラメータ

    prevPts = []
    for (x, y, w, h) in faces:
        print("detected face:", x, y, w, h)
        face_gray = gray[y:y+h, x:x+w]
        face_color = frame[y:y+h, x:x+w]

        # 顔領域に対して特徴点抽出
        # Ref: http://docs.opencv.org/3.1.0/dd/d1a/group__imgproc__feature.html
        """
        pts = cv2.goodFeaturesToTrack(face_gray, mask=None,
            maxCorners=40, qualityLevel=0.4, minDistance=8, blockSize=8)
        pts[:,:,0] += x
        pts[:,:,1] += y
        prevPts.extend(pts.tolist())
        """
        # 口唇領域抽出
        mouth = mouth_cascade.detectMultiScale(
            face_gray, scaleFactor=1.1, minNeighbors=10)

        try:
            # 口部領域を特徴点として追加
            mx, my, mw, mh = mouth[0]
            mx += x
            my += y

            # 口唇領域の明度画像
            mouth_value_img = value_image[my:my+mh, mx:mx+mw]
            mouth_value_color_img = frame[my:my+mh, mx:mx+mw]  # 口唇領域のカラー画像
            cv2.imwrite("mouth_value.png", mouth_value_img)
            ret, mouth_binary_img = cv2.threshold(
                mouth_value_img, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)  # cv2.adaptivethreshold
            print("threshold: ", ret)
            cv2.imwrite("mouth_binary.png", mouth_binary_img)
            print(mouth_binary_img[0, 0])  # ???

            # 口唇の左/右端の取得
            num, labels, stats, centroids = cv2.connectedComponentsWithStats(
                mouth_binary_img)
            print("stats:", stats)
            region = stats[1]  # 二つ目の抽出領域
            # region[cv2.CC_STAT_LEFT] # 左端のx座標
            # region[cv2.CC_STAT_TOP] #上端のy座標
            # region[cv2.CC_STAT_WIDTH] #幅
            # region[cv2.CC_STAT_HEIGHT] #高さ
            prevPts.extend(
                [[[mx+region[cv2.CC_STAT_LEFT], my+region[cv2.CC_STAT_TOP]]]])  # 左
            prevPts.extend(
                [[[mx+region[cv2.CC_STAT_LEFT]+region[cv2.CC_STAT_WIDTH], my+region[cv2.CC_STAT_TOP]]]])  # 右

            # THRESH_BINARY_"INV":INVERSEは逆の意味を持っているためBINARYの逆を表示している

            # 口唇の下点取得
            mouth_under_value_img = a_image[my:my+mh, mx:mx+mw]
            cv2.imwrite("mouth_under_value.png", mouth_under_value_img)
            ret2, mouth_under_binary_img = cv2.threshold(
                mouth_under_value_img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            print("threshold2: ", ret2)
            cv2.imwrite("mouth_under_binary.png", mouth_under_binary_img)
            print("???", mouth_under_binary_img[0, 0])  # ???
            #label_under = cv2.connectedComponentsWithStats(mouth_under_binary_img)
            num_under, labels_under, stats_under, centroids_under = cv2.connectedComponentsWithStats(
                mouth_under_binary_img)
            print("stats_under", stats_under)
            print("centroids_under", centroids_under)
            region_under = stats_under[1]
            center_under = centroids_under[1]
            np.set_printoptions(threshold=np.inf)
            # print("labels_under:",labels_under)
            for i in range(region_under[cv2.CC_STAT_HEIGHT]):
                for j in range(region_under[cv2.CC_STAT_WIDTH]):
                    if labels_under[i][j] == 1:
                        max_y = i
                        max_x = j
                        break
            for k in range(j, region_under[cv2.CC_STAT_WIDTH]):
                if labels[max_y][k] != 1:
                    break
            max_x = (max_x + k)/2
            prevPts.extend(
                [[[mx+region_under[cv2.CC_STAT_LEFT]+max_x, my+region_under[cv2.CC_STAT_TOP]+max_y]]])
            cv2.imwrite("labels_under.png", labels_under*40)

            #"""
            #口唇の上点
            
            #test = cv2.imread("mouth_value.png", cv2.IMREAD_COLOR)#BGRなので気をつける "./Img/input.jpg"
            #gray_test = cv2.imread("mouth_value.png",cv2.IMREAD_GRAYSCALE)
            test = mouth_value_color_img
            gray_test = mouth_value_img
            height = test.shape[0]
            width = test.shape[1]
            size = height * width
            #画像の書き出し
            #cv2.imwrite('mouth_value.png', test)
            #cv2.imwrite('mouth_value.png',gray_test)
            
            #N = n1+n2
            #sgb = n1*(P1-Pm)*(P1-Pm)+n2*(P2-Pm)*(P2-Pm)
            #for i in range(1,N)
                #sgT = (P[i]-Pm)*(P[i]-Pm)
            #eata = sgb / sgT
            

            N = 600 #頂点数

            v = np.zeros((N,2))
            start_v = np.zeros((N,2))
            vec_g = np.zeros(2)
            for i in range(0,N):
                if(i<N/4):
                    v[i] = [height/(N/4)*i,0]
                elif(i<2*N/4):
                    v[i] = [height-1,width/(N/4)*(i-N/4)]
                elif(i<3*N/4):
                    v[i] = [height-1-height/(N/4)*(i-2*N/4),width-1]
                else:
                    v[i] = [0,width-1 - width/(N/4)*(i-3*N/4)]

            start_v = copy.deepcopy(v)
            display = copy.deepcopy(test)

            """
            # def EpsEx
            #value = 0
            #vec0 = []
           #pix = []
            #x = int(vec0[0])
            #y = int(vec0[1])

            # if(x+1 >= height or y+1 >= width):
            # return float('inf')
            # else:
            #I = [abs(int(pix[x+1,y]) - int(pix[x,y])) ,abs(int(pix[x,y+1])-int(pix[x,y]))]
            # value = -gamma*np.linalg.norm(I)**2
        #         print("Ex:"+str(value))
            """

            #探索
            n = 1500
            dx = [-1, -1, -1, 0, 0, 0, 1, 1, 1]
            dy = [1, 0, -1, 1, 0, -1, 1, 0, -1]
            # dx = [1,1,1,0,0,0,-1,-1,-1]
            # dy = [1,-1,0,1,-1,0,1,-1,0]
            #210
            #543
            #876

            #i → m j →p

            flag = 4
            for loop in range(0,n):
                for m in range(0,N):
                    flag = 4
                    eps_min = float('inf') 
                    vec_g = [0,0]

                    #重心中心にするならこれ
            #         for j in range(0,N):
            #             vec_g += [v[j,0],v[j,1]]

                    for p in range(0,9):            
                        move  = [v[m,0]+dx[p], v[m,1]+dy[p]]
                        if(move[0] < 0 or move[1] < 0 or move[0] >= height  or move[1] >= width):
                            continue #はみ出し処理

                        #重心中心にするならこれ
                        #vec_g += [dx[j],dy[j]]
                        #vec_g =[vec_g[0]/N, vec_g[1]/N]
                        #画像中心を基準に
                        #vec_g = [int(height/2),int(width/2)]
                        energy = Energy(move,gray_test)
                        
                      
                        if(eps_min>energy):
                            eps_min = energy
                            flag = p
                    v[m] += [dx[flag],dy[flag]]
                if(loop%10==0):
                    #cv2.imwrite(str(loop)+'.png', display) #'./Img/result'
                    display = copy.deepcopy(test)
                    for m in range(0,N):
                        cv2.line(display, (int(v[m,1]),int(v[m,0])), (int(v[(m+1)%N,1]),int(v[(m+1)%N,0])), (0, 255, 0), 2)

                        
                    
            for i in range(0,N):
                cv2.line(display, (int(v[i,1]),int(v[i,0])), (int(v[(i+1)%N,1]),int(v[(i+1)%N,0])), (0, 255, 0), 2)

            for i in range(0,N):
                cv2.line(display, (int(start_v[i,1]),int(start_v[i,0])), (int(start_v[(i+1)%N,1]),int(start_v[(i+1)%N,0])), (255, 0, 0), 2)

            #cv2.line(display, (int(v[i,1]),int(v[i,0])), (int(v[(i+1)%N,1]),int(v[(i+1)%N,0])), (0, 255, 0), 2)
            cv2.imwrite('mouth_value2.png', display) #./Img/input.jpg
            #"""
            #center_under = np.delete(label_under[3],0,0)
            # print("領域の個数:",n)
            # print("領域の高さ:",data_under[:,3])
            # print("領域の中心座標:",center_under[:,0])

            print("detected mouth:", mx, my, mw, mh)
            # for y in range(my, my+mh+1,mh//2):
            # for x in range(mx, mx+mw+1, mw//2):
            #prevPts.extend([[[x, y]]])
            # prevPts.extend([[[(mw//2)+mx, my]]]) #上唇
            prevPts.extend(
                [[[mx+region_under[cv2.CC_STAT_LEFT]+max_x, my+region_under[cv2.CC_STAT_TOP]]]])  # 上唇
            #prevPts.extend([[[mx, (mh//2)+my]]])
            # prevPts.extend([[[mx+left_x[0], my+middle_y[0]]]]) #左
            #prevPts.extend([[[mx+mw+1, (mh//2)+my]]])
            # prevPts.extend([[[mx+right_x[0], my+middle_y[0]]]])#右
            #prevPts.extend([[[center_under[:,0]+mx, my+data_under[:,3]]]])
            print("prevPts:", prevPts)
            mouthsize = mouth
        except IndexError:
            pass
    # for y in range(my, my+mh+1, mh//5):for x in range(mx, mx+mw+1, mw//5):
    cv2.imwrite('frame.png', frame)

    # 特徴点の配列をnumpy.ndarrayに変換
    prevPts = np.array(prevPts, dtype=np.float32)

    # 図示用カラー生成
    color = np.random.randint(0, 255, (100, 3))  # mw * mh

    # フレームごとの処理
    pre_frame, pre_gray = frame, gray

    # 保存処理用スレッド
    dumper = DumpTh()

    while(cap.isOpened()):
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        """
        faces = cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=8)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            face_gray = gray[y:y+h, x:x+w]
            face_color = frame[y:y+h, x:x+w]
            #mouth = mouth_cascade.detectMultiScale(face_gray)
            try:
                #mx, my, mw, mh = mouth[0]
                mx, my, mw, mh = mouthsize[0]
                cv2.rectangle(face_color, (mx, my), (mx + mw, my + mh), (0, 255, 0), 2)
            except IndexError:
                pass
        """

        nextPts, status, err = cv2.calcOpticalFlowPyrLK(
            pre_gray, gray, prevPts, None, winSize=(24, 24), maxLevel=3
        )
        # Select good points
        """
        print(len(status))
        try:
            good_new = nextPts[status==1]
            good_old = prevPts[status==1]
        except:
            break
        """
        good_new = nextPts
        good_old = prevPts

        # draw the tracks
        mask = np.zeros_like(frame)
        frameVector = []

        for k, (new, old) in enumerate(zip(good_new, good_old)):
            # 動きベクトル
            a, b = new.ravel()  # 移動先の点座標
            c, d = old.ravel()  # 移動元の点座標
            frameVector.append(np.array([a-c, b-d]))
            # 動きベクトルと特徴点の描画
            mask = cv2.line(mask, (a, b), (c, d), color[k].tolist(), 2)
            frame = cv2.circle(frame, (a, b), 2, color[k].tolist(), -1)
        img = cv2.add(frame, mask)
        cv2.imshow('frame', img)

        sc = score(prevPts)
        print("score",sc)

        # 動きベクトルを保存用に追加
        dumper.append(frameVector)
        frameVector = []

        # Now update the previous frame and previous points
        pre_gray = gray.copy()
        prevPts = good_new.reshape(-1, 1, 2)

        if cv2.waitKey(33) & 0xff == ord('q'):
            break

    dumper.stop()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    print('OpenCV:', cv2.__version__)
    main()
