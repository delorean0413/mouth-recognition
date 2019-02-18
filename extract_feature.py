from scipy import ndimage
import cv2
import os
import time
import numpy as np
import threading
import time
import datetime
import pickle
import sys

# 特徴点の数
N_HORIZONTAL_POINTS = 6 # 横の配置
N_VERTICAL_POINTS   = 4 # 縦の配置

# 特徴量データの保存ディレクトリ
SAVE_DIR = "./extracted_features"

class DumpTh:
    def __init__(self, interval=5, name="test"):
        self.interval = interval
        self.name = name
        self.data = []
        self.timer = threading.Timer(interval, self.save)
        self.timer.start()

    def save(self):
        path = os.path.join(SAVE_DIR, self.name + ".pickle")
        with open(path, "wb") as fp:
            pickle.dump(self.data, fp)
        self.timer = threading.Timer(self.interval, self.save)
        self.timer.start()

    def append(self, elm):
        self.data.append(elm)

    def stop(self):
        self.timer.cancel()


def main():
    # コマンドライン引数
    if len(sys.argv) != 2:
        print("Usage: {} <video file|device index>".format(sys.argv[0]))
        exit(-1)

    video_src = sys.argv[1]
    if video_src.isdigit():
        # カメラ入力
        recording_video = True
        video_src = int(video_src)
    else:
        recording_video = False

    # VideoCapture
    cap = cv2.VideoCapture(video_src)
    if recording_video:
        cap.set(cv2.CAP_PROP_FPS, 30)

    # カスケード分類器の特徴量を取得する
    cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    mouth_cascade = cv2.CascadeClassifier('haarcascade_mcs_mouth.xml')

    # 最初のフレーム
    faces = []
    gray = []
    while(len(faces) == 0): # 顔が検出されるまで
        ret, frame = cap.read()
        if ret == False:
            print("End of File")
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    
    # Video出力
    if recording_video:
        outfile = '{}.avi'.format(datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
        fps = cap.get(cv2.CAP_PROP_FPS)
        codecs = 'XVID'
        height, width, ch = frame.shape
        fourcc = cv2.VideoWriter_fourcc(*codecs)
        writer = cv2.VideoWriter(outfile, fourcc, fps, (width, height))

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

        mouth = mouth_cascade.detectMultiScale(face_gray, scaleFactor=1.1, minNeighbors=10)
        try:
            # 口部領域を特徴点として追加
            mx, my, mw, mh = mouth[0]
            mx += x # 口部の左上x
            my += y # 口部の左上y
            
            print("detected mouth:", mx, my, mw, mh)

            #特徴点数と配置
            margin_x = mw // N_HORIZONTAL_POINTS
            margin_y = mh // N_VERTICAL_POINTS
            for yi in range(0, N_VERTICAL_POINTS):
                for xi in range(0, N_HORIZONTAL_POINTS):
                    prevPts.extend([[[mx + xi*margin_x, my + yi*margin_y]]])

        except IndexError:
            pass

    cv2.imwrite('frame.png', frame)

    # 特徴点の配列をnumpy.ndarrayに変換
    prevPts = np.array(prevPts, dtype=np.float32)

    # 図示用カラー生成
    color = np.random.randint(0, 255, (100, 3))#mw * mh

    # フレームごとの処理
    pre_frame, pre_gray = frame, gray

    # 保存処理用スレッド
    if recording_video:
        dump_name = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    else:
        dump_name = os.path.basename(video_src).split(".")[0]
    dumper = DumpTh(name=dump_name)
    
    while(cap.isOpened()):
        ret, frame = cap.read()
        if not ret:
            break
        
        if recording_video:
            writer.write(frame) # 動画書き出し
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

        nextPts, status, err  = cv2.calcOpticalFlowPyrLK(
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
            a, b = new.ravel() # 移動先の点座標
            c, d = old.ravel() # 移動元の点座表
            frameVector.append(np.array([a-c, b-d]))
            # 動きベクトルと特徴点の描画
            mask = cv2.line(mask, (a,b), (c,d), color[k].tolist(), 2)
            frame = cv2.circle(frame, (a,b), 2, color[k].tolist(), -1)
        img = cv2.add(frame, mask)
        cv2.imshow('frame', img)

        # 動きベクトルを保存用に追加
        dumper.append(frameVector)
        frameVector = []

        # Now update the previous frame and previous points
        pre_gray = gray.copy()
        prevPts = good_new.reshape(-1, 1, 2)
        
        if cv2.waitKey(33) & 0xff == ord('q'):
            break
    
    if recording_video:
        writer.release()
    dumper.stop()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    print('OpenCV:', cv2.__version__)
    main()


