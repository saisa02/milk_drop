import os,sys
import cv2
import copy
import numpy as np

milk_drop = "C:\image_processing\milkdrop.bmp"

#画像入力
img = cv2.imread(milk_drop, 1)

#表示用関数
def imageshow(window_name, img_data):
    cv2.namedWindow(window_name, cv2.WINDOW_GUI_NORMAL)
    cv2.imshow(window_name, img_data)
    cv2.waitKey(-1)
    cv2.destroyAllWindows()

#画像水平方向結合
def imgmerge(img1, img2):
    img_h = cv2.hconcat([img1, img2])
    return img_h

#グレースケール化、2値化、背景ノイズ(大)削除
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, binary = cv2.threshold(gray_img, 135, 255, cv2.THRESH_BINARY)
binary[:, 0:10] = 0
binary[binary.shape[1] - 10 : binary.shape[1], :] = 0

#2値化結果比較表示
imageshow("origin vs binary", imgmerge(img, cv2.merge((binary, binary, binary))))

#輪郭表示用画像生成
img_disp = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

#輪郭抽出
contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
#ノイズ除去
contours = list(filter(lambda x: cv2.contourArea(x) > 20, contours))
#ミルククラウン一部消えないように画像下側のノイズのみ削除
contours = list(filter(lambda x: (cv2.moments(x)["m01"] / cv2.moments(x)["m00"]) < 200, contours))

#マスク画像生成
contour_mask = np.zeros_like(binary)
cv2.drawContours(contour_mask, contours, -1, color=255, thickness=-1)
cv2.drawContours(img_disp, contours, -1, (0, 255, 0), 1)

#輪郭抽出結果表示
imageshow("origin vs contour",imgmerge(img, img_disp))
imageshow("binary vs mask",imgmerge(binary, contour_mask))

#マスク生成
masked_img = copy.deepcopy(img)
masked_img[contour_mask==0] = [0, 0, 0]

#入力画像とマスク適用画像表示
imageshow("origin vs masked", imgmerge(img, masked_img))