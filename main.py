import cv2
import numpy as np

# isShow，是否显示处理过程中的图片，程序的main在最后面
# isShow = True
isShow = False


def getTextFromPicture():
    pass


def getColor(img):
    resultList = []
    colorList = [(255, 255, 255), (0, 0, 255), (0, 0, 0), (0, 255, 0), (255, 255, 0)]
    # nameList = ["白", "蓝", "黑", "绿", "黄"]
    nameList = ["white", "blue", "black", "green", "yellow"]
    b = int(np.mean(img[:, :, 0]))
    g = int(np.mean(img[:, :, 1]))
    r = int(np.mean(img[:, :, 2]))
    for item in colorList:
        result = (item[0] - r) * (item[0] - r) + (item[1] - g) * (item[1] - g) + (item[2] - b) * (item[2] - b)
        resultList.append(result)
    minVal = min(resultList)
    minIndex = resultList.index(minVal)
    # 白，蓝，黑，绿，黄
    # print(minIndex)
    return nameList[minIndex]


def verifyByArea(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    val, binImg = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)
    cv2.imshow(binImg)
    return True


def getBinaryPicture(gray):
    '''
    :param gray:
    :return:
    '''
    # # 直方图均衡化
    # equ = cv2.equalizeHist(gray)
    # 高斯滤波
    gaussian = cv2.GaussianBlur(gray, (3, 3), 0, 0, cv2.BORDER_DEFAULT)
    # 中值滤波
    median = cv2.medianBlur(gaussian, 5)
    # median = cv2.blur(median, (5, 5))
    # Sobel算子，X方向求梯度

    sobel = cv2.Sobel(median, cv2.CV_8U, 1, 0, ksize=3)

    # 求y方向的梯度
    # sobel = cv2.Sobel(median, cv2.CV_8U, 0, 1, ksize=3)
    # sobel = cv2.Canny(median, 10, 200)
    # 二值化
    ret, binary = cv2.threshold(sobel, 170, 255, cv2.THRESH_BINARY)
    # 膨胀和腐蚀操作的结构元素，车牌的长宽比大于2.5，
    element1 = cv2.getStructuringElement(cv2.MORPH_RECT, (12, 4))
    element2 = cv2.getStructuringElement(cv2.MORPH_RECT, (12, 4))
    # 膨胀一次，让轮廓突出
    img = cv2.dilate(binary, element2, iterations=2)
    # 腐蚀一次，去掉细节
    img = cv2.erode(img, element1, iterations=1)

    # 再次膨胀，让轮廓明显一些
    img = cv2.dilate(img, element2, iterations=2)
    # cv2.imshow('dilation2', dilation2)

    return img


def findRegion(img):
    region = []
    # 查找轮廓
    contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # 筛选面积小的
    for i in range(len(contours)):
        cnt = contours[i]
        # 计算该轮廓的面积
        area = cv2.contourArea(cnt)

        # 面积小，大了的都筛选掉
        if (area < 3000 or area > 18000):
            continue

        # 轮廓近似，作用很小
        epsilon = 0.001 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        # 找到最小的矩形，该矩形可能有方向
        rect = cv2.minAreaRect(cnt)
        # print("rect is: ")
        # print(rect)

        # box是四个点的坐标
        box = cv2.boxPoints(rect)
        box = np.int0(box)

        # 计算高和宽
        height = abs(box[0][1] - box[2][1])
        width = abs(box[0][0] - box[2][0])
        # 车牌正常情况下长高比在2-5之间
        ratio = float(width) / float(height)
        # print(ratio)

        # 计算长宽比
        if (ratio > 5 or ratio < 2):
            continue

        region.append(box)
    return region


def detect(img):
    # 转化成灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 形态学变换的预处理
    binImg = getBinaryPicture(gray)

    # 查找车牌区域
    region = findRegion(binImg)

    # 用红线画出这些找到的轮廓
    for box in region:
        ys = [box[0, 1], box[1, 1], box[2, 1], box[3, 1]]
        xs = [box[0, 0], box[1, 0], box[2, 0], box[3, 0]]
        ys_sorted_index = np.argsort(ys)
        xs_sorted_index = np.argsort(xs)
        x1 = box[xs_sorted_index[0], 0]
        x2 = box[xs_sorted_index[3], 0]
        y1 = box[ys_sorted_index[0], 1]
        y2 = box[ys_sorted_index[3], 1]
        img_org2 = img.copy()
        # 车牌的可能区域

        img_plate = img_org2[y1:y2, x1:x2]

        # 判断车牌区域的主色调
        color = getColor(img_plate)
        ret, binary = cv2.threshold(img_plate, 150, 255, cv2.THRESH_BINARY)
        # 竖直膨胀的结构元素
        se = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
        binary = cv2.dilate(binary, se, iterations=20)

        # 统计0-1和1-0跳变次数
        # h = len(binary)
        # w = len(binary[0])
        # dVal = 0
        # for i in range(0, w):
        #     dVal = dVal + abs(binary(i + 1, int(h / 2)) - binary(i + 1, int(h / 2)))
        # if dVal < 10:
        #     return
        # cv2.namedWindow('input_image', cv2.WINDOW_AUTOSIZE)

        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # 记录车牌区域的坐标和颜色

        # cv2.imshow('number plate', img_plate)
        # cv2.imshow('binary', binary)
        # cv2.waitKey(0)

        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # if verifyByArea(img_plate):

        # grayPlate = cv2.cvtColor(img_plate, cv2.COLOR_BGR2GRAY)
        # ret, binary = cv2.threshold(gray, 0, 1, cv2.THRESH_OTSU)
        # pointSum = np.sum(binary)
        # proportion = pointSum / (len(binary) * len(binary[0]))
        # if proportion < 0.48 or proportion > 0.52:

        # text = ocr.image_to_text(img_plate)
        # print(text)
        # cv2.imshow("img",img)
        # cv2.waitKey(0)
        # for box in region:
        #     cv2.drawContours(img, [box], 0, (0, 255, 0), 2)
        # ys = [box[0, 1], box[1, 1], box[2, 1], box[3, 1]]
        # xs = [box[0, 0], box[1, 0], box[2, 0], box[3, 0]]
        # ys_sorted_index = np.argsort(ys)
        # xs_sorted_index = np.argsort(xs)
        #
        # x1 = box[xs_sorted_index[0], 0]
        # x2 = box[xs_sorted_index[3], 0]
        #
        # y1 = box[ys_sorted_index[0], 1]
        # y2 = box[ys_sorted_index[3], 1]
        #
        # img_org2 = img.copy()
        # img_plate = img_org2[y1:y2, x1:x2]
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        if isShow:
            cv2.imshow("img", img)
            cv2.imshow('number plate', img_plate)
            cv2.imshow('binary', binary)
            cv2.waitKey(0)
        cv2.drawContours(img, [box], 0, (0, 0, 255), 2)
        with open("result.txt", "a") as f:
            f.write("坐标：{} {} {} {}，颜色：{}\n".format(x1, x2, y1, y2, color))
            f.close()
    return img, img_plate, color


if __name__ == '__main__':

    with open("result.txt", "w") as f:
        f.truncate()
        f.close()

    # 图片的识别数量范围，默认使用0-50张，最多为10000张
    # 运行前需先运行newCrop.py
    for i in range(1000, 10000):
        try:
            path = './newPic/new{}.jpg'.format(i)
            img = cv2.imread(path)
            rawImg = img
            img, img_plate, color = detect(img)
            # print(color)
            cv2.putText(img, '{}'.format(color), (50, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
            cv2.imwrite("./result/result{}.jpg".format(i), img)
            cv2.imwrite("./area/area{}.jpg".format(i), img_plate)
            print(i)
        except:
            pass
