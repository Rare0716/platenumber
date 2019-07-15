import cv2

if __name__ == '__main__':
    try:
        pathList = []
        for i in range(1, 10):
            path = "pic/0000{}.jpg".format(i)
            pathList.append(path)
        for i in range(10, 100):
            path = "pic/000{}.jpg".format(i)
            pathList.append(path)
        for i in range(100, 10000):
            path = "pic/00{}.jpg".format(i)
            pathList.append(path)
        for i in range(1000, 10000):
            path = "pic/0{}.jpg".format(i)
            pathList.append(path)
        for j in range(1, 10000):
            newPath = "newPic/new{}.jpg".format(j)
            im = cv2.imread(pathList[j])
            newIm = im[100:1000, 150:1300]
            cv2.imwrite(newPath, newIm)
            print(j)
    except:
        pass
