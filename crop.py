import cv2

if __name__ == '__main__':
    for i in range(1, 3000):
        path = "pic/{}.jpg".format(i)
        newPath = "newPic/new{}.jpg".format(i)
        im = cv2.imread(path)
        newIm = im[100:1000, 150:1300]
        cv2.imwrite(newPath, newIm)
        print(i)
