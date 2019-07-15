import cv2
import pytesseract

if __name__ == '__main__':
    pytesseract.pytesseract.tesseract_cmd = 'D:/Program Files (x86)/Tesseract-OCR/tesseract.exe'
    for i in range(1, 300):
        try:
            path = "result/result{}.jpg".format(i)
            img = cv2.imread(path)

            platePath = "area/area{}.jpg".format(i)
            plateImg = cv2.imread(platePath)
            plateImg = cv2.cvtColor(plateImg, cv2.COLOR_BGR2RGB)
            # s = pytesseract.image_to_string(plateImg, lang='chi_sim')
            s = pytesseract.image_to_string(plateImg)
            print(s)
            cv2.putText(img, s, (100, 100), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
            cv2.imwrite("./result/result{}.jpg".format(i), img)
        except:
            pass
