from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import numpy as np
import cv2
import imutils
import sys
import pytesseract
import pandas as pd
import time
import os
from time import sleep
global img_resized1
global img_resized2


app = Flask(__name__)

@app.route("/")
@app.route("/login")
def login():
    return render_template('login.html')

@app.route("/logout")
def logout():
    return render_template('login.html')

@app.route("/adminlogin", methods=['GET', 'POST'])
def adminlogin():
    if request.method == 'POST':
        usr = request.form['email']
        pswd = request.form['password']
        print(usr, pswd)
        if usr=='admin' and pswd=='admin':
            return render_template("index.html")
        else:
            return render_template("login.html")
    else:
        return render_template("login.html")

@app.route("/index")
def index():
    dataset1 = pd.read_csv('check-in.csv', header=None)
    row_count1 = dataset1.shape[0]
    print(row_count1)
    """conn = sqlite3.connect('parkingdata.db')
    
    cursor = conn.cursor()
    cursor.execute("CREATE TABLE IF NOT EXISTS 'check_in'(date VARCHAR(255) NOT NULL,v_number CHAR(25) NOT NULL)")
    table_name = 'check_in'
    query = f"SELECT COUNT(*) FROM {table_name}"
    cursor.execute(query)
    result = cursor.fetchone()
    row_count1 = result[0]

    cursor.close()
    conn.close()"""

    dataset2 = pd.read_csv('check-out.csv', header=None)
    row_count2 = dataset2.shape[0]
    print(row_count2)
    """conn = sqlite3.connect('parkingdata.db')
    cursor = conn.cursor()
    cursor.execute("CREATE TABLE IF NOT EXISTS 'check_out'(date VARCHAR(255) NOT NULL,v_number CHAR(25) NOT NULL)")
    table_name = 'check_out'
    query = f"SELECT COUNT(*) FROM {table_name}"
    cursor.execute(query)
    result = cursor.fetchone()
    row_count2 = result[0]

    cursor.close()
    conn.close()"""
    
    total_count=row_count1-row_count2
    if total_count>20:
        text="parking slot full please wait "
        return render_template('index.html', row_count1=row_count1, row_count2=row_count2, total_count=total_count, text=text)
    r_count=str(20-total_count)
    text="parking slot available for " + r_count
    return render_template('index.html', row_count1=row_count1, row_count2=row_count2, total_count=total_count, text=text)
    
# @app.route("/checkin")
# def checkin():
#     return render_template("checkin.html")

@app.route("/checkin", methods=['GET','POST'])
def checkin():
    if request.method == 'POST':
        key = cv2. waitKey(1)
        webcam = cv2.VideoCapture(0)
        sleep(2)
        while True:

            try:
                check, frame = webcam.read()
                print(check) #prints true as long as the webcam is running
                print(frame) #prints matrix values of each framecd 
                cv2.imshow("Capturing", frame)
                key = cv2.waitKey(1)
                if key == ord('s'): 
                    cv2.imwrite(filename='C:/Users/VARSHITH/OneDrive/Desktop/minor/parking-system - Copy/static/uploads/saved_img.jpg', img=frame)
                    webcam.release()
                    print("Processing image...")
                    file = cv2.imread('C:/Users/VARSHITH/OneDrive/Desktop/minor/parking-system - Copy/static/uploads/saved_img.jpg', cv2.IMREAD_ANYCOLOR)
                    """print("Converting RGB image to grayscale...")
                    gray = cv2.cvtColor(img_, cv2.COLOR_BGR2GRAY)
                    print("Converted RGB image to grayscale...")
                    print("Resizing image to 28x28 scale...")
                    file = cv2.resize(gray,(28,28))
                    print("Resized...")
                    #img_resized1 = cv2.imwrite(filename='C:/Users/lenovo/Downloads/parking-system/static/uploads/saved_img-final.jpg', img=img_)
                    #print("Image saved!")
                    #file = img_resized1
                    # basepath = os.path.dirname(__file__)
                    # file_path = os.path.join(basepath, 'static', 'uploads', secure_filename(file.filename))
                    # file.save(file_path)"""
                    process_image(file)
                    #print(file_path)
                    
                    break
                
                elif key == ord('q'):
                    webcam.release()
                    cv2.destroyAllWindows()
                    break
    
            except(KeyboardInterrupt):
                print("Turning off camera.")
                webcam.release()
                print("Camera off.")
                print("Program ended.")
                cv2.destroyAllWindows()
                break
        """file = img_resized1
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'static', 'uploads', secure_filename(file.filename))
        file.save(file_path)
        process_image(file_path)
        print(file_path)"""
    return render_template("checkin.html")

# @app.route("/checkout")
# def checkout():
#     return render_template("checkout.html")

def process_image(file):
    # Read the image
    #image = cv2.imread('C:/Users/lenovo/Downloads/parking-system/static/uploads/saved_img_final.jpg')
    image=file
    # Resize the image
    image = imutils.resize(image, width=500)
    cv2.imshow("Original Image", image)
    print("############################# original")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    edged = cv2.Canny(gray, 170, 200)
    cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:30]
    NumberPlateCnt = None
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            NumberPlateCnt = approx
            print("*******************",NumberPlateCnt)
            break
    # Mask the part other than the number plate
    if NumberPlateCnt is not None and len(NumberPlateCnt) == 4:
        mask = np.zeros(gray.shape, np.uint8)
        new_image = cv2.drawContours(mask, [NumberPlateCnt], 0, 255, -1)
        new_image = cv2.bitwise_and(image, image, mask=mask)
        cv2.namedWindow("Final_image", cv2.WINDOW_NORMAL)
        cv2.imshow("Final_image", new_image)
        config = '-l eng --oem 1 --psm 3'
        text = pytesseract.image_to_string(new_image, config=config)
        print("############################# final",text)
    
    else:
        print("NumberPlateCnt is not valid:", NumberPlateCnt)
    # Configuration for tesseract
    config = '-l eng --oem 1 --psm 3'
    # Run tesseract OCR on image
    if 'new_image' in locals():
        text = pytesseract.image_to_string(new_image, config=config)
        # Data is stored in CSV file
        raw_data = {'date': [time.asctime(time.localtime(time.time()))],
                    'v_number': [text]}
        datee=[time.asctime(time.localtime(time.time()))]
        v_number=[text]
       
        
        print("///////////////////////", raw_data)
        df = pd.DataFrame(raw_data, columns=['date', 'v_number'])
        df.to_csv('check-in.csv', mode='a', header=False)
        # Print recognized text
        print(text)
    else:
        print("new_image is not defined. Could not perform OCR.")

    cv2.waitKey(0)
    cv2.destroyAllWindows()

@app.route("/checkout", methods=['GET','POST'])
def checkout():
    if request.method == 'POST':
        key = cv2. waitKey(1)
        webcam = cv2.VideoCapture(0)
        sleep(2)
        while True:

            try:
                check, frame = webcam.read()
                print(check) #prints true as long as the webcam is running
                print(frame) #prints matrix values of each framecd 
                cv2.imshow("Capturing", frame)
                key = cv2.waitKey(1)
                if key == ord('s'): 
                    cv2.imwrite(filename='C:/Users/VARSHITH/OneDrive/Desktop/minor/parking-system - Copy/static/uploads/saved_img.jpg', img=frame)
                    webcam.release()
                    print("Processing image...")
                    file = cv2.imread('C:/Users/VARSHITH/OneDrive/Desktop/minor/parking-system - Copy/static/uploads/saved_img.jpg', cv2.IMREAD_ANYCOLOR)
                    process_image2(file)
                    print(file)
                    
                    break
                
                elif key == ord('q'):
                    webcam.release()
                    cv2.destroyAllWindows()
                    break
    
            except(KeyboardInterrupt):
                print("Turning off camera.")
                webcam.release()
                print("Camera off.")
                print("Program ended.")
                cv2.destroyAllWindows()
                break
       
    return render_template("checkout.html")

def process_image2(file):
  
    image=file
    image = imutils.resize(image, width=500)
    cv2.imshow("Original Image", image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    edged = cv2.Canny(gray, 170, 200)
    cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:30]
    NumberPlateCnt = None
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            NumberPlateCnt = approx
            break
    # Mask the part other than the number plate
    if NumberPlateCnt is not None and len(NumberPlateCnt) == 4:
        mask = np.zeros(gray.shape, np.uint8)
        new_image = cv2.drawContours(mask, [NumberPlateCnt], 0, 255, -1)
        new_image = cv2.bitwise_and(image, image, mask=mask)
        cv2.namedWindow("Final_image", cv2.WINDOW_NORMAL)
        cv2.imshow("Final_image", new_image)
    else:
        print("NumberPlateCnt is not valid:", NumberPlateCnt)
    # Configuration for tesseract
    config = '-l eng --oem 1 --psm 3'
    # Run tesseract OCR on image
    if 'new_image' in locals():
        text = pytesseract.image_to_string(new_image, config=config)
        # Data is stored in CSV file
        raw_data = {'date': [time.asctime(time.localtime(time.time()))],
                    'v_number': [text]}
        
        df = pd.DataFrame(raw_data, columns=['date', 'v_number'])
        df.to_csv('check-out.csv', mode='a', header=False)
        # Print recognized text
        print(text)
    else:
        print("new_image is not defined. Could not perform OCR.")

    cv2.waitKey(0)
    cv2.destroyAllWindows()




if __name__ == "__main__":
    app.run(debug=True)
