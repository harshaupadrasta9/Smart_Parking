## Parking System
## Introduction

* This is a simple parking system implemented using Flask, OpenCV, and Tesseract OCR. The system allows users to check in and check out vehicles, and it keeps track of the parking occupancy.

## Requirements
 
 * Make sure you have the following dependencies installed:

* Flask,
OpenCV,
Tesseract OCR,
NumPy,
pandas.
imutils,
You can install them using the following command:

   pip install Flask opencv-python pytesseract numpy pandas imutils
## How to Run
Clone the repository to your local machine.
Navigate to the project directory using the command line.
Run the following command to start the Flask application:

python app.py

Open a web browser and go to http://127.0.0.1:5000/ to access the application.
Usage
Login: Visit the login page (/login) and use the credentials to log in as an admin.

## Dashboard 

* After logging in, you will be redirected to the dashboard (/index). The dashboard shows the current parking occupancy and availability.

* Check-In: Navigate to the check-in page (/checkin) to capture and process the image of a vehicle entering the parking lot.

* Check-Out: Similarly, go to the check-out page (/checkout) to capture and process the image of a vehicle leaving the parking lot.

## Note

   Ensure that your system has a webcam connected for capturing images.
Make sure Tesseract OCR is properly installed and configured.
