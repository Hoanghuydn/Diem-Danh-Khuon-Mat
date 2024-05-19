import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import pymysql

# Load the trained face recognition model
model = load_model('face_recognition_model.h5')

# Define a function to preprocess the image
def preprocess_image(img):
    img = cv2.resize(img, (150, 150))
    img = img / 255.0  # Normalize pixel values to [0, 1]
    return img

# Define the face cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Open the camera
cap = cv2.VideoCapture(0)

# Connect to MySQL database
conn = pymysql.connect(
    host="localhost", user="root", password="", database="students"
)

while True:
    # Read a frame from the camera
    ret, frame = cap.read()
    
    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    
    # Process each detected face
    for (x, y, w, h) in faces:
        # Extract the face region from the frame
        face = frame[y:y+h, x:x+w]
        
        # Preprocess the face image
        processed_face = preprocess_image(face)
        
        # Reshape the processed face image to match the input shape of the model
        processed_face = np.expand_dims(processed_face, axis=0)
        
        # Predict the label of the face using the model
        predicted_label = np.argmax(model.predict(processed_face), axis=-1)
        
        # Query the database for the name associated with the predicted label
        cur = conn.cursor()
        #query = "SELECT * FROM data WHERE roll_no= " + str(id)
        cur.execute("SELECT name FROM data WHERE roll_no = %s", (str(predicted_label),))
        row = cur.fetchone()
        name = row[0] if row else "Unknown"
        print("Name:", name)  # Debug print 

        # Display the name on the frame
        cv2.putText(frame, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        # Draw a rectangle around the detected face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
    
    # Display the frame
    cv2.imshow('Face Recognition', frame)
    
    # Check for key press to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close the OpenCV windows
cap.release()
cv2.destroyAllWindows()

# Close MySQL connection
conn.close()
