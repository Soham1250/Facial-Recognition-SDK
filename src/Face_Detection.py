import cv2

# Load the pre-trained Haar Cascade classifier file
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Read the image where you want to detect faces
img = cv2.imread('your_image.jpg')

# Resize the image if it exceeds 1920 x 1080
max_width = 1920
max_height = 1080

height, width = img.shape[:2]
if width > max_width or height > max_height:
    scaling_factor = min(max_width / width, max_height / height)
    img = cv2.resize(img, (int(width * scaling_factor), int(height * scaling_factor)))

# Convert the image to grayscale (Haar Cascades work better on grayscale images)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect faces in the image
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

# Draw rectangles around detected faces
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

# Display the output
cv2.namedWindow('Face Detection',cv2.WINDOW_NORMAL)
cv2.resizeWindow('Face Detection',600, 800)

cv2.imshow('Face Detection', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
