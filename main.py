import cv2

image = cv2.imread("family.jpg")

if image is None:
    print("Error: Could not load image.")
    exit()
    
rotated = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

brightness = 50
bright_image = cv2.convertScaleAbs(image, alpha=1.0, beta=brightness)

height, width = image.shape[:2]

cropped = image[
    height // 4: 3 * height // 4,
    width // 4: 3 * width // 4
]

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

faces = face_cascade.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(30, 30)
)

for (x, y, w, h) in faces:
    cv2.rectangle(
        image,
        (x, y),
        (x + w, y + h),
        (0, 255, 0),
        2
    )

count = len(faces)

cv2.putText(
    image,
    "Faces: " + str(count),
    (20, 40),
    cv2.FONT_HERSHEY_SIMPLEX,
    1,
    (0, 255, 0),
    2
)

cv2.imshow("Face Detection", image)
cv2.imshow("Rotated Image", rotated)
cv2.imshow("Bright Image", bright_image)
cv2.imshow("Cropped Image", cropped)

cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite("rotated.jpg", rotated)
cv2.imwrite("bright_image.jpg", bright_image)
cv2.imwrite("cropped.jpg", cropped)
cv2.imwrite("face_detection.jpg", image)

print("Number of faces detected:", count)
print("Images saved successfully!")