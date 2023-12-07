import cv2

# Open a connection to the webcam (adjust the index based on your system)
webcam_index = 2
cap = cv2.VideoCapture(webcam_index)

# Check if the webcam is opened successfully
if not cap.isOpened():
    print(f"Error: Could not open webcam with index {webcam_index}.")
    exit()

# Counter for naming images
image_counter = 0

# Loop to continuously capture frames from the webcam
while True:
    # Read a frame from the webcam
    ret, frame = cap.read()

    # Check if the frame was read successfully
    if not ret:
        print("Error: Failed to capture frame.")
        break

    # Display the frame
    cv2.imshow('Webcam Feed', frame)

    # Capture an image when the 'c' key is pressed
    key = cv2.waitKey(10)
    if key == ord('c'):
        # Save the captured image
        image_filename = f"captured_image_{image_counter}.jpg"
        cv2.imwrite(image_filename, frame)
        print(f"Image saved as {image_filename}")
        image_counter += 1

    # Break the loop if 'q' key is pressed
    elif key == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()