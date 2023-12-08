import cv2

# Create a VideoCapture object
cap = cv2.VideoCapture(1)  # 0 corresponds to the default camera (usually the built-in webcam)

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Set the window name
window_name = "Webcam Feed"

# Create a window
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()

    # Break the loop if reading the frame fails
    if not ret:
        print("Error: Failed to capture frame.")
        break

    # Display the frame
    cv2.imshow(window_name, frame)

    # Save the frame as a JPEG image when 'w' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('w'):
        cv2.imwrite('captured_frame.jpg', frame)
        print("Frame saved as 'captured_frame.jpg'")

    # Exit the loop when 'q' key is pressed
    elif cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the VideoCapture and close the window
cap.release()
cv2.destroyAllWindows()
