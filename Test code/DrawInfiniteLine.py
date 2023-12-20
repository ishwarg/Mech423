import cv2
import numpy as np

def DrawInfiniteLine(image,point,unit_vector):

    # Scale the vector to extend the line across the entire image
    scale = max(image.shape) * 2
    end_point = (int(point[0] + scale * unit_vector[0]), int(point[1] + scale * unit_vector[1]))
    start_point = (int(point[0] - scale * unit_vector[0]), int(point[1] - scale * unit_vector[1]))

    # Draw the line on the image
    color = (0, 255, 0)  # Green color
    thickness = 2
    cv2.line(image, start_point, end_point, color, thickness)

    # Display the image
    cv2.imshow('Infinite Line', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    image = np.zeros((1000, 2000, 3), dtype=np.uint8)
    # Create a black image
    # Define the known point (x, y) and the known unit vector (vx, vy)
    point = (500, 250)
    unit_vector = (1, 1)
    DrawInfiniteLine(image,point,unit_vector)
