import cv2
import cv2.aruco as aruco
import os

def generate_aruco_markers(marker_ids, marker_size=200, output_directory=None):
    # Create the output directory if it doesn't exist
    if output_directory is None:
        output_directory = os.path.dirname(__file__)
       

    # Get the Aruco dictionary
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
    

    # Create the first marker
    marker1_image = aruco.generateImageMarker(aruco_dict, marker_ids[0], marker_size)
    marker1_filename = os.path.join(output_directory, f"marker_{marker_ids[0]}.png")
    cv2.imwrite(marker1_filename, marker1_image)

    # Create the second marker
    marker2_image = aruco.generateImageMarker(aruco_dict, marker_ids[1], marker_size)
    marker2_filename = os.path.join(output_directory, f"marker_{marker_ids[1]}.png")
    cv2.imwrite(marker2_filename, marker2_image)

    print(f"Markers saved to {output_directory}")

# Example usage: Generate markers with IDs 4 and 5 and save them in the current directory
generate_aruco_markers([4, 5])
