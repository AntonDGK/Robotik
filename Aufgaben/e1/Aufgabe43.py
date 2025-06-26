from Aufgabe41 import *
def rotate_polygon(polygon, theta):
    """
    Rotate a polygon counterclockwise by theta radians around the origin.

    Parameters:
    - polygon (np.ndarray): An (N, 2) array of vertex coordinates.
    - theta (float): Rotation angle in radians.

    Returns:
    - np.ndarray: The rotated polygon as an (N, 2) array.
    """
    # add code here 
    verticesX = []
    verticesY = []
    for i in range(len(polygon[0])):
        tempX = (polygon[0][i] * np.cos(theta)) - (polygon[1][i] * np.sin(theta))
        verticesX.append(tempX)
        tempY = (polygon[0][i] * np.sin(theta)) + (polygon[1][i] * np.cos(theta))
        verticesY.append(tempY)
    return (verticesX, verticesY)

# Rotate by 45 degrees (π/4 radians)
square_vertices = [(1.5, 1.5), (2.5, 1.5), (2.5, 2.5), (1.5, 2.5)]
square = create_polygon(square_vertices)

rotated_square = rotate_polygon(square, np.pi / 4)

# Plot both
plot_polygons([square, rotated_square], colors=['red', 'purple'], labels=['Original', 'Rotated 45°'])