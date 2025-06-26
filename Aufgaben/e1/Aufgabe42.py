from Aufgabe41 import *

def translate_polygon(polygon, dx, dy):
    """
    Translate a polygonal body by a given offset.

    Parameters:
    - polygon (np.ndarray): An (N, 2) array of vertex coordinates.
    - dx (float): Translation in x-direction.
    - dy (float): Translation in y-direction.

    Returns:
    - np.ndarray: The translated polygon as an (N, 2) array.
    """
    # Add code here
    verticesX = []
    for i in polygon[0]:
        verticesX.append(i+dx)
    verticesY = []
    for j in polygon[1]:
        verticesY.append(j+dy)
    return (verticesX, verticesY)

# Translate the square robot
square_vertices = [(1, 1), (2, 1), (2, 2), (1, 2)]
square = create_polygon(square_vertices)
translated_square = translate_polygon(square, dx=2, dy=1)

# Visualize both
plot_polygons([square, translated_square], colors=['red', 'orange'], labels=['Original', 'Translated'])