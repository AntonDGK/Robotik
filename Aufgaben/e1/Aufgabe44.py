from Aufgabe41 import *
from Aufgabe42 import *
from Aufgabe43 import *
def apply_transform(polygon, transform):
    """
    Apply a 2D homogeneous transformation matrix to a polygon.

    Parameters:
    - polygon (np.ndarray): An (N, 2) array of vertex coordinates.
    - transform (np.ndarray): A (3, 3) homogeneous transformation matrix.

    Returns:
    - np.ndarray: The transformed polygon as an (N, 2) array.
    """
    # Convert to homogeneous coordinates (N, 3)
    # add code here
    verticesX = []
    verticesY = []
    for i in range(len(polygon[0])):
        tempX = polygon[0][i] * transform[0][0] + polygon[1][i] * transform[0][1] + transform[0][2]
        verticesX.append(tempX)
        tempY = polygon[0][i] * transform[1][0] + polygon[1][i] * transform[1][1] + transform[1][2]
        verticesY.append(tempY)
    return (verticesX, verticesY)

# Create transformation: rotate 45° and translate by (2, 2)

cos_t, sin_t = np.cos(np.pi / 4), np.sin(np.pi / 4)
dx = 2
dy = 2
T = ([cos_t, -sin_t, dx], [sin_t, cos_t, dy], [0, 0, 1])# add code here

# Apply transformation
square_vertices = [(1.5, 1.5), (2.5, 1.5), (2.5, 2.5), (1.5, 2.5)]
square = create_polygon(square_vertices)

transformed_square = apply_transform(square, T)

# Plot original and transformed
plot_polygons([square, transformed_square], colors=['red', 'green'], labels=['Original', 'Transformed'])
