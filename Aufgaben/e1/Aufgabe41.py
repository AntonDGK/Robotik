import numpy as np
import matplotlib.pyplot as plt


def create_polygon(vertices):
    """
    Create a polygonal body represented by an array of 2D vertices.

    Parameters:
    - vertices (list of tuple): A list of (x, y) coordinates defining the polygon corners in order.

    Returns:
    - np.ndarray: An (N, 2) NumPy array representing the polygon.
    """
    arrayX = []
    arrayY = []
    for i in vertices:
        arrayX.append(i[0])
        arrayY.append(i[1])
    arrayX.append(vertices[0][0])
    arrayY.append(vertices[0][1])

         # Create a figure containing a single Axes.            
    return (arrayX, arrayY)

def plot_polygons(polygons, colors=None, labels=None, title="Robot Bodies"):
    """
    Plot multiple polygons on a fixed 10x10 field.

    Parameters:
    - polygons (list of np.ndarray): List of (N, 2) arrays of vertices.
    - colors (list of str): Optional list of colors for each polygon.
    - labels (list of str): Optional list of labels for the legend.
    - title (str): Plot title.
    """
    #plt.figure(figsize=(6, 6))
    
    for i, poly in enumerate(polygons):
        closed_poly = np.vstack([poly, poly[0]])
        color = colors[i] if colors and i < len(colors) else 'blue'
        label = labels[i] if labels and i < len(labels) else f"Polygon {i+1}"

        # Add code here
        
        pol = plt.fill(polygons[i][0], polygons[i][1], label=label, edgecolor=color, facecolor=color, alpha=0.3)
        plt.legend()
        
    
    plt.xlim(0, 10)
    plt.ylim(0, 10)
    plt.xticks(np.arange(0, 11, 1))
    plt.yticks(np.arange(0, 11, 1))
    plt.gca().set_aspect('equal', adjustable='box')
    plt.grid(True)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title(title)
    plt.show()

# Create a triangle shaped
triangle_vertices = [(0, 0), (1, 0), (0.5, 1)]
triangle = create_polygon(triangle_vertices)
plot_polygons([triangle], colors=['green'], labels=['Triangle Bot'])