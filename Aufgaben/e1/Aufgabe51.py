from Aufgabe41 import *
from Aufgabe42 import *
from Aufgabe43 import *
from Aufgabe44 import *
def create_transform(theta, dx, dy):
    """
    Construct a 2D homogeneous transformation matrix.

    Parameters:
    - theta (float): Rotation angle in radians.
    - dx (float): Translation in x-direction.
    - dy (float): Translation in y-direction.

    Returns:
    - np.ndarray: A (3, 3) homogeneous transformation matrix.
    """
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    t = ([cos_t, -sin_t, dx], [sin_t, cos_t, dy], [0, 0, 1])
    return t# add code here
def draw_two_link_arm(base_length, arm_length, joint_angle):
    """
    Create and visualize a 2-link arm with a revolute joint.

    Parameters:
    - base_length (float): Length of the first link A₁.
    - arm_length (float): Length of the second link A₂.
    - joint_angle (float): Rotation angle of A₂ in radians.
    """
    # Link A₁ (base)
    width = base_length
    height = 0.5
    A1 = create_polygon([
        (-(width/2), -(height/2)), ((width/2), -(height/2)), ((width/2), (height/2)), (-(width/2), (height/2))
    ])
    T_A1 = create_transform(theta=0.0, dx=3.0, dy=5.0)
    A1_world = apply_transform(A1, T_A1)

    # Link A₂ (child)
    widthA2 = arm_length
    A2 = create_polygon([
        (-(widthA2/2), -(height/2)), ((widthA2/2), -(height/2)), ((widthA2/2), (height/2)), (-(widthA2/2), (height/2))
    ])
    # Add code here 
    A2 = translate_polygon(A2, arm_length/2, 0)

    # Combine all transforms from A₂ to world
    T_A2 = create_transform(theta=joint_angle, dx=(3.0+base_length/2), dy=5.0)  #add code here 
    A2_world = apply_transform(A2, T_A2)

    # Plot both links
    plot_polygons(
        [A1_world, A2_world],
        colors=['black', 'crimson'],
        labels=['Link A₁', 'Link A₂'],
        title="Two-Link Robot Arm"
    )

draw_two_link_arm(base_length=4.0, arm_length=3.0, joint_angle=np.pi / 8)
draw_two_link_arm(base_length=4.0, arm_length=3.0, joint_angle=np.pi / 4)