from Aufgabe41 import *
from Aufgabe42 import *
from Aufgabe43 import *
from Aufgabe44 import *
from Aufgabe51 import *
def draw_three_link_arm(l1, l2, l3, theta2, theta3):
    """
    Visualize a 3-link robot arm with two revolute joints.

    Parameters:
    - l1, l2, l3 (float): Lengths of links A₁, A₂, and A₃.
    - theta2 (float): Joint angle at the base of A₂ (relative to A₁).
    - theta3 (float): Joint angle at the base of A₃ (relative to A₂).
    """
    #add code here
    def create_polygon_withWidth(leng):
        width = leng
        height = 0.5
        A = create_polygon([
            (-(width/2), -(height/2)), ((width/2), -(height/2)), ((width/2), (height/2)), (-(width/2), (height/2))
        ])
        return A
    startDx = 3.0
    startDy = 5.0

    A1 = create_polygon_withWidth(l1)
    A2 = create_polygon_withWidth(l2)
    A3 = create_polygon_withWidth(l3)
    T_A1 = create_transform(theta=0.0, dx=startDx, dy=startDy)
    A1_world = apply_transform(A1, T_A1)
    A2 = translate_polygon(A2, l2/2, 0)
    T_A2 = create_transform(theta=theta2, dx=(startDx+l1/2), dy=startDy) 
    A2_world = apply_transform(A2, T_A2)
    A3 = translate_polygon(A3, l3/2, 0)
    T_A3 = create_transform(theta=theta2+theta3, dx=(startDx+l1/2+l2*np.cos(theta2)), dy=(startDy+l2*np.sin(theta2)))
    A3_world = apply_transform(A3, T_A3)
        # Plot all links
    plot_polygons(
        [A1_world, A2_world, A3_world],
        colors=['black', 'crimson', 'teal'],
        labels=['Link A₁', 'Link A₂', 'Link A₃'],
        title="Three-Link Robot Arm"
    )

draw_three_link_arm(
    l1=3.0, l2=2.5, l3=2.0,
    theta2=np.pi / 6,
    theta3=-np.pi / 4
)
draw_three_link_arm(
    l1=3.0, l2=2.5, l3=2.0,
    theta2=-np.pi / 6,
    theta3=-np.pi / 4
)