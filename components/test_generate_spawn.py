import xml.etree.ElementTree as ET
import numpy as np


def modify_5bar_xml(
        xml_file,
        output_file,
        l1,
        l2,
        hip_torque,
        base_height=-0.5):

    tree = ET.parse(xml_file)
    root = tree.getroot()

    # --- Compute IK for straight leg ---
    def ik(x, z):
        c2 = (x**2 + z**2 - l1**2 - l2**2) / (2*l1*l2)
        theta2 = np.arctan2(-np.sqrt(1-c2**2), c2)
        A = l1 + l2*c2
        B = l2*np.sin(theta2)
        theta1 = np.arctan2(A*x + B*z, x*B - A*z)
        return theta1, theta2

    theta1, theta2 = ik(0, base_height)

    # --- Move base ---
    for body in root.findall(".//body[@name='root']"):
        body.set("pos", f"0 0 {-base_height}")

    # --- Modify LEFT branch ---
    for body in root.findall(".//body[@name='link1_left']"):
        body.set("euler", f"0 {(np.pi/2)-theta1} 0")
        geom = body.find("geom")
        geom.set("fromto", f"0 0 0 {l1} 0 0")

    for body in root.findall(".//body[@name='link2_left']"):
        body.set("pos", f"{l1} 0 0")
        body.set("euler", f"0 {-theta2} 0")
        geom = body.find("geom")
        geom.set("fromto", f"0 0 0 {l2} 0 0")

    # --- RIGHT branch mirrored ---
    for body in root.findall(".//body[@name='link1_right']"):
        body.set("euler", f"0 {(np.pi/2)-theta1} 0")
        geom = body.find("geom")
        geom.set("fromto", f"0 0 0 {l1} 0 0")

    for body in root.findall(".//body[@name='link2_right']"):
        body.set("pos", f"{l1} 0 0")
        body.set("euler", f"0 {-theta2} 0")
        geom = body.find("geom")
        geom.set("fromto", f"0 0 0 {l2} 0 0")

    # --- Update torque limits ---
    for motor in root.findall(".//motor"):
        motor.set("ctrlrange", f"-{hip_torque} {hip_torque}")

    tree.write(output_file)

base_xml = "/home/stochlab/repo/optimal-design-legged-robots/xmls/5bar_base.xml"
generated_xml = "/home/stochlab/repo/optimal-design-legged-robots/xmls/5bar_generated.xml"

# Example parameters
l1 = 0.35
l2 = 0.45
hip_torque = 60
base_height = -0.6

modify_5bar_xml(
    base_xml,
    generated_xml,
    l1=l1,
    l2=l2,
    hip_torque=hip_torque,
    base_height=base_height
)
