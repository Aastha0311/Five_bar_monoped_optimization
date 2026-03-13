import xml.etree.ElementTree as ET


def modify_5bar_xml(
        xml_file,
        output_file,
        l1,
        l2,
        torso_width,
        torque_left,
        torque_right,
        mass_left,
        mass_right):

    tree = ET.parse(xml_file)
    root = tree.getroot()

    half_width = torso_width / 2

    # -----------------------------
    # Update torso width + mass
    # -----------------------------
    for geom in root.findall(".//geom[@name='torso']"):

        size = geom.get("size").split()
        size[0] = str(half_width)
        geom.set("size", " ".join(size))

        base_mass = 0.0
        total_mass = base_mass + mass_left + mass_right
        geom.set("mass", str(total_mass))

    # -----------------------------
    # Move hip joint locations
    # -----------------------------
    for body in root.findall(".//body[@name='l1_left']"):
        body.set("pos", f"{-half_width} 0 0")

    for body in root.findall(".//body[@name='l1_right']"):
        body.set("pos", f"{half_width} 0 0")

    # -----------------------------
    # Update thigh lengths
    # -----------------------------
    for geom in root.findall(".//geom[@name='thigh_left']"):
        geom.set("fromto", f"0 0 0 {l1} 0 0")

    for geom in root.findall(".//geom[@name='thigh_right']"):
        geom.set("fromto", f"0 0 0 {l1} 0 0")

    # -----------------------------
    # Move knee joints
    # -----------------------------
    for body in root.findall(".//body[@name='l2_left']"):
        body.set("pos", f"{l1} 0 0")

    for body in root.findall(".//body[@name='l2_right']"):
        body.set("pos", f"{l1} 0 0")

    # -----------------------------
    # Update shank lengths
    # -----------------------------
    for geom in root.findall(".//geom[@name='shank_left']"):
        geom.set("fromto", f"0 0 0 {l2} 0 0")

    for geom in root.findall(".//geom[@name='shank_right']"):
        geom.set("fromto", f"0 0 0 {l2} 0 0")

    # -----------------------------
    # Move feet
    # -----------------------------
    for geom in root.findall(".//geom[@name='foot_left']"):
        geom.set("pos", f"{l2} 0 0")

    for geom in root.findall(".//geom[@name='foot_right']"):
        geom.set("pos", f"{l2} 0 0")

    # -----------------------------
    # Move coupler sites
    # -----------------------------
    for site in root.findall(".//site[@name='left_tip']"):
        site.set("pos", f"{l2} 0 0")

    for site in root.findall(".//site[@name='right_tip']"):
        site.set("pos", f"{l2} 0 0")

    # -----------------------------
    # Update torque limits
    # -----------------------------
    for motor in root.findall(".//motor[@name='motor_left']"):
        motor.set("ctrlrange", f"-{torque_left} {torque_left}")

    for motor in root.findall(".//motor[@name='motor_right']"):
        motor.set("ctrlrange", f"-{torque_right} {torque_right}")

    tree.write(output_file)