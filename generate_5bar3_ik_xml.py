import xml.etree.ElementTree as ET

def generate_5bar_xml(base_xml, output_xml, l1, l2, base_height):

    tree = ET.parse(base_xml)
    root = tree.getroot()

    root_body = root.find(".//body[@name='root']")
    root_body.set("pos", f"0 0 {base_height}")

    # left
    root.find(".//geom[@name='l1_left_geom']").set(
        "fromto", f"0 0 0 {l1} 0 0"
    )
    root.find(".//body[@name='l2_left']").set(
        "pos", f"{l1} 0 0"
    )
    root.find(".//geom[@name='l2_left_geom']").set(
        "fromto", f"0 0 0 {l2} 0 0"
    )
    root.find(".//site[@name='left_tip']").set(
        "pos", f"{l2} 0 0"
    )

    # right
    root.find(".//geom[@name='l1_right_geom']").set(
        "fromto", f"0 0 0 {l1} 0 0"
    )
    root.find(".//body[@name='l2_right']").set(
        "pos", f"{l1} 0 0"
    )
    root.find(".//geom[@name='l2_right_geom']").set(
        "fromto", f"0 0 0 {l2} 0 0"
    )
    root.find(".//site[@name='right_tip']").set(
        "pos", f"{l2} 0 0"
    )

    tree.write(output_xml)
    print("Generated:", output_xml)



generate_5bar_xml(
    "/home/stochlab/repo/optimal-design-legged-robots/xmls/5bar_base.xml",
    "/home/stochlab/repo/optimal-design-legged-robots/xmls/5bar_generated.xml",
    l1=0.4,
    l2=0.4,
    base_height=0.6
)

