"""Named household props with real MuJoCo free bodies and collision geometry."""
import math
import xml.etree.ElementTree as ET

OBJECTS = {
    "bottle": {"label":"Blue bottle", "aliases":["bottle", "blue bottle", "water bottle"]},
    "mug": {"label":"Red mug", "aliases":["mug", "red mug", "cup"]},
    "cube": {"label":"Blue cube", "aliases":["cube", "blue cube", "block"]},
    "apple": {"label":"Green apple", "aliases":["apple", "green apple"]},
    "ball": {"label":"Yellow ball", "aliases":["ball", "yellow ball"]},
    "can": {"label":"Purple can", "aliases":["can", "purple can"]},
    "bowl": {"label":"Orange bowl", "aliases":["bowl", "orange bowl"]},
    "book": {"label":"Red book", "aliases":["book", "red book"]},
    "sponge": {"label":"Yellow sponge", "aliases":["sponge", "yellow sponge"]},
    "remote": {"label":"TV remote", "aliases":["remote", "tv remote", "remote control"]},
}


def add_props(world):
    definitions = [
        ("mug", (0.46,-0.28,0.765), "mug", (.035,.04), "0.82 0.16 0.12 1", .12),
        ("cube", (0.65,-0.28,0.76), "box", (.035,.035,.035), "0.12 0.38 0.85 1", .10),
        ("apple", (0.83,-0.28,0.76), "sphere", (.035,), "0.29 0.65 0.15 1", .15),
        ("ball", (0.28,0.00,0.76), "sphere", (.035,), "0.98 0.77 0.08 1", .05),
        ("can", (0.46,0.00,0.78), "cylinder", (.026,.055), "0.55 0.25 0.75 1", .12),
        ("bowl", (0.65,0.08,0.745), "bowl", (.055,.02), "0.95 0.4 0.09 1", .15),
        ("book", (0.83,0.00,0.741), "box", (.055,.075,.016), "0.67 0.13 0.16 1", .25),
        ("sponge", (0.4,0.20,0.74), "box", (.04,.026,.015), "0.94 0.8 0.2 1", .03),
        ("remote", (0.67,0.20,0.735), "box", (.055,.022,.01), "0.10 0.12 0.15 1", .07),
    ]
    for name, pos, shape, size, color, mass in definitions:
        body = ET.SubElement(world, "body", name=name, pos=" ".join(map(str,pos)))
        ET.SubElement(body, "freejoint", name=name+"_free")
        if shape in ("mug","bowl"):
            radius, height = size
            ET.SubElement(body,"geom",name=name+"_base",type="cylinder",pos=f"0 0 {-height+.004}",size=f"{radius} .004",rgba=color,mass=str(mass/3))
            for i in range(16):
                a = 2*math.pi*i/16
                x,y=radius*math.cos(a),radius*math.sin(a)
                ET.SubElement(body,"geom",name=f"{name}_wall_{i}",type="capsule",fromto=f"{x} {y} {-height+.008} {x} {y} {height-.005}",size=".006",rgba=color,mass=str(mass/24))
            if shape=="mug":
                for i in range(12):
                    a,b=2*math.pi*i/12,2*math.pi*(i+1)/12
                    points=[radius+.021+.025*math.cos(a),0,.027*math.sin(a),radius+.021+.025*math.cos(b),0,.027*math.sin(b)]
                    ET.SubElement(body,"geom",name=f"mug_handle_{i}",type="capsule",fromto=" ".join(map(str,points)),size=".005",rgba=color,mass=".002")
        else:
            ET.SubElement(body,"geom",name=name+"_body",type=shape,size=" ".join(map(str,size)),rgba=color,mass=str(mass),friction="1 0.01 0.001")
