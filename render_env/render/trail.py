from collections import deque

from pandac.PandaModules import GeomVertexData, GeomVertexFormat, Geom
from pandac.PandaModules import GeomVertexWriter, GeomTristrips, GeomNode
from pandac.PandaModules import OmniBoundingVolume,TransparencyAttrib


class Trail:
    def __init__(self, render, parent, length=500, width=8):
        self.parent = parent
        self.vertex_data = GeomVertexData('data', GeomVertexFormat().getV3c4(), Geom.UHDynamic)

        color_writer = GeomVertexWriter(self.vertex_data, 'color')
        self.length = length + 1
        self.render = render

        color = parent.getColor()
        if color[0] != 0 and color[2] == 0:
            color[0] += 1.0
        elif color[0] == 0 and color[2] != 0:
            color[2] += 1.0
        prim = GeomTristrips(Geom.UHStatic)
        for i in range(self.length):
            prim.addVertex(2 * i)
            prim.addVertex(2 * i + 1)
            alpha = 0.8 * i / self.length
            color_writer.addData4f(color[0], color[1], color[2], alpha)
            color_writer.addData4f(color[0], color[1]+0.8, color[2], alpha)

        prim.closePrimitive()

        geom = Geom(self.vertex_data)
        geom.addPrimitive(prim)
        geom.doublesideInPlace()

        node = GeomNode('trail')
        node.addGeom(geom)
        node.setBounds(OmniBoundingVolume())
        node.setFinal(True)

        self.trail = self.render.attachNewNode(node)
        self.trail.setTransparency(TransparencyAttrib.MDual)

        self.root = parent.attachNewNode('root')

        self.p1 = self.root.attachNewNode('p1')
        self.p1.setX(-width * 0.5)

        self.p2 = self.root.attachNewNode('p2')
        self.p2.setX(width * 0.5)

        self.trail_pos = deque(maxlen=self.length)

    def reset(self):
        self.trail_pos.clear()

    def step(self):
        pos = (self.p1.getPos(self.render), self.p2.getPos(self.render))
        if len(self.trail_pos) == 0:
            for i in range(self.length):
                self.trail_pos.append(pos)
        else:
            self.trail_pos.append(pos)

        vertex_writer = GeomVertexWriter(self.vertex_data, 'vertex')
        for p1_pos, p2_pos in self.trail_pos:
            vertex_writer.setData3f(p1_pos)
            vertex_writer.setData3f(p2_pos)


    def removeNode(self):
        self.p1.removeNode()
        self.p2.removeNode()
        self.root.removeNode()
        self.trail.removeNode()
