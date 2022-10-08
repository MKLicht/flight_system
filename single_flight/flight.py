from math import pi, sin, cos, tan
import math
import numpy as np

from direct.showbase.ShowBase import ShowBase
from direct.task import Task
from direct.actor.Actor import Actor
from panda3d.core import *
from direct.interval.IntervalGlobal import Sequence
from panda3d.core import Point3
from panda3d.core import Vec3
from direct.interval.FunctionInterval import Wait, Func
from panda3d.core import TextNode, TransparencyAttrib
from panda3d.core import LPoint3, LVector3
from direct.gui.OnscreenText import OnscreenText
from pandac.PandaModules import *


PLA_INIT_VEL = 0.5
PLA_TURN_RATE = 2.0
PLA_LIFT_RATE = 180.0
LIFT_ROLL_RATE = 10.0
PLA_ROLL_RATE = 180.0
PLA_LIFT_VEL = 1.0
ENG_ACCEL = 0.2
BRAKE_ACCEL = -0.1
MAX_VEL = 3.0
MIN_VEL = 0.1
MAX_PITCH = 30
MIN_PITCH = -30
TRAIL_LEN = 10000
TRAIL_DIM = 1.0/10000



class Planes(ShowBase):

    def __init__(self):
        ShowBase.__init__(self)

        # background color
        base.setBackgroundColor(0, 0.6, 0.8)

        # set up scene
        self.scene = loader.loadModel("models/back3")
        self.scene.reparentTo(render)
        self.scene.setScale(10, 10, 6)

        # set up ambient light
        ambientLight = AmbientLight('ambientLight')
        ambientLight.setColor((0.1, 0.1, 0.1, 1))
        ambientLightNP = render.attachNewNode(ambientLight)
        render.setLight(ambientLightNP)

        # set up directional light
        dlight = DirectionalLight('dlight')
        dlight.setColor((0.7, 0.9, 1.2, 1))
        dlnp = render.attachNewNode(dlight)
        dlnp.setPos(0, 0, 30)
        dlnp.setHpr(37.3, -50, 0)
        render.setLight(dlnp)

        dlight1 = DirectionalLight('dlight1')
        dlight1.setColor((0.3, 0.3, 0.3, 1))
        dlnp1 = render.attachNewNode(dlight1)
        dlnp1.setPos(20, 20, 50)
        dlnp1.setHpr(0, -90, 0)
        render.setLight(dlnp1)

        #self.taskMgr.add(self.spinCameraTask, "SpinCameraTask")

        # initial pos and hpr
        self.xPos = 0.0
        self.yPos = 0.0
        self.tilt = 0.0
        self.lift = 0.0
        self.head = 0.0

        # set up plane
        #self.plane = loader.loadModel('./models/plane/boeing707')
        self.plane = loader.loadModel('./models/F-16')
        self.plane.reparentTo(render)
        self.plane.setScale(1.0, 1.0, 1.0)
        self.plane.setPosHpr(self.xPos, self.yPos, 50, self.head, self.lift, self.tilt)
        
        # set initial velocity,rate,angle and control
        v = LVector3(0, 1, 0) * PLA_INIT_VEL
        rate = LVector3(0, -0.01, 0)
        ABangle = LVector2(0, 0)
        controls = LVector4(20000, 0.01, -0.09, 0.01)
        self.setVelocity(self.plane, v)
        self.setRate(self.plane, rate)
        self.setABangle(self.plane, ABangle)
        self.setControls(self.plane, controls)

        self.camera1=self.createCamera((0,0.4,0,0.7))
        self.camera1.reparentTo(render)
        self.camera1.lookAt(self.plane)
        self.camera2=self.createCamera((0.4,1.0,0,1))
        self.camera2.reparentTo(render)
        self.camera2.lookAt(self.plane)
        self.camera2.node().setCameraMask(BitMask32.bit(1))
        self.camera3=self.createCamera((0,0.3,0.7,1))
        self.camera3.reparentTo(render)
        self.camera3.setPosHpr(0,0,150,0,-90,0)
        self.camera3.node().setCameraMask(BitMask32.bit(2))
        base.camNode.setActive(False)

        self.plane.hide(BitMask32.bit(2))
        #self.scene.hide(BitMask32.bit(2))

        # set a node to calculate velocity direction 
        self.node = render.attachNewNode("empty")
        self.node.reparentTo(self.plane)
        self.node.setPos(0, 1, 0)

        self.node_v = render.attachNewNode("empty_v")
        self.node_v.reparentTo(self.plane)
        self.node_v.setPos(1, 0, 0)

        self.node_w = render.attachNewNode("empty_w")
        self.node_w.reparentTo(self.plane)
        self.node_w.setPos(0, 0, 1)

        self.cam = render.attachNewNode("cam")
        self.cam.reparentTo(self.plane)
        self.cam.setPosHpr(0, -7, 0, 0, 0, 0)

        # get motion trail 
        #self.trail = loader.loadModel('./models/point')
        #self.trail.setScale(1, 1, 1)
        self.trail = render.attachNewNode("trail")
        self.trail.reparentTo(self.plane)
        self.trail.setPos(0, 0, -0.5)
  
        # input
        self.keys = {"turnLeft": 0, "turnRight": 0, "liftUp": 0, "liftDown": 0,
                     "rollLeft": 0, "rollRight": 0, "accel": 0, "reverse": 0, "fire": 0}

        self.accept('a',             self.setKey, ["turnLeft", 1])
        self.accept('a-up',          self.setKey, ["turnLeft", 0])
        self.accept('d',             self.setKey, ["turnRight", 1])
        self.accept('d-up',          self.setKey, ["turnRight", 0])
        self.accept('w',             self.setKey, ["accel", 1])
        self.accept('w-up',          self.setKey, ["accel", 0])
        self.accept('s',             self.setKey, ["reverse", 1])
        self.accept('s-up',          self.setKey, ["reverse", 0])
        self.accept("arrow_up",      self.setKey, ["liftUp", 1])
        self.accept("arrow_up-up",   self.setKey, ["liftUp", 0])
        self.accept("arrow_down",    self.setKey, ["liftDown", 1])
        self.accept("arrow_down-up", self.setKey, ["liftDown", 0])
        self.accept("arrow_left",    self.setKey, ["rollLeft", 1])
        self.accept("arrow_left-up", self.setKey, ["rollLeft", 0])
        self.accept("arrow_right",    self.setKey, ["rollRight", 1])
        self.accept("arrow_right-up", self.setKey, ["rollRight", 0])
        #self.accept("space",         self.setKey, ["fire", 1])

        self.gameTask = taskMgr.add(self.gameLoop, "gameLoop")

        # prepare for motion trail
        hi = 2
        self.vdata = GeomVertexData('name',GeomVertexFormat.getV3c4(),Geom.UHDynamic)
        self.vertex = GeomVertexWriter(self.vdata, 'vertex')        
        self.color = GeomVertexWriter(self.vdata, 'color')

        #trail length
        self.nr=10000   
        self.dim=1.0/self.nr

        for i in range(self.nr):
            alpha=(10000.0-i)/10000.0
            if(alpha < 0):
                alpha = 0 

            self.vertex.addData3f(1.0,1.0,0)
            self.color.addData4f(1,0,0,alpha/2.0)

        prim = GeomTristrips(Geom.UHDynamic)
        for i in range(self.nr):
            prim.addVertex(i)
        prim.closePrimitive()

        self.geom=Geom(self.vdata)
        self.geom.addPrimitive(prim)
                 
        self.gnode=GeomNode('gnode')
        self.gnode.addGeom(self.geom)
        
        self.geom.doublesideInPlace()
         
        self.gnodePath = render.attachNewNode(self.gnode)        
        self.gnodePath.setTransparency(TransparencyAttrib.MAlpha)
        
        self.root=self.trail.attachNewNode('root')
        self.root.hide(BitMask32.bit(1))
                
        self.p1=self.root.attachNewNode('p1')
        self.p1.setX(hi*0.5)
        
        self.p2=self.root.attachNewNode('p2')
        self.p2.setX(-hi*0.5)
        
        self.trpos=[]
        pos=[self.p1.getPos(),self.p2.getPos()]
        
        for i in range(self.nr>>1):
            self.trpos.append((pos))

    #set camera
    def createCamera(self,dispRegion):
        camera=base.makeCamera(base.win,displayRegion=dispRegion)
        camera.node().getLens().setAspectRatio(3.0/4.0)
        camera.node().getLens().setFov(45) #optional.
        return camera
    
    #def spinCameraTask(self, task):
        #self.camera.setPos(self.cam.getX(render), self.cam.getY(render), self.plane.getZ())
        #self.camera.setHpr(self.cam.getH(render), 0, 0)
        #self.camera.lookAt(self.plane)
        #self.camera.setPos(0, 0, 70)
        #self.camera.setHpr(0, -90, 0)
        #return Task.cont

    def setKey(self, key, val):
        self.keys[key] = val
    
    def setVelocity(self, obj, val):
        obj.setPythonTag("velocity", val)

    def getVelocity(self, obj):
        return obj.getPythonTag("velocity")

    def setRate(self, obj, val):
        obj.setPythonTag("rate", val)

    def getRate(self, obj):
        return obj.getPythonTag("rate")

    def setABangle(self, obj, val):
        obj.setPythonTag("ABangle", val)

    def getABangle(self, obj):
        return obj.getPythonTag("ABangle")

    def setControls(self, obj, val):
        obj.setPythonTag("controls", val)

    def getControls(self, obj):
        return obj.getPythonTag("controls")

    # loop game
    def gameLoop(self, Task):
        dt = 0.1
        self.updatePla(dt)
        self.camera1.setPos(self.plane.getX(), self.plane.getY(), 80)
        self.camera1.setHpr(0, -90, 0)
        #self.camera2.setPos(self.cam.getX(render), self.cam.getY(render), self.cam.getZ(render))
        #self.camera2.setHpr(self.cam.getH(render), self.cam.getP(render), self.cam.getR(render))
        self.camera2.setPos(self.plane.getX(), self.plane.getY()-7, self.plane.getZ())
        self.camera2.setHpr(0, 0, 0)

        #show motion trail
        self.trpos.pop()
        self.trpos.insert(0,[self.p1.getPos(render),self.p2.getPos(render)])
        pos = GeomVertexWriter(self.vdata, 'vertex')

        for i in range(1,self.nr>>1):  
            npos=self.root.getPos(render)
            xr,yr,zr=self.root.getPos()
            x1,y1,z1=self.trpos[i][0]
            x2,y2,z2=self.trpos[i][1]
            xx=x1-x2
            yy=y1-y2
            zz=z1-z2

            xdim=self.dim*xx
            ydim=self.dim*yy
            zdim=self.dim*zz
            
            pos.setData3f(x1-xdim,y1-ydim,z1-zdim)
            pos.setData3f(x2+xdim,y2+ydim,z2+zdim)
        
        self.Dynamics(self.plane, dt)

        return Task.cont

    # update velocity 
    #def updateVel(self, obj1, obj2):
    #    vel = self.getVelocity(obj1)
    #    actual_vel = math.sqrt(vel.getX()*vel.getX()+vel.getY()*vel.getY()+vel.getZ()*vel.getZ())
    #    if(actual_vel < 0.1):
    #        actual_vel = 0.1
    #        print("Warning: You should speed up the plane")
    #    v1 = obj2.getPos(render)-obj1.getPos()
    #    v1.normalize()
    #    self.setVelocity(obj1, v1*actual_vel)

    # update position

    def EulerAngle(self, P, Q, R):
        heading, pitch, roll = self.plane.getHpr()
        phi, theta, psi = roll, pitch, -heading
        #P, Q, R = self.getRate(self.plane)
        sphi = sin(phi*pi/180)
        cphi = cos(phi*pi/180)
        st = sin(theta*pi/180)
        ct = cos(theta*pi/180)
        tt = tan(theta*pi/180)
        spsi = sin(psi*pi/180)
        cpsi = cos(psi*pi/180)
        if (pitch/90)%2 != 1:
            phi += P + tt*(Q*sphi + R*cphi)
            theta += Q*cphi - R*sphi
            psi += (Q*sphi + R*cphi)/ct
        else:
            phi += 0
            theta += Q*cphi - R*sphi
            psi += 0
        self.plane.setHpr(-psi, theta, phi)

    def updatePla(self, dt):
        heading = self.plane.getH() 
        pitch = self.plane.getP() 
        roll = self.plane.getR()
        vel = self.getVelocity(self.plane)

        # change heading if q or e is being pressed
        if self.keys["turnRight"]:
            p, q, r = self.getRate(self.plane)
            rate = LVector3(p, q, 0.005)
            self.setRate(self.plane, rate)

            T, ail, el, rud = self.getControls(self.plane)
            controls = LVector4(T, ail, el, -10.0)
            self.setControls(self.plane, controls)
            self.EulerAngle(p, q, 0.1)
            #heading = dt * PLA_TURN_RATE
            #self.plane.setH(self.plane, -heading % 360)
        elif self.keys["turnLeft"]:
            p, q, r = self.getRate(self.plane)
            rate = LVector3(p, q, -0.005)
            self.setRate(self.plane, rate)

            T, ail, el, rud = self.getControls(self.plane)
            controls = LVector4(T, ail, el, 10.0)
            self.setControls(self.plane, controls)
            self.EulerAngle(p, q, -0.1)
            #heading = dt * PLA_TURN_RATE
            #self.plane.setH(self.plane, heading % 360)

        # accelerate or reverse if w or s is being pressed
        if self.keys["accel"]:
            T, ail, el, rud = self.getControls(self.plane)
            controls = LVector4(80000, ail, el, rud)
            self.setControls(self.plane, controls)

            #v1 = self.node.getPos(render)-self.plane.getPos()
            #v1.normalize()
            #newVel = v1 * ENG_ACCEL * dt
            #newVel += self.getVelocity(self.plane)
            #if newVel.lengthSquared() > MAX_VEL:
            #    newVel.normalize()
            #    newVel *= MAX_VEL
            #self.setVelocity(self.plane, newVel)

        elif self.keys["reverse"]:
            T, ail, el, rud = self.getControls(self.plane)
            controls = LVector4(0, ail, el, rud)
            self.setControls(self.plane, controls)

            #v1 = self.node.getPos(render)-self.plane.getPos()
            #v1.normalize()
            #newVel = v1 * BRAKE_ACCEL * dt
            #newVel += self.getVelocity(self.plane)
            #if newVel.lengthSquared() < MIN_VEL:
            #    newVel.normalize()
            #    newVel *= MIN_VEL
            #self.setVelocity(self.plane, newVel)

        # lift up or down if allow-up or allow-down is being presses
        if self.keys["liftUp"]:
            p, q, r = self.getRate(self.plane)
            rate = LVector3(p, 0.02, r)
            self.setRate(self.plane, rate)

            T, ail, el, rud = self.getControls(self.plane)
            controls = LVector4(T, ail, -30, rud)
            self.setControls(self.plane, controls)
            self.EulerAngle(p, 5.0, r)
            #self.plane.setP(self.plane, pitch)
        elif self.keys["liftDown"]:
            T, ail, el, rud = self.getControls(self.plane)
            controls = LVector4(T, ail, -30, rud)
            self.setControls(self.plane, controls)
            #rate = LVector3(0, -5, 0)
            #self.setRate(self.plane, rate)
            #self.Dynamics(self.plane)
            p, q, r = self.getRate(self.plane)
            rate = LVector3(p, -0.02, r)
            self.setRate(self.plane, rate)
            self.EulerAngle(p, -5.0, r)
            #pitch = dt * PLA_LIFT_RATE
            #self.plane.setP(self.plane, -pitch)

        if self.keys["rollLeft"]:
            #rate = LVector3(10, 0, 0)
            #self.setRate(self.plane, rate)

            p, q, r = self.getRate(self.plane)
            rate = LVector3(0.02, q, r)
            self.setRate(self.plane, rate)

            T, ail, el, rud = self.getControls(self.plane)
            controls = LVector4(T, 21.5, el, rud)
            self.setControls(self.plane, controls)
            self.EulerAngle(5.0, q, r)
            #roll = dt * PLA_ROLL_RATE
            #self.plane.setR(self.plane, roll % 360)
        elif self.keys["rollRight"]:
            #rate = LVector3(-10, 0, 0)
            #self.setRate(self.plane, rate)
            p, q, r = self.getRate(self.plane)
            rate = LVector3(-0.02, q, r)
            self.setRate(self.plane, rate)

            T, ail, el, rud = self.getControls(self.plane)
            controls = LVector4(T, -21.5, el, rud)
            self.setControls(self.plane, controls)
            self.setControls(self.plane, controls)
            self.EulerAngle(-5.0, q, r)
            #roll = dt * PLA_ROLL_RATE
            #self.plane.setR(self.plane, -roll % 360)

        #self.updateVel(self.plane, self.node)
        #self.updatePos(self.plane, dt)

    def fix(self, dinput):
        if(dinput >= 0.0):
            dout = math.floor(dinput)
        elif(dinput < 0.0):
            dout = math.ceil(dinput)
        return dout
    
    def sign(self, dinput):
        if(dinput > 0.0):
            dout = 1
        elif(dinput < 0.0):
            dout = -1
        elif(dinput == 0.0):
            dout = 0
        return dout

    #########damping aero-coeffs###########
    def damping(self, alpha, coeff):
        A = np.array([
        [-.267, .110,  .308,  1.34,  2.08,  2.91,  2.76,  2.05,   1.5,  1.49,  1.83,  1.21],
        [ .882,  .852,  .876,  .958,  .962,  .974,  .819,  .483,  .590,  1.21, -.493, -1.04],
        [-.108, -.108, -.188,  .110,  .258,  .226,  .344,  .362,  .611,  .529,  .298, -2.27],
        [ -8.8, -25.8, -28.9, -31.4, -31.2, -30.7, -27.7, -28.2,   -29, -29.8, -38.3, -35.3],
        [-.126, -.026,  .063,  .113,  .208,  .230,  .319,  .437,  .680,    .1,  .447, -.330],
        [ -.36, -.359, -.443,  -.42, -.383, -.375, -.329, -.294,  -.23,  -.21,  -.12,   -.1],
        [-7.21,  -.54, -5.23, -5.26, -6.11, -6.64, -5.69,    -6,  -6.2,  -6.4,  -6.6,    -6],
        [ -.38, -.363, -.378, -.386,  -.37, -.453,  -.55, -.582, -.595, -.637, -1.02,  -.84],
        [ .061,  .052,  .052, -.012, -.013, -.024,   .05,   .15,   .13,  .158,   .24,   .15]])

        s = .2*alpha
        k = self.fix(s)
        if(k <= -2):
            k = -1
        elif(k >= 9):
            k = 8
        da = s - k
        L = k + self.fix(1.1*self.sign(da))
        k = int(k + 3)
        L = int(L + 3)

        coeff[0] = A[0][k-1] + math.fabs(da)*(A[0][L-1] - A[0][k-1])
        coeff[1] = A[1][k-1] + math.fabs(da)*(A[1][L-1] - A[1][k-1])
        coeff[2] = A[2][k-1] + math.fabs(da)*(A[2][L-1] - A[2][k-1])
        coeff[3] = A[3][k-1] + math.fabs(da)*(A[3][L-1] - A[3][k-1])
        coeff[4] = A[4][k-1] + math.fabs(da)*(A[4][L-1] - A[4][k-1])
        coeff[5] = A[5][k-1] + math.fabs(da)*(A[5][L-1] - A[5][k-1])
        coeff[6] = A[6][k-1] + math.fabs(da)*(A[6][L-1] - A[6][k-1])
        coeff[7] = A[7][k-1] + math.fabs(da)*(A[7][L-1] - A[7][k-1])	
        coeff[8] = A[8][k-1] + math.fabs(da)*(A[8][L-1] - A[8][k-1])

    ##########control inputs###############
    def dmomdocon(self, alpha, beta, coeff):
        ALA = np.array([
        [-.041, -.052, -.053, -.056, -.050, -.056, -.082, -.059, -.042, -.038, -.027, -.017],
        [-.041, -.053, -.053, -.053, -.050, -.051, -.066, -.043, -.038, -.027, -.023, -.016],
        [-.042, -.053, -.052, -.051, -.049, -.049, -.043, -.035, -.026, -.016, -.018, -.014],
        [-.040, -.052, -.051, -.052, -.048, -.048, -.042, -.037, -.031, -.026, -.017, -.012],
        [-.043, -.049, -.048, -.049, -.043, -.042, -.042, -.036, -.025, -.021, -.016, -.011],
        [-.044, -.048, -.048, -.047, -.042, -.041, -.020, -.028, -.013, -.014, -.011, -.010],
        [-.043, -.049, -.047, -.045, -.042, -.037, -.003, -.013, -.010, -.003, -.007, -.008]])

        ALR = np.array([
        [.005, .017, .014, .010, -.005, .009, .019, .005,   0.0, -.005, -.011, .008],
        [.007, .016, .014, .014,  .013, .009, .012, .005,   0.0,  .004,  .009, .007],
        [.013, .013, .011, .012,  .011, .009, .008, .005, -.002,  .005,  .003, .005],
        [.018, .015, .015, .014,  .014, .014, .014, .015,  .013,  .011,  .006, .001],
        [.015, .014, .013, .013,  .012, .011, .011, .010,  .008,  .008,  .007, .003],
        [.021, .011, .010, .011,  .010, .009, .008, .010,  .006,  .005,   0.0, .001],
        [.023, .010, .011, .011,  .011, .010, .008, .010,  .006,  .014,  .020,  0.0]])

        ANA = np.array([
        [ .001, -.027, -.017, -.013, -.012, -.016,  .001,  .017,  .011, .017,  .008, .016],
        [ .002, -.014, -.016, -.016, -.014, -.019, -.021,  .002,  .012, .016,  .015, .011],
        [-.006, -.008, -.006, -.006, -.005, -.008, -.005,  .007,  .004, .007,  .006, .006],
        [-.011, -.011, -.010, -.009, -.008, -.006,   0.0,  .004,  .007, .010,  .004, .010],
        [-.015, -.015, -.014, -.012, -.011, -.008, -.002,  .002,  .006, .012,  .011, .011],
        [-.024, -.010, -.004, -.002, -.001,  .003,  .014,  .006, -.001, .004,  .004, .006],
        [-.022,  .002, -.003, -.005, -.003, -.001, -.009, -.009, -.001, .003, -.002, .001]])

        ANR = np.array([
        [-.018, -.052, -.052, -.052, -.054, -.049, -.059, -.051, -.030, -.037, -.026, -.013],
        [-.028, -.051, -.043, -.046, -.045, -.049, -.057, -.052, -.030, -.033, -.030, -.008],
        [-.037, -.041, -.038, -.040, -.040, -.038, -.037, -.030, -.027, -.024, -.019, -.013],
        [-.048, -.045, -.045, -.045, -.044, -.045, -.047, -.048, -.049, -.045, -.033, -.016],
        [-.043, -.044, -.041, -.041, -.040, -.038, -.034, -.035, -.035, -.029, -.022, -.009],
        [-.052, -.034, -.036, -.036, -.035, -.028, -.024, -.023, -.020, -.016, -.010, -.014],
        [-.062, -.034, -.027, -.028, -.027, -.027, -.023, -.023, -.019, -.009, -.025, -.010] 
        ]) 

        s = 0.2*(alpha)
        k = self.fix(s) 
        if(k <= -2):
            k = -1
        elif(k >= 9):
            k = 8
        da = s - k
        L = k + self.fix(1.1*self.sign(da)) 
        s = 0.2*math.fabs(beta)
        m = self.fix(s)
       #if(m >= 7):
        #    m = 6
        if(m == 0):
            m = 1
        elif(m >= 6):
            m = 5
        
        db = s - m
        n = m + 1  
        k = int(k + 3)
        L = int(L + 3)
        m = int(m + 1)
        n = int(n + 1)

        t = ALA[m-1][k-1]
        u = ALA[n-1][k-1]
        v = t + math.fabs(da)*(ALA[m-1][L-1] - t)
        w = u + math.fabs(da)*(ALA[n-1][L-1] - u)
        dlda = v + (w-v)*db

        t = ALR[m-1][k-1]
        u = ALR[n-1][k-1]
        v = t + math.fabs(da)*(ALR[m-1][L-1] - t)
        w = u + math.fabs(da)*(ALR[n-1][L-1] - u)
        dldr = v + (w-v)*db

        t = ANA[m-1][k-1]
        u = ANA[n-1][k-1]
        v = t + math.fabs(da)*(ANA[m-1][L-1] - t)
        w = u + math.fabs(da)*(ANA[n-1][L-1] - u)
        dnda = v + (w-v)*db

        t = ANR[m-1][k-1]
        u = ANR[n-1][k-1]
        v = t + math.fabs(da)*(ANR[m-1][L-1] - t)
        w = u + math.fabs(da)*(ANR[n-1][L-1] - u)
        dndr = v + (w-v)*db

        coeff[0] = dlda
        coeff[1] = dldr
        coeff[2] = dnda
        coeff[3] = dndr

    ##########Cl and Cn aero-coeff#########
    def clcn(self, alpha, beta, coeff):
        AL = np.array([
        [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
        [-.001,-.004,-.008,-.012,-.016,-.019,-.020,-.020,-.015,-.008,-.013,-.015],
        [-.003,-.009,-.017,-.024,-.030,-.034,-.040,-.037,-.016,-.002,-.010,-.019],
        [-.001,-.010,-.020,-.030,-.039,-.044,-.050,-.049,-.023,-.006,-.014,-.027],
        [    0,-.010,-.022,-.034,-.047,-.046,-.059,-.061,-.033,-.036,-.035,-.035],
        [ .007,-.010,-.023,-.034,-.049,-.046,-.068,-.071,-.060,-.058,-.062,-.059],
        [ .009,-.011,-.023,-.037,-.050,-.047,-.074,-.079,-.091,-.076,-.077,-.076] 
        ])

        AN = np.array([
        [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
        [ .018, .019, .018, .019, .019, .018, .013, .007, .004, -.014, -.017, -.033],
        [ .038, .042, .042, .042, .043, .039, .030, .017, .004, -.035, -.047, -.057],
        [ .056, .057, .059, .058, .058, .053, .032, .012, .002, -.046, -.071, -.073],
        [ .064, .077, .076, .074, .073, .057, .029, .007, .012, -.034, -.065, -.041],
        [ .074, .086, .093, .089, .080, .062, .049, .022, .028, -.012, -.002, -.013],
        [ .079, .090, .106, .106, .096, .080, .068, .030, .064,  .015,  .011, -.001] 
        ])

        s = .2*(alpha)
        k = self.fix(s)
        if(k <= -2):
            k = -1
        elif(k >= 9):
            k = 8
        da = s - k
        L = k + self.fix(1.1*self.sign(da))
        s = .2*math.fabs(beta)
        m = self.fix(s)
        if(m == 0):
            m = 1
        elif(m >= 6):
            m = 5
        db = s - m
        n = m + self.fix(1.1*self.sign(db))
        k = int(k + 3)
        L = int(L + 3)
        m = int(m + 1)
        n = int(n + 1)

        t = AL[m-1][k-1]
        u = AL[n-1][k-1]
        v = t + math.fabs(da)*(AL[m-1][L-1] - t)
        w = u + math.fabs(da)*(AL[n-1][L-1] - u)
        dum = v + (w-v)*math.fabs(db)
        cl = dum*self.sign(beta)

        t = AN[m-1][k-1]
        u = AN[n-1][k-1]
        v = t + math.fabs(da)*(AN[m-1][L-1] - t)
        w = u + math.fabs(da)*(AN[n-1][L-1] - u)
        dum = v + (w-v)*math.fabs(db)
        cn = dum*self.sign(beta)

        coeff[0] = cl
        coeff[1] = cn

    ##########Cx and Cm aero-coeffs########
    def cxcm(self, alpha, dele, coeff):
        AX = np.array([
        [-.099, -.081, -.081, -.063, -.025, .044, .097, .113, .145, .167, .174, .166],
        [-.048, -.038, -.040, -.021,  .016, .083, .127, .137, .162, .177, .179, .167],
        [-.022, -.020, -.021, -.004,  .032, .094, .128, .130, .154, .161, .155, .138],
        [-.040, -.038, -.039, -.025,  .006, .062, .087, .085, .100, .110, .104, .091],
        [-.083, -.073, -.076, -.072, -.046, .012, .024, .025, .043, .053, .047, .040]])

        AM = np.array([
        [ .205,  .168,  .186,  .196,  .213,  .251,  .245,  .238,  .252,  .231,  .198,  .192],
        [ .081,  .077,  .107,  .110,  .110,  .141,  .127,  .119,  .133,  .108,  .081,  .093],
        [-.046, -.020, -.009, -.005, -.006,  .010,  .006, -.001,  .014,   0.0, -.013,  .032],
        [-.174, -.145, -.121, -.127, -.129, -.102, -.097, -.113, -.087, -.084, -.069, -.006],
        [-.259, -.202, -.184, -.193, -.199, -.150, -.160, -.167, -.104, -.076, -.041, -.005]])

        s = .2*(alpha)
        k = self.fix(s)
        if(k <= -2):
            k = -1
        elif(k >= 9):
            k = 8
        da = s - k
        L = k + self.fix(1.1*self.sign(da))
        s = dele/12.0
        m = self.fix(s)
        if(m <= -2):
            m = -1
        elif(m >= 2):
            m = 1
        de = s - m
        n = m + self.fix(1.1*self.sign(de))
        k = int(k + 3)
        L = int(L + 3)
        m = int(m + 3)
        n = int(n + 3)

        t = AX[m-1][k-1]
        u = AX[n-1][k-1]
        v = t + math.fabs(da)*(AX[m-1][L-1] - t)
        w = u + math.fabs(da)*(AX[n-1][L-1] - u)
        cx = v + (w-v)*math.fabs(de)

        t = AM[m-1][k-1]
        u = AM[n-1][k-1]
        v = t + math.fabs(da)*(AM[m-1][L-1] - t)
        w = u + math.fabs(da)*(AM[n-1][L-1] - u)
        cm = v + (w-v)*math.fabs(de)

        coeff[0] = cx
        coeff[1] = cm

    ##########Cz aero-coeff################
    def cz(self, alpha, beta, dele, coeff):
        A = np.array([.770, .241, -.100, -.416, -.731, -1.053, -1.366, 
                    -1.646, -1.917, -2.120, -2.248, -2.229])
        s = .2*(alpha)
        k = self.fix(s)
        if(k <= -2):
            k = -1
        elif(k >= 9):
            k = 8
        da = s - k
        L = k + self.fix(1.1*self.sign(da))

        k = int(k + 3)
        L = int(L + 3)

        s = A[k-1] + math.fabs(da)*(A[L-1]-A[k-1])
        cz = s*(1-math.pow((beta/57.3),2))-.19*(dele)/25
        coeff[0] = cz

    ##########atmospheric effect###########
    def atmos(self, alt, vt, coeff):
        vt *= vt * 100.0
        rho0 = 2.377e-3
        tfac = 1- .703e-5*(alt)
        if(tfac < 0):
            print("Warning: The aircraft overflew the altitude limit")
        temp = 519.0 * tfac
        if(alt >= 35000.0):
            temp = 390
        rho = rho0 * pow(tfac, 4.14)
        mach = (vt)/math.sqrt(1.4*1716.3*temp)
        qbar = .5*rho*pow(vt,2)
        ps = 1715.0*rho*temp
        if(ps == 0):
            ps = 1715
        
        coeff[0] = mach
        coeff[1] = qbar
        coeff[2] = ps

    ##########flight dynamics##############
    def Dynamics(self, obj, dt):
        #F16 Nasa Data
        g = 9.8 #gravity
        m = 9295.44 #mass
        B = 9.144 #span
        S = 27.87 #planform area
        cbar = 3.45 #mean aero chord
        xcgr = 0.35
        xcg = 0.30

        Heng = 0.0
        pi = math.acos(-1)
        r2d = 180.0/pi

        Jy = 75674
        Jxz = 1331
        Jz = 85552
        Jx = 12875

        # position and altitude 
        y, x, z = obj.getPos()
        # orientation angles in degrees
        heading, pitch, roll = obj.getHpr()
        # directional velocities
        V, U, W = self.getVelocity(obj)
        vt = math.sqrt(U*U + V*V + W*W)
        # roll rate, pitch rate and yaw rate
        P, Q, R = self.getRate(obj)
        # angle of attack and sideslip angle
        alpha, beta = self.getABangle(obj)
        # controls: thrust, aileron, elevator, rudder
        T, ail, el, rud = self.getControls(obj)

        R = -R

        alt = z * 500

        # normalized aganist max angle
        dail = ail/21.5
        drud = rud/30.0
        #dlef = (1 - lef/25.0)

        ######atmospheric effects####
        temp = [0, 0, 0]
        self.atmos(alt, vt, temp)
        mach = temp[0]
        qbar = temp[1]
        ps = temp[2]

        ############AeroData###############
        dlef = 0.0
        temp1 = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.damping(alpha,temp1)
        Cxq = temp1[0]
        Cyr = temp1[1]
        Cyp = temp1[2]
        Czq = temp1[3]
        Clr = temp1[4]
        Clp = temp1[5]
        Cmq = temp1[6]
        Cnr = temp1[7]
        Cnp = temp1[8]

        temp2 = [0, 0, 0, 0]
        self.dmomdocon(alpha,beta, temp2)
        delta_Cl_a20 = temp2[0]
        delta_Cl_r30 = temp2[1]
        delta_Cn_a20 = temp2[2]
        delta_Cn_r30 = temp2[3]

        temp3 = [0, 0]
        self.clcn(alpha,beta,temp3)
        Cl = temp[0]
        Cn = temp[1]

        temp4 = [0, 0]
        self.cxcm(alpha,el,temp4)
        Cx = temp4[0]
        Cm = temp4[1]

        temp5 = [0]
        self.cz(alpha,beta,el,temp5)
        Cz = temp5[0]

        Cy = -.02*beta + .021*dail + .086*drud

        delta_Cx_lef    = 0.0
        delta_Cz_lef    = 0.0
        delta_Cm_lef    = 0.0
        delta_Cy_lef    = 0.0
        delta_Cn_lef    = 0.0
        delta_Cl_lef    = 0.0
        delta_Cxq_lef   = 0.0
        delta_Cyr_lef   = 0.0
        delta_Cyp_lef   = 0.0
        delta_Czq_lef   = 0.0
        delta_Clr_lef   = 0.0
        delta_Clp_lef   = 0.0
        delta_Cmq_lef   = 0.0
        delta_Cnr_lef   = 0.0
        delta_Cnp_lef   = 0.0
        delta_Cy_r30    = 0.0
        delta_Cy_a20    = 0.0
        delta_Cy_a20_lef= 0.0
        delta_Cn_a20_lef= 0.0
        delta_Cl_a20_lef= 0.0
        delta_Cnbeta    = 0.0
        delta_Clbeta    = 0.0
        delta_Cm        = 0.0
        eta_el          = 1.0   
        delta_Cm_ds     = 0.0

        ########C_tot###############
        dXdQ = (cbar/(2*vt))*(Cxq + delta_Cxq_lef*dlef)
        Cx_tot = Cx + delta_Cx_lef*dlef + dXdQ*Q

        dZdQ = (cbar/(2*vt))*(Czq + delta_Cz_lef*dlef)
        Cz_tot = Cz + delta_Cz_lef*dlef + dZdQ*Q

        dMdQ = (cbar/(2*vt))*(Cmq + delta_Cmq_lef*dlef)
        Cm_tot = Cm*eta_el + Cz_tot*(xcgr-xcg) + delta_Cm_lef*dlef + dMdQ*Q + delta_Cm + delta_Cm_ds

        dYdail = delta_Cy_a20 + delta_Cy_a20_lef*dlef
        dYdR = (B/(2*vt))*(Cyr + delta_Cyr_lef*dlef)
        dYdP = (B/(2*vt))*(Cyp + delta_Cyp_lef*dlef)
        Cy_tot = Cy + delta_Cy_lef*dlef + dYdail*dail + delta_Cy_r30*drud + dYdR*R + dYdP*P

        dNdail = delta_Cn_a20 + delta_Cn_a20_lef*dlef
        dNdR = (B/(2*vt))*(Cnr + delta_Cnr_lef*dlef)
        dNdP = (B/(2*vt))*(Cnp + delta_Cnp_lef*dlef)
        Cn_tot = Cn + delta_Cn_lef*dlef - Cy_tot*(xcgr-xcg)*(cbar/B) + dNdail*dail + delta_Cn_r30*drud + dNdR*R + dNdP*P + delta_Cnbeta*beta

        dLdail = delta_Cl_a20 + delta_Cl_a20_lef*dlef
        dLdR = (B/(2*vt))*(Clr + delta_Clr_lef*dlef)
        dLdP = (B/(2*vt))*(Clp + delta_Clp_lef*dlef)        
        Cl_tot = Cl + delta_Cl_lef*dlef + dLdail*dail + delta_Cl_r30*drud + dLdR*R + dLdP*P + delta_Clbeta*beta
      
        ########Compute UVW##########
        Udot = R*V - Q*W - g*sin(pitch*pi/180) + qbar*S*Cx_tot/m + T/m
        Vdot = P*W - R*U + g*cos(pitch*pi/180)*sin(roll*pi/180) + qbar*S*Cy_tot/m
        Wdot = -(Q*U - P*V + g*cos(pitch*pi/180)*cos(roll*pi/180) + qbar*S*Cz_tot/m)
        vt_dot = (U*Udot + V*Vdot + W*Wdot)/vt
        #print(Udot, Vdot, Wdot)

        v, u, w = self.getVelocity(self.plane)
        if(v > 0):
            v += Vdot/10000.0
        elif(v < 0):
            v -= Vdot/10000.0
        if(u > 0):
            u += Udot/10000.0
        elif(u < 0):
            u -= Udot/10000.0
        if(w > 0):
            w += Wdot/10000.0
        if(w < 0):
            w -= Wdot/10000.0
        velocity_dot = LVector3(v, u, w)
        self.setVelocity(self.plane, velocity_dot)
        #print(v, u, w)

        #########set alpha, beta######
        if(U > 0):
            alpha_dot = -math.atan(W/U)*r2d
        elif(U == 0):
            alpha_dot = 90
        elif(U < 0):
            alpha_dot = math.atan(W/U)*r2d
        beta_dot = math.asin(V/vt)*r2d
        ABangle = LVector2(alpha_dot, beta_dot)
        self.setABangle(self.plane, ABangle)

        ########Compute PQR##########
        L_tot = Cl_tot*qbar*S*B
        M_tot = Cm_tot*qbar*S*cbar
        N_tot = Cn_tot*qbar*S*B

        denom = Jx*Jz - Jxz*Jxz

        #R = -R
        P_tot = (Jz*L_tot + Jxz*N_tot - (Jz*(Jz-Jy)+Jxz*Jxz)*Q*R + Jxz*(Jx-Jy+Jz)*P*Q + Jxz*Q*Heng)/denom
        Q_tot = (M_tot + (Jz-Jx)*P*R - Jxz*(P*P-R*R) - R*Heng)/Jy
        R_tot = (Jx*N_tot + Jxz*L_tot + (Jx*(Jx-Jy)+Jxz*Jxz)*P*Q - Jxz*(Jx-Jy+Jz)*Q*R + Jx*Q*Heng)/denom
        P += P_tot
        Q += Q_tot
        R += R_tot
        rate = LVector3(P, Q, -R)
        self.setRate(self.plane, rate)

        ########Euler Angle##########
        heading, pitch, roll = self.plane.getHpr()
        phi, theta, psi = roll, pitch, -heading
        #P, Q, R = self.getRate(self.plane)
        sphi = sin(phi*pi/180)
        cphi = cos(phi*pi/180)
        st = sin(theta*pi/180)
        ct = cos(theta*pi/180)
        tt = tan(theta*pi/180)
        spsi = sin(psi*pi/180)
        cpsi = cos(psi*pi/180)
        if (pitch/90)%2 != 1:
            phi += P + tt*(Q*sphi + R*cphi)
            theta += Q*cphi - R*sphi
            psi += (Q*sphi + R*cphi)/ct
        else:
            phi += 0
            theta += Q*cphi - R*sphi
            psi += 0
        self.plane.setHpr(-psi, theta, phi)

        V, U, W = self.getVelocity(self.plane)
        phi, theta, psi = roll, pitch, -heading
        pi = math.acos(-1)
        sphi = sin(phi*pi/180)
        cphi = cos(phi*pi/180)
        st = sin(theta*pi/180)
        ct = cos(theta*pi/180)
        spsi = sin(psi*pi/180)
        cpsi = cos(psi*pi/180)
        x_dot = U*(ct*spsi) + V*(sphi*spsi*st + cphi*cpsi) + W*(cphi*st*spsi - sphi*cpsi)
        y_dot = U*(ct*cpsi) + V*(sphi*cpsi*st - cphi*spsi) + W*(cphi*st*cpsi + sphi*spsi)
        z_dot = U*st - V*(sphi*ct) - W*(cphi*ct)
        newPos = obj.getPos() + (x_dot*dt, y_dot*dt, z_dot*dt)
        obj.setPos(newPos)

        if(Q>0.00001 and Q<0.1):
            rate = LVector3(P, Q-0.00005, R)
            self.setRate(self.plane, rate)
        elif(Q < 0.00001):
            rate = LVector3(P, Q+0.00005, R)
            self.setRate(self.plane, rate)
        elif(abs(Q)>0.1):
            Q = 0.0
            rate = LVector3(P, Q, R)
            self.setRate(self.plane, rate)

        if(P > 0.00001):
            rate = LVector3(P-0.00005, Q, R)
            self.setRate(self.plane, rate)
        elif(P < -0.00001):
            rate = LVector3(P+0.00005, Q, R)
            self.setRate(self.plane, rate)
        elif(abs(P)>0.1):
            P = 0.0
            rate = LVector3(P, Q, R)
            self.setRate(self.plane, rate)

        if(R > 0.00001):
            rate = LVector3(P, Q, R-0.0005)
            self.setRate(self.plane, rate)
        elif(R < -0.00001):
            rate = LVector3(P, Q, R+0.0005)
            self.setRate(self.plane, rate)
        elif(abs(R)>0.1):
            R = 0.0
            rate = LVector3(P, Q, R)
            self.setRate(self.plane, rate)

        controls = LVector4(20000, 0, 0, 0)
        self.setControls(self.plane, controls)
        
       
app = Planes()
app.run()