from direct.gui.OnscreenText import OnscreenText
from direct.showbase.ShowBase import ShowBase
from direct.task import Task
from pandac.PandaModules import *

from client_thread import ClientThread
from trail import Trail

value_add = 1
value_sub = -1
value_empty = 0


class RenderApp(ShowBase):
    def __init__(self, host='localhost', port=9020):
        ShowBase.__init__(self)
        self.thread = ClientThread(self, host, port)
        self.thread.setDaemon(True)

        self.set_light()

        self.ground = self.loader.loadModel("bin/ground")
        self.ground.reparentTo(self.render)
        self.ground.setScale(10, 10, 2)

        self.state_text = self.set_text()

        self.fighter_cam = self.render.attachNewNode('fighter_cam')
        self.focus = -1
        self.track(-1)
        self.fighters = []
        self.trails = []

        self.cs = -1
        self.red_num = 0
        self.fighter_num = 0
        self.taskMgr.add(self.step, 'step')
        self.action = [value_empty, value_empty, value_empty, value_empty]
        self.set_accept()
        self.thread.start()

    def createCamera(self,dispRegion):
        camera=base.makeCamera(base.win,displayRegion=dispRegion)
        camera.node().getLens().setAspectRatio(3.0/4.0)
        camera.node().getLens().setFov(45) #optional.
        return camera
    
    def track(self, i):
        if 0 <= i < len(self.fighters):
            self.focus = i
            self.fighter_cam.reparentTo(self.fighters[i])
            self.fighter_cam.setPos(0, -50, 5)
            self.fighter_cam.setHpr(0, -5, 0)

        elif i == -2:
            if self.fighter_num == 2:
                self.focus = -2
                self.fighter_cam.reparentTo(self.render)
                p1 = self.fighters[0].getPos()
                p2 = self.fighters[1].getPos()
                x, y, z = (p1 + p2) / 2
                dist = ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5
                self.fighter_cam.setPos(x, y, z + 20 * (1 + dist ** 0.5))
                self.fighter_cam.setHpr(0, -90, 0)

        elif i == -3:
            self.focus = -3
            self.fighter_cam.reparentTo(self.render)
            p1 = self.fighters[0].getPos()
            p2 = self.fighters[1].getPos()
            x, y, z = (p1 + p2) / 2
            dist = ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5
            self.fighter_cam.setPos(x, y - 20 * (1 + dist ** 0.5), z)
            self.fighter_cam.setHpr(0, 0, 0)

        else:
            self.focus = -1
            self.fighter_cam.reparentTo(self.render)
            self.fighter_cam.setPos(0, 0, 400)
            self.fighter_cam.setHpr(0, -90, 0)

    def set_action(self, key, pre_value, value):
        if self.action[key] != pre_value:
            self.action[key] = value

    def set_accept(self):
        self.accept('z', self.track, [-1])
        self.accept('x', self.track, [-2])
        self.accept('c', self.track, [-3])
        self.accept('1', self.track, [0])
        self.accept('2', self.track, [1])
        self.accept('w', self.set_action, [0, value_add, value_add])
        self.accept('s', self.set_action, [0, value_sub, value_sub])
        self.accept('w-up', self.set_action, [0, value_sub, value_empty])
        self.accept('s-up', self.set_action, [0, value_add, value_empty])
        self.accept('arrow_up', self.set_action, [1, value_sub, value_sub])
        self.accept('arrow_down', self.set_action, [1, value_add, value_add])
        self.accept('arrow_up-up', self.set_action, [1, value_add, value_empty])
        self.accept('arrow_down-up', self.set_action, [1, value_sub, value_empty])
        self.accept('arrow_left', self.set_action, [2, value_sub, value_sub])
        self.accept('arrow_right', self.set_action, [2, value_add, value_add])
        self.accept('arrow_left-up', self.set_action, [2, value_add, value_empty])
        self.accept('arrow_right-up', self.set_action, [2, value_sub, value_empty])
        self.accept('a', self.set_action, [3, value_sub, value_sub])
        self.accept('d', self.set_action, [3, value_add, value_add])
        self.accept('a-up', self.set_action, [3, value_add, value_empty])
        self.accept('d-up', self.set_action, [3, value_sub, value_empty])

    def set_text(self):
        # pass
        return OnscreenText(
            pos=(-1.2, 0.8),
            fg=(0.95, 0, 0, 1),
            align=TextNode.ALeft,
            shadow=(0.5, 0.5, 0.5, 0.5),
            scale=0.05
        ), OnscreenText(
            pos=(1.2, 0.8),
            fg=(0, 0, 0.95, 1),
            align=TextNode.ARight,
            shadow=(0.5, 0.5, 0.5, 0.5),
            scale=0.05
        )

    def set_light(self):
        self.setBackgroundColor(0, 0.6, 0.8)
        ambient_light = AmbientLight('ambientLight')
        ambient_light.setColor((0.0, 0.0, 0.0, 0.5))
        ambient_light_np = self.render.attachNewNode(ambient_light)
        self.render.set_light(ambient_light_np)

        directional_light = DirectionalLight('directionalLight')
        directional_light.setColor((1.0, 1.0, 1.0, 0.5))
        directional_light_np = self.render.attachNewNode(directional_light)
        directional_light_np.setPos(0, 0, 10000)
        directional_light_np.setHpr(0, -90, 0)
        self.render.set_light(directional_light_np)

    def load_fighters(self, red_num, fighter_num):
        if self.red_num == red_num and self.fighter_num == fighter_num:
            for trail in self.trails:
                trail.reset()
            return
        for trail in self.trails:
            trail.removeNode()
        del self.trails[:]
        #self.trails.clear()
        for fighter in self.fighters:
            fighter.removeNode()
        del self.fighters[:]
        self.red_num = red_num
        self.fighter_num = fighter_num
        for i in range(fighter_num):
            fighter = self.loader.loadModel("bin/F-16")
            fighter.setScale(0.1, 0.1, 0.1)
            if i < red_num:
                fighter.setColor((0.8, 0.0, 0.0, 0.1))
            else:
                fighter.setColor((0.0, 0.0, 0.8, 0.1))

            fighter.reparentTo(self.render)
            fighter.hide()
            self.fighters.append(fighter)
            self.trails.append(Trail(self.render, fighter, 100,10))

    def step(self, task):
        state_dict = self.thread.state_dict
        if state_dict is not None and state_dict['current_step'] != self.cs:
            if self.cs < 0 or self.cs > state_dict['current_step']:
                fighter_num = len(state_dict['fighters'])
                self.load_fighters(state_dict['red_num'], fighter_num)
            self.cs = state_dict['current_step']

            for i, fighter_dict in enumerate(state_dict['fighters']):
                fighter = self.fighters[i]
                x, y, z = fighter_dict['y'] / 100, fighter_dict['x'] / 100, fighter_dict['z'] / 100
                h, p, r = -fighter_dict['yaw'], fighter_dict['pitch'], fighter_dict['roll']
                fighter.setPosHpr(x, y, z, h, p, r)
                if fighter_dict['dead'] == 0:
                    self.trails[i].step()
                    fighter.show()
                else:
                    fighter.hide()
            if self.focus == -2:
                self.track(-2)
            elif self.focus == -3:
                self.track(-3)

        if state_dict is not None:
            for i in range(2):
                fighter_dict = state_dict['fighters'][i]
                x, y, z, v = fighter_dict['x'], fighter_dict['y'], fighter_dict['z'], fighter_dict['v']
                pitch, yaw, roll = fighter_dict['pitch'], fighter_dict['yaw'], fighter_dict['roll']
                reward = fighter_dict['reward']
                angle1 = ''
                for angle_dict in fighter_dict['angle1']:
                    angle1 += '{:<3d} {:<4.2f}\n'.format(angle_dict['id'], angle_dict['angle'])
                self.state_text[i].setText(
                    'time: {:.1f}\nid:{}\nx: {:.2f}\ny: {:.2f}\nz: {:.2f}\nv: {:.2f}\npitch: {:.2f}\nyaw: {:.2f}\nroll: {:.2f}\nreward: {}\nangle :\n{}'.format(
                        state_dict['current_step'] * state_dict['delay'] * state_dict['dt'],
                        i,
                        x, y, z, v,
                        pitch, yaw, roll,
                        reward, angle1
                    ))

        self.camera.setPos(self.fighter_cam.getPos(self.render))
        self.camera.setHpr(self.fighter_cam.getHpr(self.render))
        return Task.cont
