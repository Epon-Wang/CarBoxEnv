import random
import time
import os
import mujoco
import glfw
import numpy as np
import cv2
from mujoco import MjModel, MjData, Renderer, mj_step





# env File Structure
#   /envMainFolder
#   -> /tb3Burger
#      -> /assets
#      -> envComponents.xml
#      -> turtlebot3_body.xml
#   -> /envSubFolder_1
#   -> /envSubFolder_2
#   -> /...
#   -> /envSubFolder_n





class CarBoxEnv:
    def __init__(self, xMax, yMax, nBox, sBox=0.05, xCar=0, yCar=0, mapPath=""):
        self.arena = (xMax, yMax)           # arena size = (2*xMax) * (2*yMax)
        self.carPos = (xCar, yCar)          # car init location
        self.boxPos = []                    # box positions dictionary
        self.boxNum = nBox                  # num of boxes in arena
        self.boxSize = sBox                 # box size
        self.envPath = mapPath              # ./path/to/envMainFolder
        self.wallColor = ["0.5 0.3 0.3 1", 
                          "0.3 0.5 0.3 1", 
                          "0.3 0.3 0.5 1", 
                          "0.6 0.6 0.6 1"]  # arena wall color "r g b a"
        self.CamTopDown = (3.5, 90)         # arena topDown camera (height, fovy)
    


    ### Test if (x,y) is conflicted with any box/car position
    def isPosConflict(self, x, y):
        xm, ym = self.arena
        hl = self.boxSize
        # True if exceeds boundary
        if abs(x) >= (xm-hl) or abs(y) >= (ym-hl):
            return True
        # True if conflicts with car
        elif abs(x - self.carPos[0]) < 0.3 and abs(y - self.carPos[1]) < 0.3:
            return True
        # True if conflicts with other boxes
        else:
            for pos in self.boxPos:
                if abs(x - pos[0]) < 0.1 and abs(y - pos[1]) < 0.1:
                    return True
            return False
    


    ### Generate box positions at random
    def genBoxPos(self):
        xm, ym = self.arena
        hl = self.boxSize

        xp = xm
        yp = ym
        
        # box position based on percentage
        while self.isPosConflict(xp, yp):
            xPrct = random.random()
            yPrct = random.random()
            xp = 2 * (xm-hl) * xPrct - (xm-hl)
            yp = 2 * (ym-hl) * yPrct - (ym-hl)
        
        return xp, yp
    


    ### Set arena wall color by index (1=N 2=S 3=E 4=W)
    def setArenaWallColor(self, rgb, wallIdx):
        colorStr = "{} {} {} 1".format(rgb[0], rgb[1], rgb[2], rgb[3])
        # directly convert rgb to string for the convenience of env generation
        self.wallColor[wallIdx] = colorStr
    


    ### Generate new envSubFolder
    def genEnv(self):
        t = time.localtime()
        envId = "_{}{}{}{}_{}{}{}".format(
            t.tm_zone, 
            t.tm_year, t.tm_mon, t.tm_mday, 
            t.tm_hour, t.tm_min, t.tm_sec
        )

        if not os.path.exists(self.envPath):
            os.mkdir(self.envPath)
        
        os.mkdir(self.envPath + "/env" + envId)

        # box position generate
        for i in range(self.boxNum):
            bx, by = self.genBoxPos()
            self.boxPos.append((bx, by))


        ## box map (.xml) generate

        # BoxMap headers & footers
        hdBox = "<mujoco>\n  <worldbody>\n"
        wall_hdBox = '''    <body name="arena">\n'''
        wall_ftBox = "    </body>\n"
        ftBox = "  </worldbody>\n</mujoco>"

        # Arena Settings
        wallLen0 = self.arena[0] + 0.05
        wallLen1 = self.arena[1] + 0.05
        wallSize0 = '''size="{} 0.05 0.25"'''.format(wallLen0)
        wallSize1 = '''size="0.05 {} 0.25"'''.format(wallLen1)
        wallPos1 = '''pos="0 {} 0.25"'''.format(wallLen1)
        wallPos2 = '''pos="0 {} 0.25"'''.format(-wallLen1)
        wallPos3 = '''pos="{} 0 0.25"'''.format(wallLen0)
        wallPos4 = '''pos="{} 0 0.25"'''.format(-wallLen0)
        wall1 = '''      <geom name="wall_1" type="box" {} {} rgba="{}"/>\n'''.format(wallPos1, wallSize0, self.wallColor[0])
        wall2 = '''      <geom name="wall_2" type="box" {} {} rgba="{}"/>\n'''.format(wallPos2, wallSize0, self.wallColor[1])
        wall3 = '''      <geom name="wall_3" type="box" {} {} rgba="{}"/>\n'''.format(wallPos3, wallSize1, self.wallColor[2])
        wall4 = '''      <geom name="wall_4" type="box" {} {} rgba="{}"/>\n'''.format(wallPos4, wallSize1, self.wallColor[3])

        newMap = "boxMap" + envId + ".xml"
        with open(self.envPath + "/env" + envId + "/" + newMap, "w") as f:
            f.write(hdBox)
            i = 0
            for e in self.boxPos:
                boxTitle = '''    <body name="box_{}" pos="{} {} 0">\n\t<freejoint/>\n'''.format(i, e[0], e[1])
                boxLight = '''\t<light name="box_top_light_{}" pos="0 0 2" mode="trackcom" diffuse=".4 .4 .4"/>\n'''.format(i)
                boxBody = '''\t<geom name="some_box_{0}" type="box" size="{1} {1} {1}" rgba="1 0 0 1" mass=".05" />\n'''.format(i, self.boxSize)
                boxFoot = "    </body>\n"
                f.write(boxTitle + boxLight + boxBody + boxFoot)
                i += 1
            f.write(wall_hdBox)
            f.write(wall1 + wall2 + wall3 + wall4)
            f.write(wall_ftBox)
            f.write(ftBox)


        ## env wrapper (.xml) generate

        # EnvMain headers & footers
        hdMain = "<mujoco>\n"
        mdMain = '''  <include file="../tb3Burger/envComponents.xml"/>\n\n  <worldbody>\n'''
        ftMain = '''      <include file="../tb3Burger/turtlebot3_body.xml"/>\n    </body>\n  </worldbody>\n</mujoco>'''

        # Env Settings
        boxMap = '''  <include file="{}"/>\n'''.format(newMap)
        litDef = '''    <light pos="0 0 1.5" dir="0 0 -1" directional="true"/>\n'''
        camDef = '''    <camera name="topDownGlobal" pos="0 0 {}" xyaxes="1 0 0 0 1 0" fovy="{}"/>\n'''.format(self.CamTopDown[0], self.CamTopDown[1])
        fldDef = '''    <geom name="floor" size="0 0 0.01" type="plane" material="groundplane"/>\n'''
        carDef = '''    <body name="theCar" pos="{} {} .00">\n'''.format(self.carPos[0], self.carPos[1])

        newMain = "/envMain" + envId + ".xml"
        with open(self.envPath + "/env" + envId + newMain, "w") as f:
            f.write(hdMain + boxMap + mdMain + litDef + camDef + fldDef + carDef + ftMain)





class EnvPlayground:
    def __init__(self, modelPath):
        self.path = modelPath           # ./path/to/envMain.xml
        self.velocity = 10              # Car WHEEL velocity in radians
        self.windowSize = (1280, 960)    # monitor window size, size up to 1920*1080
    
    ### Set car wheel velocity
    def setVelocity(self, vel):
        self.velocity = vel

    ### Set monitor window size
    def setWindow(self, width, height):
        self.windowSize = (width, height)
    
    ### Initiate Playground (view = "pilot" "topDownGlobal" "topDownTracking")
    def play(self, view = "pilot"):
        if not glfw.init():
            raise RuntimeError("Failed to initialize GLFW")

        window = glfw.create_window(320, 240, "Ctrl Input Window", None, None)
        glfw.make_context_current(window)

        model = MjModel.from_xml_path(self.path)
        data = MjData(model)

        renderer = Renderer(model, width=self.windowSize[0], height=self.windowSize[1])

        # Ctrl Parameters
        velocity = self.velocity  
        ctrl = np.zeros(2)

        print("Press ESC to exit")

        while not glfw.window_should_close(window):
            glfw.poll_events()

            # Keyboard ctrl callback
            if glfw.get_key(window, glfw.KEY_UP)    == glfw.PRESS:
                ctrl[:] = [velocity, velocity]
                # print("Forward")
            elif glfw.get_key(window, glfw.KEY_DOWN)  == glfw.PRESS:
                ctrl[:] = [-velocity, -velocity]
                # print("Backward")
            elif glfw.get_key(window, glfw.KEY_LEFT)  == glfw.PRESS:
                ctrl[:] = [-velocity, velocity]
                # print("Left")
            elif glfw.get_key(window, glfw.KEY_RIGHT) == glfw.PRESS:
                ctrl[:] = [velocity, -velocity]
                # print("Right")
            else:
                ctrl[:] = [0.0, 0.0]
                # print("Stop")

            data.ctrl[:] = ctrl
            mj_step(model, data)

            renderer.update_scene(data, camera=view)
            img = renderer.render()
            cv2.imshow("Arena View", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)

            # Press ESC to exit
            if glfw.get_key(window, glfw.KEY_ESCAPE) == glfw.PRESS:
                break

        cv2.destroyAllWindows()
        renderer.close()
        glfw.terminate()






# a = CarBoxEnv(1, 1, 6, xCar=0, yCar=0, mapPath="/home/epon/carBlockEnv") 
# a.genEnv()

a = EnvPlayground("/home/epon/carBlockEnv/env_EDT202577_15854/envMain_EDT202577_15854.xml")
a.play()


