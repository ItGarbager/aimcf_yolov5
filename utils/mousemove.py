"""
from https://stackoverflow.com/questions/44467329/pyautogui-mouse-movement-with-bezier-curve
"""

import pyautogui
import bezier
import numpy as np
from time import time


# Disable pyautogui pauses (from DJV's answer)
pyautogui.MINIMUM_DURATION = 0
pyautogui.MINIMUM_SLEEP = 0
pyautogui.PAUSE = 0

# We'll wait 5 seconds to prepare the starting position
start_delay = 2
print("Drawing curve from mouse in {} seconds.".format(start_delay))
pyautogui.moveTo(100, 100)
pyautogui.sleep(start_delay)

# For this example we'll use four control points, including start and end coordinates
start = pyautogui.position()
print(start)
end = start[0]+600, start[1]+200
print(end)

begin_time = time()
# Two intermediate control points that may be adjusted to modify the curve.
control1 = start[0]+125, start[1]+100
control2 = start[0]+375, start[1]+50

# Format points to use with bezier
control_points = np.array([start, control1, control2, end])
points = np.array([control_points[:,0], control_points[:,1]]) # Split x and y coordinates

# You can set the degree of the curve here, should be less than # of control points
degree = 3
# Create the bezier curve
curve = bezier.Curve(points, degree)
# You can also create it with using Curve.from_nodes(), which sets degree to len(control_points)-1
# curve = bezier.Curve.from_nodes(points)

curve_steps = 50  # How many points the curve should be split into. Each is a separate pyautogui.moveTo() execution
delay = 1/curve_steps  # Time between movements. 1/curve_steps = 1 second for entire curve

listx = []
listy = []

# Move the mouse
for i in range(1, curve_steps+1):
    # The evaluate method takes a float from [0.0, 1.0] and returns the coordinates at that point in the curve
    # Another way of thinking about it is that i/steps gets the coordinates at (100*i/steps) percent into the curve
    x, y = curve.evaluate(i/curve_steps)
    listx.append(x[0])
    listy.append(y[0])
    # print(x, y)
    # pyautogui.moveTo(x, y)  # Move to point in curve
    # pyautogui.sleep(delay)  # Wait delay

nlistx = np.diff(np.array(listx))
nlisty = np.diff(np.array(listy))

print(nlistx)
print(nlisty)

print(nlistx.size)

for i in range(nlistx.size):
    pyautogui.moveRel(int(round(nlistx[i])), int(round(nlisty[i])))
    pyautogui.sleep(delay)

end_time = time() - begin_time
print(end_time)
