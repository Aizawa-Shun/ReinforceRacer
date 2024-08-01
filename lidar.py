import pybullet as p
import math
import numpy as np

numRays = 10
rayFrom = []
rayTo = []
rayIds = []
rayLen = 2.90
rayStartLen = 0.10
rayColor = [0,1,0]
rayHitColor = [1,0,0]

def set(model, lidar):
    for i in range(numRays):
        rayX = math.sin(-0.25*math.pi+0.75*2.0*math.pi*float(i)/numRays)
        rayY = math.cos(-0.25*math.pi+0.75*2.0*math.pi*float(i)/numRays)
        rayFrom.append([rayStartLen*rayX, rayStartLen*rayY, 0])
        rayTo.append([rayLen*rayX, rayLen*rayY, 0])
        rayIds.append(p.addUserDebugLine(rayFrom[i], rayTo[i], rayColor, parentObjectUniqueId = model, parentLinkIndex = lidar))

def detection(model, lidar):
    distances = np.zeros(numRays)
    for i in range(numRays):
        results = p.rayTestBatch(rayFrom, rayTo, parentObjectUniqueId = model, parentLinkIndex = lidar)
        hitDetection = results[i][2]

        if (hitDetection==1.0):
            p.addUserDebugLine(rayFrom[i], rayTo[i], rayColor, replaceItemUniqueId = rayIds[i], parentObjectUniqueId = model, parentLinkIndex = lidar)
            distances[i] = rayStartLen + rayLen
        else:
            xHit = rayFrom[i][0] + hitDetection * (rayTo[i][0] - rayFrom[i][0])
            yHit = rayFrom[i][1] + hitDetection * (rayTo[i][1] - rayFrom[i][1])
            zHit = rayFrom[i][2] + hitDetection * (rayTo[i][2] - rayFrom[i][2])
            localHitTo = [xHit, yHit, zHit]
            p.addUserDebugLine(rayFrom[i], localHitTo, rayHitColor, replaceItemUniqueId=rayIds[i], parentObjectUniqueId = model, parentLinkIndex = lidar)
            distances[i] = math.sqrt(xHit**2 + yHit**2)
    return distances