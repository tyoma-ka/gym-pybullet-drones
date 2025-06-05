from gym_pybullet_drones.envs.HoverAviary import HoverAviary
from gym_pybullet_drones.envs.PositionFlyToAviary import PositionFlyToAviary
from gym_pybullet_drones.envs.PositionObstacleFlyToAviary import PositionObstacleFlyToAviary
from gym_pybullet_drones.envs.OrientationFlyToAviary import OrientationFlyToAviary
from gym_pybullet_drones.envs.OrientationObstacleFlyToAviary import OrientationObstacleFlyToAviary

aviary_map = {
    "HoverAviary": HoverAviary,
    "HV": HoverAviary,
    "PositionFlyToAviary": PositionFlyToAviary,
    "PF": PositionFlyToAviary,
    "PositionObstacleFlyToAviary": PositionObstacleFlyToAviary,
    "POF": PositionObstacleFlyToAviary,
    "OrientationFlyToAviary": OrientationFlyToAviary,
    "OF": OrientationFlyToAviary,
    "OrientationObstacleFlyToAviary": OrientationObstacleFlyToAviary,
    "OOF": OrientationObstacleFlyToAviary
}

aviary_names = {
    PositionFlyToAviary: "Distance-based",
    PositionObstacleFlyToAviary: "Distance-based",
    OrientationFlyToAviary: "Orientation-based",
    OrientationObstacleFlyToAviary: "Orientation-based"
}