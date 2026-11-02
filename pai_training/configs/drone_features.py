features = {
    "observation.images.front": {
        "dtype": "video",
        "shape": (3, 480, 640),
        "names": ["channels", "height", "width"],
    },
    "observation.state": {
        "dtype": "float32",
        "shape": (6,),
        "names": ["x", "y", "z", "vx", "vy", "vz"],
    },
    "action": {
        "dtype": "float32",
        "shape": (4,),
        "names": ["vx", "vy", "vz", "wz"],
    },
}

robot_type = "drone"
repo_id = "custom_drone_flight"
task_name = "teleop"