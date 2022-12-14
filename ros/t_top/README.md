# t_top

This ROS package contains the configuration files, the launch files and common code files.

## Common Code Descriptions

### `vector_to_angles` function

This function convert a direction vector to angles.

### `MovementCommands` class

This is an helper function to move the head and the torso.

## Nodes

### `robot_status.py`

This node send the robot status to Opentera-WebRTC.

#### Parameters

- `simulation` (bool): Indicates if it's used in the simulation.
