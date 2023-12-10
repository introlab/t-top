# watchdog
This folder contains a watchdog node.

## Nodes
### `watchdog_node.py`
This node monitors a topic of a node and kill it if the node no longer publishes messages.

#### Parameters
 - `node_name` (string): The node name.
 - `timeout_duration_s` (double): The timeout value in secondes.

#### Subscribed Topics
 - `topic` (Any): The topic to monitor.
