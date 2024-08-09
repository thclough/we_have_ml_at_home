
class Node:

    _node_id_counter = 0

    def __init__(self, str_id=None):
        self.node_id = Node._node_id_counter

        self.str_id=str_id

        Node._node_id_counter += 1

    def __hash__(self):
        return hash(self.node_id)

    def __eq__(self, other):
        if isinstance(other, Node):
            return self.node_id == other.node_id
        return False

    def __repr__(self):
        if self.str_id is not None:
            return f"{self.str_id}"
        else:
            return f"{self.node_id} {type(self)}"