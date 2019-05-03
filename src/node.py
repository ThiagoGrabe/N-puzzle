class Node:
    """
    This class represents the nodes that defines each search algorithm.
    """

    def __init__(self, state, parent, movement, depth, cost, heuristic):
        # super(Node, self).__init__()

        self.state = state
        self.parent = parent
        self.move = movement
        self.depth = depth
        self.cost = cost
        self.heuristic = heuristic
        # self.estimated_cost = self.cost + self.heuristic

        if self.state:
            self.map = ''.join(str(pseudo_state) for pseudo_state in self.state)

    def __eq__(self, other):
        return self.map == other.map

    def __lt__(self, other):
        return self.map < other.map
