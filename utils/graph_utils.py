class GraphNode:
    def __init__(self, id, name, cluster, children=[], parents=[]):
        self.id = id
        self.name = name
        self.cluster = set(cluster)
        self.children = set(children)
        self.parents = set(parents)

    def to_dict(self):
        return {'id': self.id, 
                'name': self.name, 
                'cluster': list(sorted(self.cluster)), 
                'children': list(sorted(self.children)), 
                'parents': list(sorted(self.parents))}

class ClusterNode:
    def __init__(self, type: str, arg, index=None):
        self.type = type
        self.index = index

        if isinstance(arg, int):
            self.num = arg
            self.cluster = None
        else:
            assert isinstance(arg, list)
            self.cluster = arg
            self.num = len(arg)
        
        self.parent = None
        self.child_num = 0
        self.children = []

    def remove_empty_children(self):
        self.children = [child for child in self.children if child.num > 0]
        self.child_num = len(self.children)
    
# Replace node1 with node2, traversing all its parents and children.
# Ensure correct transformation to node clusters.
def replace(node1: GraphNode, node2: GraphNode, top_node_indices: set[str], nodes_id_dict: dict[str, GraphNode]) -> None:
    node1_id, node2_id = node1.id, node2.id
    if node1_id in top_node_indices or node2_id in top_node_indices:
        top_node_indices.discard(node1_id)
        top_node_indices.add(node2_id)
    for parent_id in node1.parents:
        parent_node = nodes_id_dict[parent_id]
        parent_node.children.remove(node1_id)
        if parent_id != node2_id:
            parent_node.children.add(node2_id)
            node2.parents.add(parent_id)
    for child_id in node1.children:
        child_node = nodes_id_dict[child_id]
        child_node.parents.remove(node1_id)
        if child_id != node2_id:
            child_node.parents.add(node2_id)
            node2.children.add(child_id)
    del nodes_id_dict[node1_id]

# Replace node with its children, traversing all its parents and children.
def remove(node: GraphNode, top_node_indices: set[str], nodes_id_dict: dict[str, GraphNode]) -> None:
    node_id = node.id
    if node_id in top_node_indices:
        top_node_indices.discard(node_id)
    for child_id in node.children:
        nodes_id_dict[child_id].parents.remove(node_id)
    for parent_id in node.parents:
        parent_node = nodes_id_dict[parent_id]
        parent_node.children.remove(node_id)
        parent_node.cluster.update(node.cluster)
        for child_id in node.children:
            assert parent_id != child_id
            parent_node.children.add(child_id)
            nodes_id_dict[child_id].parents.add(parent_id)
    del nodes_id_dict[node_id]
