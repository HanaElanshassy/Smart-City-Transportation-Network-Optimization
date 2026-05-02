import pandas as pd

# The Node class: Represents each neighborhood or facility
class Node:
    def __init__(self, node_id, name, pop, node_type, x, y):
        self.id = str(node_id)
        self.name = name
        self.population = pop
        self.type = node_type
        self.x = float(x)
        self.y = float(y)

    def __repr__(self):
        return f"[{self.id}] {self.name} ({self.type})"

# The Edge class: Represents the roads connecting the nodes
class Edge:
    def __init__(self, from_id, to_id, distance, capacity, condition, cost):
        self.from_id = str(from_id)
        self.to_id = str(to_id)
        self.distance = float(distance)
        self.capacity = float(capacity)
        self.condition = float(condition)
        self.cost = float(cost)
        
        # The dynamic weight that your AI prediction model will update later
        self.current_weight = self.distance 

    def __repr__(self):
        return f"{self.from_id} -> {self.to_id} (Dist: {self.distance}km)"

# The main Graph class: The overarching data structure
class CairoGraph:
    def __init__(self):
        self.nodes = {} 
        self.edges = [] 
        self.adjacency_list = {} 

    def load_data(self, nodes_file, edges_file):
        # 1. Parse and store Nodes
        nodes_df = pd.read_csv(nodes_file)
        for _, row in nodes_df.iterrows():
            node = Node(row['ID'], row['Name'], row['Population'], row['Type'], row['X'], row['Y'])
            self.nodes[str(row['ID'])] = node
            self.adjacency_list[str(row['ID'])] = [] # Initialize empty connections list

        # 2. Parse and store Edges
        edges_df = pd.read_csv(edges_file)
        for _, row in edges_df.iterrows():
            edge = Edge(row['FromID'], row['ToID'], row['Distance'], row['Capacity'], row['Condition'], row['Cost'])
            self.edges.append(edge)
            
            # Map the edge in the adjacency list (vital for Dijkstra and A*)
            if str(row['FromID']) in self.adjacency_list:
                self.adjacency_list[str(row['FromID'])].append(edge)

        print(f"✅ Setup Complete: Loaded {len(self.nodes)} Nodes and {len(self.edges)} Edges.")

# ==========================================
# Sound Check (Testing the logic)
# ==========================================
if __name__ == "__main__":
    graph = CairoGraph()
    graph.load_data('nodes.csv', 'edges.csv')
    
    # Let's test the connections pulling from Maadi (ID: 1)
    print("\nRoads connected to Maadi:")
    for edge in graph.adjacency_list['1']:
        print(edge)