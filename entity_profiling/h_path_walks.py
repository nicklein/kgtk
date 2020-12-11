import numpy as np


# Performs a random walk from each start node given
# Returns a 2d array where each row is a walk made up of vertex indexes
def random_walk(g, start_nodes, walk_length=10):
    # pre-allocate walks array with initial values of -1
    walks = np.ones((len(start_nodes),walk_length), dtype=int) * -1
    
    walks[:,0] = start_nodes
    cur_length = 1
    while cur_length < walk_length:
        cur_nodes = walks[:,cur_length - 1]
        # if we've previously hit a dead end, then we could have -1 as a 
        # current node value in this case, we want to continue filling in -1.
        neighbors = [np.array([-1]) if v < 0 else g.get_out_neighbors(v) for v in cur_nodes]
        # if there are no outbound edges we can take for 
        # some vertex, we'll make the next vertex we visit = -1
        neighbors = [np.array([-1]) if len(arr) < 1 else arr for arr in neighbors]
        next_nodes = [np.random.choice(arr) for arr in neighbors]
        walks[:,cur_length] = next_nodes
        cur_length += 1
    # trim -1's from any walks that reached a dead end
    walks = [arr[arr >= 0] for arr in walks]
    return walks

# Performs num_walks random walks at each node in the graph
# i.e. if there are 10 nodes in the gaph and num_walks=10, then we'll do 10*10 = 100 random walks.
# Returns a list of walks
def build_corpus(g, walk_length=10, num_walks=10):
    start_nodes = np.repeat(g.get_vertices(), num_walks)
    walks_by_index = random_walk(g, start_nodes)
    walks_by_qnode = [[g.vp.name[v_ix] for v_ix in walk] for walk in walks_by_index]
    return walks_by_qnode