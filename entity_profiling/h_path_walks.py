import numpy as np
from kgtk.gt.gt_load import load_graph_from_kgtk
from kgtk.io.kgtkreader import KgtkReader
import pathlib
from functools import lru_cache
from tqdm.auto import tqdm

# Performs num_walks random walks at each node in the graph
# i.e. if there are 10 nodes in the graph and num_walks=10, then we'll do 10*10 = 100 random walks.
# Appends walks to specified file, where each walk is a list of Q-node strings
def gt_random_walks_to_file(g, walks_file, walk_length=10, num_walks=10, batch_size=50000):
    vertices = g.get_vertices()
    num_batches = int(np.ceil(len(vertices) / batch_size))
    print("num_batches: {}".format(num_batches))
    for batch_num in tqdm(range(num_batches)):
        start_nodes = vertices[batch_num*batch_size : (batch_num+1)*batch_size]
        walks = gt_random_walks_from_nodes(g, start_nodes, walk_length, num_walks)
        # Explicitly cast list of lists to ndarray with dtype=object to avoid ragged nested sequences message
        walks = np.array(walks, dtype=object)
        with open(walks_file, "a") as f:
            np.savetxt(f, walks, fmt="%s")

# Performs num_walks random walks starting from each start node in the input array
# i.e. if there are 10 nodes in the start node array and num_walks=10, then we'll do 10*10 = 100 random walks.
# Returns a list of walks, where each walk is a list of Q-node strings
def gt_random_walks_from_nodes(g, start_nodes, walk_length, num_walks):
    start_nodes = np.repeat(start_nodes, num_walks)

    # pre-allocate walks array with initial values of -1
    walks = np.ones((len(start_nodes),walk_length), dtype=int) * -1
    
    walks[:,0] = start_nodes
    cur_length = 1
    while cur_length < walk_length:
        cur_nodes = walks[:,cur_length - 1]
        # Pad dead ends with -1s
        next_nodes = [-1 if v < 0 else np.random.choice(gt_get_out_neighbors(g,v)) for v in cur_nodes]
        walks[:,cur_length] = next_nodes
        cur_length += 1
    # trim -1's from any walks that reached a dead end
    walks = [arr[arr >= 0] for arr in walks]
    # change vertex indexes to Q-node names
    walks_by_qnode = [[g.vp.name[v_ix] for v_ix in walk] for walk in walks]
    return walks_by_qnode

@lru_cache(maxsize = 10000000)
def gt_get_out_neighbors(g, v):
    return g.get_out_neighbors(v)

def get_h_walks_from_kgtk_item_file(item_file, walks_file, directed=False, walk_length=10, num_walks=10, batch_size=50000):
    kr = KgtkReader.open(pathlib.Path(item_file))
    g = load_graph_from_kgtk(kr, directed=directed, hashed=True)
    gt_random_walks_to_file(g, walks_file, walk_length, num_walks, batch_size)

