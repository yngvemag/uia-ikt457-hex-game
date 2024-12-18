from GraphTsetlinMachine.graphs import Graphs
import symbols
from hexboard_features import HexboardFeatures


def position_to_edge_id(pos, board_size):
    return pos[0] * board_size + pos[1]

def init_graphs_from(init_graph: Graphs, hexboards) -> Graphs:
    graphs = Graphs(len(hexboards) , init_with=init_graph)
    graphs = _init_graphs(graphs, hexboards)
    return graphs

def init_graphs(hexboards, hypervector_size: int = 16, hypervector_bits: int = 2):
    """
    Initialize graphs and add node features and edges for each hex board.
    """
    #print(symbols.NEIGHBOR_SYMBOLS)
    sym_names = []
    if HexboardFeatures.FEATURE_COUNT > 0:
        sym_names = [f"Feature:{i}" for i in range(HexboardFeatures.FEATURE_COUNT)]
    # Initialize Graphs object with the total number of hexboards (i.e., graphs)

    #print(symbols.ALL_SYMBOLS(hexboards[0].size))
    graphs = Graphs(
        number_of_graphs=len(hexboards),
        symbols=symbols.ALL_SYMBOLS(hexboards[0].size) + sym_names,
        hypervector_size=hypervector_size,
        hypervector_bits=hypervector_bits,
        double_hashing=False
    )

    # Populate graphs using the helper function
    graphs = _init_graphs(graphs, hexboards)
    return graphs



def _init_graphs(graphs: Graphs,  hexboards):
    """
    Helper function to initialize graph nodes, edges, and properties for each hexboard.
    """
    board_size = hexboards[0].size
    edges = []
    for i in range(board_size):
        for j in range(1, board_size):
            edges.append(((i, j-1), (i, j)))    # Connect rows
            edges.append(((j-1, i), (j, i)))    # Connect columns
            if i < board_size - 1:
                edges.append(((i, j), (i+1, j-1)))  # Connect diagonals

    n_edges_list = [
        2 if i == 0 or i == board_size**2-1 else
        3 if i == board_size - 1 or i == board_size**2-board_size else
        4 if i // board_size == 0 or i // board_size == board_size - 1 or i % board_size == 0 or i % board_size == board_size-1 else
        6 for i in range(board_size**2)
    ]

    # Prepare nodes and configurations
    # Prepare nodes
    for graph_id in range(len(hexboards)):
        graphs.set_number_of_graph_nodes(
            graph_id=graph_id,
            number_of_graph_nodes=board_size**2,
        )
    graphs.prepare_node_configuration()

    # Prepare edges
    for graph_id in range(len(hexboards)):
        for k in range(board_size**2):
            graphs.add_graph_node(graph_id, k, n_edges_list[k])
    graphs.prepare_edge_configuration()


    # Assign edges for each graph
    # Create the graph
    for graph_id, hexboard in enumerate(hexboards):

        #Add nodes blocking
        for id in hexboard.get_blocked_nodes():
            graphs.add_graph_node_property(graph_id, id, symbols.BLOCKED_SYMBOL[0])

        # if a player is fully connected - add extra info to each node of that player
        connected_player = hexboard.get_edge_connected_player()
        if connected_player != 0:
            for id in range(board_size**2):
                if hexboard.get_player_occupation(id) == connected_player:
                    for s in symbols.EDGE_TO_EDGE_ADVANTAGE:
                        graphs.add_graph_node_property(graph_id, id, s)

        for node_id in range(board_size**2):
            sym = hexboard.get_symbol_player_mapping(node_id)
            graphs.add_graph_node_property(graph_id, node_id, sym)

            feature_symbols = hexboard.get_feature_symbols(node_id)
            for f in feature_symbols:
                if f == "":
                    continue
                graphs.add_graph_node_property(graph_id, node_id, f)

            if HexboardFeatures.FEATURE_COUNT > 0:
                node_features = hexboard.get_feature_vector(node_id)
                if node_features.nnz > 0:  # Check if the node has any non-zero features
                    feature_indices = node_features.indices  # Get the non-zero feature indices
                    for feature in feature_indices:
                        graphs.add_graph_node_property(graph_id, node_id, f"Feature:{feature}")

        # Add edges
        for edge in edges:
            node_id = position_to_edge_id(edge[0], board_size)
            destination_node_id = position_to_edge_id(edge[1], board_size)
            graphs.add_graph_node_edge(graph_id, node_id, destination_node_id, edge_type_name=0)
            graphs.add_graph_node_edge(graph_id, destination_node_id, node_id, edge_type_name=0)



    graphs.encode()
    return graphs
