from dataclasses import dataclass, field
import numpy as np
from scipy.sparse import csr_matrix
from collections import defaultdict
from collections import deque
import symbols
from scipy.sparse.csgraph import connected_components

@dataclass
class HexboardFeatures:
    hex_game_data: np.ndarray[np.ndarray[int]]
    winner: int
    adjacency_matrix: np.ndarray
  
    # THIS IS IMPORTANT! Set the number of features returned by 'get_feature_vector()'
    # The final count below matches the number of features we will return:
    # 1. Player Steps Completed
    # 2. Distance to Goal
    # 3. Cluster Size
    # 4. Node Degree
    # 5. Average Degree in Cluster
    # 6. Edge Connectivity
    # 7. Path Robustness 
    # 8. Threat Proximity 
    # 9. Critical Gaps 
    # 10. Center Control 

   
    FEATURE_COUNT = 10  
    
    def get_feature_symbols(self, node_id: int) -> list[str]:
        """Generate a list of feature symbols for a given node."""                
        feature_symbols = []      
        
        feature_symbols += self._get_steps_completed_symbols(node_id)  
        # get only edges with the same player
        neighbour_symbols = self._get_neighbour_symbol(node_id)

        edges_symbols = self._get_edge_symbols(node_id)

        # return all feature symbols
        return feature_symbols \
            + neighbour_symbols \
            + edges_symbols
    
    def _get_steps_completed_symbols(self, node_id: int) -> list[str]:
        shape = self.hex_game_data.shape[0]
        
        player_occupation = self.hex_game_data.flatten()[node_id]        
                
        if player_occupation == 0:
            return ""
        
        steps_completed = self._calculate_steps_completed(node_id)
        player_sym = symbols.PLAYER_SYMBOLS[player_occupation]
        step_symbols = []
        for i in range(symbols.STEPS_COMPLETED_THRESHOLD, steps_completed):
            step_symbols.append(f'{player_sym}{player_sym:{player_sym}<{i}}')
     

        return step_symbols
    
    def _get_neighbour_symbol(self, node_id: int) -> list[str]:
        shape = self.hex_game_data.shape[0]
        row_nr = node_id // shape
        col_nr = node_id % shape
        player_occupation = self.hex_game_data.flatten()[node_id]  
        neigbour_symbols = []
        for row, col in self._get_valid_neighbour_positions(node_id):
            p = self.hex_game_data[row_nr + row, col_nr + col]        

            syms = symbols.DIRECTIONS[(row,col)]                        
            for sym in syms:
                psym = symbols.PLAYER_SYMBOLS[p]
                neigbour_symbols.append(f'{psym}{sym}')

        return neigbour_symbols

    def _get_edge_symbols(self, node_id: int) -> list[str]:
        edge_symbols = []
        shape = self.hex_game_data.shape[0]
        row_nr = node_id // shape
        col_nr = node_id % shape
        
        player_occupation = self.hex_game_data.flatten()[node_id]  
        
        if row_nr == 0 or row_nr == shape-1 \
            or col_nr == 0 or col_nr == shape-1:
            if (self._calculate_steps_completed(node_id) == shape):
                edge_symbols.append(symbols.EDGE_SYMBOLS[player_occupation])

        return edge_symbols
            
    def _get_valid_neighbour_positions(self, node_id: int) -> list[tuple[int, int]]:
        valid_neighbour_positions = []
        shape = self.hex_game_data.shape[0]
        row_nr = node_id // shape
        col_nr = node_id % shape
        for row, col in symbols.HEXBOARD_VALID_DIRECTION:            
            if (row < 0 and row_nr == 0) or (row > 0 and row_nr == shape-1) \
                or (col < 0 and col_nr == 0) or (col > 0 and col_nr == shape-1):
                continue

            valid_neighbour_positions.append((row, col))

        return valid_neighbour_positions

    def get_blocked_nodes(self) -> list[int]:
        blocked_nodes = []
        shape = self.hex_game_data.shape[0]
        for node_id in range(len(self.hex_game_data.flatten())):
            player_occupation = self.hex_game_data.flatten()[node_id]
            step_count = self._calculate_steps_completed(node_id)
            if step_count == shape:
                for row, col in self._get_valid_neighbour_positions(node_id):
                    p = self.hex_game_data[row, col]
                    if p == player_occupation:
                        continue # skip if not the same player

                    current_row = node_id // shape
                    current_col = node_id % shape
                    new_row = current_row + row
                    new_col = current_col + col

                    # add to blocked nodes
                    # get node_id from row and col
                    id = new_row * shape + new_col
                    blocked_nodes.append(id)

        return blocked_nodes

    # def get_bridged_nodes(self) -> list[int]:
    #     bridged_nodes = []
    #     shape = self.hex_game_data.shape[0]
    #     for node_id in range(len(self.hex_game_data.flatten())):
    #         player_occupation = self.hex_game_data.flatten()[node_id]

    def _calculate_steps_completed(self, node_id: int) -> int:
        """
        Calculate the progress towards the win condition as a fraction of the board size.
        Player 1 aims to connect vertically, while Player 2 aims to connect horizontally.
        """
        player_occupation = self.hex_game_data.flatten()[node_id]

        # If the node is unoccupied, return 0
        if player_occupation == 0:
            return 0

        # Find the cluster of connected nodes for the player
        cluster = self._find_cluster(node_id, player_occupation)
        board_size = self.hex_game_data.shape[0]

        # Calculate steps based on win condition direction
        if player_occupation == 1:
            # Player 1 aims to connect vertically, so we look at unique rows spanned
            rows_spanned = {pos // board_size for pos in cluster}  # Unique row indices
            steps_completed = len(rows_spanned) #/ board_size  # Fraction of board height covered
        elif player_occupation == -1:
            # Player 2 aims to connect horizontally, so we look at unique columns spanned
            cols_spanned = {pos % board_size for pos in cluster}  # Unique column indices
            steps_completed = len(cols_spanned) #/ board_size  # Fraction of board width covered
        

        #print(f'Node {node_id} steps completed: {steps_completed}')
        return steps_completed
   
    def _find_cluster(self, node_id: int, player: int) -> set[int]:
        """Find the connected component (cluster) for the given node using DFS."""
        visited = set()
        stack = [node_id]
        board_flatten = self.hex_game_data.flatten()

        while stack:
            current_node = stack.pop()
            if current_node not in visited and board_flatten[current_node] == player:
                visited.add(current_node)
                neighbors = self.get_possible_neighbors(current_node)
                for neighbor in neighbors:
                    if board_flatten[neighbor] == player and neighbor not in visited:
                        stack.append(neighbor)

        return visited

    def get_possible_neighbors(self, node_id: int) -> list[int]:
        """Return the neighboring node IDs for the given node in the hexagonal grid."""
        neighbors = []
        board_size = self.hex_game_data.shape[0]
        row = node_id // board_size
        col = node_id % board_size
        neighbor_directions = symbols.HEXBOARD_VALID_DIRECTION# 

        for d_row, d_col in neighbor_directions:
            n_row, n_col = row + d_row, col + d_col
            if 0 <= n_row < board_size and 0 <= n_col < board_size:
                neighbor_id = n_row * board_size + n_col
                neighbors.append(neighbor_id)

        return neighbors
    
    
    def _get_internal_graph_for_node(self, node_id: int, player: int) -> dict:
        """
        Get the graph structure of the connected component (cluster) that the node belongs to.
        This returns a dictionary where keys are nodes and values are lists of neighboring nodes.
        """
        cluster_nodes = self._find_cluster(node_id, player)
        internal_graph = defaultdict(list)

        for node in cluster_nodes:
            neighbors = self.get_possible_neighbors(node)
            for neighbor in neighbors:
                if neighbor in cluster_nodes:
                    internal_graph[node].append(neighbor)

        return internal_graph

    def _get_cluster_size(self, node_id: int, player: int) -> int:
        """Calculate the size of the cluster for the given node."""
        return len(self._find_cluster(node_id, player))

    def _get_degree_of_node(self, node_id: int, internal_graph: dict) -> int:
        """Return the degree of the node (number of neighbors in the internal graph)."""
        return len(internal_graph.get(node_id, []))

    def _get_average_degree(self, internal_graph: dict) -> float:
        """Calculate the average degree of the nodes in the internal graph."""
        total_degree = sum(len(neighbors) for neighbors in internal_graph.values())
        num_nodes = len(internal_graph)
        return total_degree / num_nodes if num_nodes > 0 else 0
    
    def _get_player_connectivity(self, player: int) -> float:
        """
        Measure the connectivity of the player's nodes as the ratio of the largest cluster
        size to the total number of player-controlled nodes.
        """
        clusters = self._get_player_clusters(player)
        
        if not clusters:
            return 0.0

        # Get the size of the largest cluster
        largest_cluster_size = max(len(cluster) for cluster in clusters)

        # Get the total number of nodes controlled by the player
        total_player_nodes = np.sum(self.hex_game_data == player)

        # Return the ratio of the largest cluster to the total player nodes
        return largest_cluster_size / total_player_nodes if total_player_nodes > 0 else 0.0
    
    def get_edge_connectivity(self, player: int) -> int:
        """
        Counts how many nodes are occupied by the specified player and connected to the edges of the board.
        """
        edge_nodes_count = 0  # Initialize edge count
        board_size = self.hex_game_data.shape[0]  # Size of one dimension of the hex board
        flattened_board = self.hex_game_data.flatten()  # Flatten for easy node ID indexing

        # Iterate over all nodes by their ID to check for player occupation and edge connectivity
        for node_id in range(len(flattened_board)):
            player_occupation = flattened_board[node_id]  # Determine the occupation of this node
            
            # Check if the node is occupied by the specified player
            if player_occupation == player:
                row = node_id // board_size  # Calculate row index
                col = node_id % board_size   # Calculate column index
                
                # Check if this node is on any edge of the board
                if row == 0 or row == board_size - 1 or col == 0 or col == board_size - 1:
                    edge_nodes_count += 1

        return edge_nodes_count
    
   
    def get_blocked_nodes(self) -> list[int]:
        blocked_nodes = []
        shape = self.hex_game_data.shape[0]
        for node_id in range(len(self.hex_game_data.flatten())):
            player_occupation = self.hex_game_data.flatten()[node_id]
            step_count = self._calculate_steps_completed(node_id)
            if step_count == shape:
                for row, col in self._get_valid_neighbour_positions(node_id):
                    p = self.hex_game_data[(node_id // shape) + row, (node_id % shape) + col]
                    if p == player_occupation:
                        continue
                    blocked_nodes.append((node_id // shape + row) * shape + (node_id % shape + col))
        return blocked_nodes
    

    def _get_path_robustness(self, player: int) -> int:
        """Measures the number of distinct connected components that connect a player's start and end edges."""
        if player not in [1, -1]:
            return 0

        size = self.hex_game_data.shape[0]
        flattened = self.hex_game_data.flatten()

        # Gather all nodes controlled by the player
        player_positions = [node_id for node_id, val in enumerate(flattened) if val == player]

        # If no nodes for this player, no paths
        if not player_positions:
            return 0

        # Build adjacency for the player's subgraph
        num_nodes = len(player_positions)
        adjacency_matrix = np.zeros((num_nodes, num_nodes), dtype=int)

        # Create a reverse map to quickly go from node_id to local index
        node_id_to_local_idx = {pid: i for i, pid in enumerate(player_positions)}

        for pid in player_positions:
            r, c = divmod(pid, size)
            local_idx = node_id_to_local_idx[pid]
            neighbors = self.get_possible_neighbors(pid)
            for n in neighbors:
                if flattened[n] == player:
                    neighbor_idx = node_id_to_local_idx[n]
                    adjacency_matrix[local_idx, neighbor_idx] = 1
                    adjacency_matrix[neighbor_idx, local_idx] = 1

        # Convert to a sparse graph and find connected components
        graph = csr_matrix(adjacency_matrix)
        _, labels = connected_components(csgraph=graph, directed=False)

        # Determine which nodes belong to the start and end edges
        if player == 1:
            # Player 1 aims to connect top (row=0) to bottom (row=size-1)
            start_ids = [n for n in player_positions if n // size == 0]
            end_ids = [n for n in player_positions if n // size == (size - 1)]
        else:
            # Player -1 aims to connect left (col=0) to right (col=size-1)
            start_ids = [n for n in player_positions if n % size == 0]
            end_ids = [n for n in player_positions if n % size == (size - 1)]

        # Get component labels for start and end nodes
        start_labels = {labels[node_id_to_local_idx[n]] for n in start_ids}
        end_labels = {labels[node_id_to_local_idx[n]] for n in end_ids}

        # The path robustness is how many connected components span from start to end
        shared_components = len(start_labels & end_labels)

        return shared_components


    
    def _get_threat_proximity(self, player: int) -> int:
        """Measures potential threats by counting cells that would complete a connection with a single move."""
        size = self.hex_game_data.shape[0]
        threat_proximity = 0
        visited_empty_cells = set()

        # Gather all positions occupied by the player
        flattened = self.hex_game_data.flatten()
        player_positions = [node_id for node_id, val in enumerate(flattened) if val == player]

        # Convert flattened indices to (row, col)
        player_nodes = [(pos_id // size, pos_id % size) for pos_id in player_positions]

        # Directions to check around a player's piece
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, 1), (1, -1)]

        for (row, col) in player_nodes:
            # Check adjacent cells for potential threat positions
            for d_row, d_col in directions:
                n_row, n_col = row + d_row, col + d_col

                # Ensure the cell is within bounds and empty
                if 0 <= n_row < size and 0 <= n_col < size and self.hex_game_data[n_row, n_col] == 0:
                    # Check if the empty cell has not been visited before
                    if (n_row, n_col) not in visited_empty_cells:
                        visited_empty_cells.add((n_row, n_col))

                        # Count connections to the player's pieces around this empty cell
                        connections = 0
                        for dd_row, dd_col in directions:
                            nn_row, nn_col = n_row + dd_row, n_col + dd_col
                            if 0 <= nn_row < size and 0 <= nn_col < size and self.hex_game_data[nn_row, nn_col] == player:
                                connections += 1

                        # If placing a piece here would connect to 2 or more player pieces, it's a threat
                        if connections >= 2:
                            threat_proximity += 1

        return threat_proximity

    
    def _get_center_control(self, player: int) -> int:
        """Counts the number of player nodes in the central region of the board."""
        size = self.hex_game_data.shape[0]

        # Define the central region
        center_start = size // 4
        center_end = size - center_start

        center_control = 0

        # Flatten the board and find positions occupied by the player
        flattened = self.hex_game_data.flatten()
        player_positions = [node_id for node_id, val in enumerate(flattened) if val == player]

        # Check how many of these positions fall within the central region
        for pid in player_positions:
            row, col = divmod(pid, size)
            if center_start <= row < center_end and center_start <= col < center_end:
                center_control += 1

        return center_control

    
    def _get_critical_gaps(self, player: int) -> int:
        """Counts the number of critical positions where a single move could connect a player's pieces."""
        size = self.hex_game_data.shape[0]
        critical_gaps = 0

        for row in range(size):
            for col in range(size):
                if self.hex_game_data[row, col] == 0:
                    # Check if placing a piece here would connect two or more clusters of the player's pieces
                    adjacent_same_player = 0
                    for d_row, d_col in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, 1), (1, -1)]:
                        n_row, n_col = row + d_row, col + d_col
                        if 0 <= n_row < size and 0 <= n_col < size:
                            if self.hex_game_data[n_row, n_col] == player:
                                adjacent_same_player += 1

                    # Increment critical gaps if this cell could connect at least two clusters of the player's pieces
                    if adjacent_same_player >= 2:
                        critical_gaps += 1

        return critical_gaps

   


    

    def _check_if_threat_cell(self, node_id: int, player: int) -> int:
        """Return 1 if placing a piece of 'player' at node_id would create a threat, else 0."""
        if player not in [1, -1]:
            return 0

        flattened = self.hex_game_data.flatten()
        size = self.hex_game_data.shape[0]

        # If the cell is not empty, we cannot place a piece here
        if flattened[node_id] != 0:
            return 0

        # Directions representing hex neighbors
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, 1), (1, -1)]
        row, col = divmod(node_id, size)
        connections = 0

        # Count how many adjacent cells are occupied by the same player
        for d_row, d_col in directions:
            n_row, n_col = row + d_row, col + d_col
            if 0 <= n_row < size and 0 <= n_col < size and self.hex_game_data[n_row, n_col] == player:
                connections += 1

        # If placing a piece here would create at least two adjacent connections, it's a threatening position
        return 1 if connections >= 2 else 0
    

    def _get_features(self, node_id) -> np.ndarray:
        """Generate a feature vector for a given node."""
        # Player occupation (1, -1, 0)
        player_occupation = self.hex_game_data.flatten()[node_id]

        # Feature 1: Player steps completed
        player_steps_completed = self._calculate_steps_completed(node_id)

        # Feature 2: Distance to the goal line
        distance_to_goal = self._get_distance_to_goal_line(node_id, player_occupation)

        # Feature 3: Cluster size of connected components
        cluster_size = self._get_cluster_size(node_id, player_occupation)

        # Feature 4 & 5: Internal graph computations for degree and average degree
        internal_graph = self._get_internal_graph_for_node(node_id, player_occupation) if player_occupation != 0 else {}
        # Feature 4: Degree of the node (number of connections in the cluster)
        node_degree = self._get_degree_of_node(node_id, internal_graph) if player_occupation != 0 else 0
        # Feature 5: Average degree in the internal graph (connectivity within the cluster)
        avg_degree = self._get_average_degree(internal_graph) if player_occupation != 0 else 0

        # Feature 6: Edge connectivity - number of nodes of this player connected to edges
        edge_connectivity = self.get_edge_connectivity(player_occupation) if player_occupation != 0 else 0

        # Feature 7: Path robustness - number of unique paths from start to end edges 
        path_robustness = self._get_path_robustness(player_occupation)

        # Feature 8: Threat proximity - cells that create a connection with one move 
        threat_proximity = self._get_threat_proximity(player_occupation)
        #or 
        #threat_proximity = self._check_if_threat_cell(node_id, player_occupation)

        # Feature 9: Critical gaps - number of cells that connect clusters with one move 
        critical_gaps = self._get_critical_gaps(player_occupation)

        # Feature 10: Center control - number of player nodes in the central region 
        center_control = self._get_center_control(player_occupation)

        # Combine all features into a single array
        features = np.array([
            player_steps_completed,
            distance_to_goal,
            cluster_size,
            node_degree,
            avg_degree,
            edge_connectivity,
            path_robustness,
            threat_proximity,
            critical_gaps,
            center_control
        ], dtype=float)
        return features

        
    def get_feature_vector(self, node_id) -> csr_matrix:
        #print(f"Feature vector (raw): {self._get_features(node_id)}")
        return csr_matrix(self._get_features(node_id))
        #return csr_matrix(HexboardFeatures.z_score_normalize(self._get_features(node_id)))


    def _get_distance_to_goal_line(self, node_id: int, player: int) -> float:
        board_size = self.hex_game_data.shape[0] 
        row, col = divmod(node_id, board_size)
        
        if player == 1:  # Player 1 aims to connect vertically
            distance_to_top = row / board_size
            distance_to_bottom = (board_size - 1 - row) / board_size
            return min(distance_to_top, distance_to_bottom)
        elif player == -1:  # Player 2 aims to connect horizontally
            distance_to_left = col / board_size
            distance_to_right = (board_size - 1 - col) / board_size
            return min(distance_to_left, distance_to_right)
        return 1.0  # Default for unoccupied nodes





    @staticmethod
    def z_score_normalize(features: np.ndarray) -> np.ndarray:
        """
        Z-score normalization with a check to avoid division by zero when standard deviation is zero.
        """
        mean = np.mean(features)
        std = np.std(features)

        # If standard deviation is zero, return an array of zeros to avoid NaN
        if std == 0:
            return np.zeros_like(features)

        return (features - mean) / std






