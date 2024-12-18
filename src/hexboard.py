from dataclasses import dataclass, field
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import csr_matrix
from collections import deque
from hexboard_features import HexboardFeatures
import symbols

@dataclass
class HexBoard:
    """
    A class to represent a hexagonal game board.
    Input: 
        raw_board_input: str containing the board layout as comma-separated values (last field is the winner)
        colors: tuple[str, str, str] = field(default_factory=lambda: ('red', 'lightblue', 'green'))
            colors[0] = player 1 color
            colors[1] = no input field
            colors[2] = player 2 color
    """
    raw_board_input: str    
    colors: tuple[str, str, str] = field(default_factory=lambda: ('skyblue', 'dimgray', 'orange'))


    # board attributes
    size: int = field(init=False)
    number_of_nodes: int = field(init=False)

    # board data
    raw_data: np.ndarray = field(init=False)
    winner: int = field(init=False)
    hex_game_data: np.ndarray[np.ndarray[int]] = field(init=False)
    hex_game_data_flatten: np.ndarray[int] = field(init=False)

    # graph data    
    _edges: list[tuple[tuple[int, int], tuple[int, int]]] = field(init=False, default_factory=list)  # List of edges

    # hexboard features
    hexboard_features: HexboardFeatures = field(init=False)

    # class variable
    # DOWN: (-1, 0) , LEFT: (0, -1),  RIGHT: (0, 1),  UP:(1, 0), UP-RIGHT(1, 1), DOWN-LEFT(-1, -1)
    HEXBOARD_VALID_DIRECTION = [(-1, 0),(0, -1), (0, 1), (1, 0), (1, 1), (-1, -1)]

    def __post_init__(self):
        length = self.raw_board_input.count(',')
        sizef = length**0.5
        
        if sizef % int(sizef) != 0:
            raise ValueError(f"Board size must be a square. {length} is not a square.")
        
        # set size
        self.size = int(sizef)
        self.number_of_nodes = self.size**2

        self.raw_data = np.array(self.raw_board_input.split(','), dtype=int)
        self.winner = self.raw_data[-1]
        self.raw_data = self.raw_data[:-1]


        # Initialize hex game data
        self.hex_game_data = [self.raw_data[i:i + self.size] for i in range(0, len(self.raw_data), self.size)]
        self.hex_game_data = np.array([np.array(row, dtype=int) for row in self.hex_game_data])
        self.hex_game_data_flatten = self.hex_game_data.flatten()

        # Initialize adjacency matrix        
        self._adjacency_matrix = np.zeros((self.size * self.size, self.size * self.size), dtype=int)
        self._create_hex_nodes()

        # initialize features
        self.hexboard_features = HexboardFeatures(self.hex_game_data, self.winner, self.adjacency_matrix)
           
    def _create_hex_nodes(self):
        """
        Create nodes for the size x size hexagonal board and connect cells with equal values (-1, 0, 1).
        Additionally, populate the player_nodes structure for each player, 
        update the adjacency matrix and the list of edges.

        Also updates the edges
        """
        for row in range(self.size):
            for col in range(self.size):
                player = self.hex_game_data[row][col]

                # Connect only nodes with equal values (same player)
                for d_row, d_col in symbols.HEXBOARD_VALID_DIRECTION:
                    n_row, n_col = row + d_row, col + d_col
                    if 0 <= n_row < self.size and 0 <= n_col < self.size:
                        neighbor_player = self.hex_game_data[n_row][n_col]

                        if player == neighbor_player and player != 0:
                            # Update adjacency matrix
                            node_index = row * self.size + col
                            neighbor_index = n_row * self.size + n_col
                            self._adjacency_matrix[node_index, neighbor_index] = 1
                            self._adjacency_matrix[neighbor_index, node_index] = 1

                            # Update edge list
                            self._edges.append(((row, col), (n_row, n_col)))

    def _hexagonal_layout(self):
        """
        Generate a hexagonal grid layout for visualization.
        The x-coordinate is staggered for a hexagonal appearance, and the y-coordinate is inverted for the traditional top-down layout.
        """
        pos = {}
        for row in range(self.size):
            for col in range(self.size):
                x = col + (row * 0.5)  # Stagger the x-coordinates
                y = -row  # Invert the y-coordinate to tilt the board
                pos[(row, col)] = (x, y)
        return pos

    @property
    def edges(self) -> list[tuple[tuple[int, int], tuple[int, int]]]:
        """Return the list of edges."""
        return self._edges

    def is_vertical_or_diagonal_up(self, node_id: int, neighbor_id: int) -> bool:
        """
        Check if the neighbor is in the vertical or diagonal direction that aids Player 1's vertical progression.
        """
        board_size = self.size
        node_row, node_col = divmod(node_id, board_size)
        neighbor_row, neighbor_col = divmod(neighbor_id, board_size)

        # Check if the neighbor is directly above or in a top diagonal position (for vertical progression)
        row_diff = neighbor_row - node_row
        col_diff = neighbor_col - node_col

        # Allow direct upward (row -1) and upward diagonals based on hexagonal layout
        return (row_diff == -1 and col_diff in {0, 1}) or (row_diff == 0 and col_diff == 1)

    def is_horizontal_or_diagonal_right(self, node_id: int, neighbor_id: int) -> bool:
        """
        Check if the neighbor is in the horizontal or diagonal direction that aids Player 2's horizontal progression.
        """
        board_size = self.size
        node_row, node_col = divmod(node_id, board_size)
        neighbor_row, neighbor_col = divmod(neighbor_id, board_size)

        # Check if the neighbor is directly to the right or in a right diagonal position (for horizontal progression)
        row_diff = neighbor_row - node_row
        col_diff = neighbor_col - node_col

        # Allow direct rightward (col +1) and rightward diagonals based on hexagonal layout
        return (col_diff == 1 and row_diff in {0, 1}) or (col_diff == 0 and row_diff == 1)

    @property
    def adjacency_matrix(self) -> np.ndarray:
        """Return the adjacency matrix."""
        return self._adjacency_matrix
    
    def get_possible_neighbors_adjancency(self, node_id: int) -> list[int]:
        """Get valid neighboring node IDs for a given node based on the adjacency matrix."""
        neighbors = []

        # Use the adjacency matrix to determine the neighbors
        for potential_neighbor_id, is_connected in enumerate(self.adjacency_matrix[node_id]):
            if is_connected:  # If there's a connection (1 in the adjacency matrix)
                neighbors.append(potential_neighbor_id)

        return neighbors


    def get_possible_neighbors(self, node_id: int) -> list[int]:
        """Get valid neighboring node IDs for a given node on the hexagonal grid."""
        neighbors = []
        board_size = self.size

        node_row = node_id // board_size
        node_col = node_id % board_size

        for row_offset, col_offset in symbols.DIRECTIONS:
            neighbor_row = node_row + row_offset
            neighbor_col = node_col + col_offset

            if 0 <= neighbor_row < board_size and 0 <= neighbor_col < board_size:
                neighbor_id = neighbor_row * board_size + neighbor_col
                neighbors.append(neighbor_id)

        return neighbors

    def get_labels(self):
        # Assuming self.winner holds the label for training (1 for player 1 win, -1 for player 2 win)
        if self.winner == 1:
            return 1
        return 0        
          
    def get_feature_vector(self, node_id: int, return_feature_names=False) -> tuple[csr_matrix, list[str]]:
        """Return the feature vector with named features."""
        return self.hexboard_features.get_feature_vector(node_id)

    @property
    def symbol_names(self) -> list[str]:
        """Return the list of symbols used in the game, including directional and node-specific symbols."""
        return symbols.PLAYER_SYMBOLS
    
    def get_feature_symbols(self, node_id: int) -> str:
        return self.hexboard_features.get_feature_symbols(node_id)
    
    def get_blocked_nodes(self) -> list[int]:
        return self.hexboard_features.get_blocked_nodes()
    
    def get_feature_vector(self, node_id: int) -> tuple[csr_matrix, list[str]]:
        """Return the feature vector with named features."""
        return self.hexboard_features.get_feature_vector(node_id)
    
    def get_symbol_player_mapping(self, node_id: int) -> str:
        """Map the player occupation for each node."""        
        player_occupation = self.hex_game_data_flatten[node_id] 
        return symbols.PLAYER_SYMBOLS[player_occupation]  
    
    def get_player_occupation(self, node_id: int) -> int:
        return self.hex_game_data_flatten[node_id]
    
    def get_edge_connected_player(self) -> int:
        for node_id in range(len(self.hex_game_data_flatten)):
            n = self.hexboard_features._calculate_steps_completed(node_id)
            if n == self.size:
                return self.hex_game_data_flatten[node_id]
            
        return 0
       
    
    @staticmethod
    def load_training_test_data(filename: str, 
                                testsize: float = 0.2,
                                has_header: bool = True,
                                randommize: bool = True) -> tuple['HexBoard', 'HexBoard']:
        """
        Load a list of HexBoards from a file, encode them using HexBoardFeatureEncoder,
        and split into training and test sets. Return HexBoardDataset for both train and test sets.
        """
        hexboards = []

        # Reading the file and creating HexBoard objects with labels
        with open(filename, 'r') as f:
            for idx, line in enumerate(f):
                if has_header and idx==0:
                    continue            
                hexboards.append(HexBoard(line))
            
        
        # Shuffle the dataset
        if randommize:
            np.random.shuffle(hexboards)
        
        # Split the data into training and test sets
        train_size = int(len(hexboards) * (1 - testsize))
        train_hexboards = hexboards[:train_size] 
        test_hexboards = hexboards[train_size:]

        return train_hexboards, test_hexboards

    def visualize_board(self, draw_connections: bool = True):
        """
        Visualize the hexagonal game board using matplotlib where nodes of the same value are connected.
        """        
        pos = self._hexagonal_layout()

        # Plot the nodes
        plt.figure(figsize=(8, 8))
        for (row, col), (x, y) in pos.items():
            color = self.colors[1] if self.hex_game_data[row][col] == 0 else (self.colors[0] if self.hex_game_data[row][col] == -1 else self.colors[2])
            plt.scatter(x, y, color=color, s=500)
            plt.text(x, y, f'{row},{col}', ha='center', va='center', fontsize=8)

        # # Draw connections between nodes based on hexagonal neighbors
        board_size = len(self.hex_game_data)

        if draw_connections:
            for row in range(board_size):
                for col in range(board_size):
                    node_value = self.hex_game_data[row][col]
                    if node_value != 0:  # Only connect non-empty nodes
                        neighbors = self.get_hex_neighbors(row, col)

                        for n_row, n_col in neighbors:
                            neighbor_value = self.hex_game_data[n_row][n_col]

                            # Only connect nodes of the same player
                            if neighbor_value == node_value:
                                x_values = [pos[(row, col)][0], pos[(n_row, n_col)][0]]
                                y_values = [pos[(row, col)][1], pos[(n_row, n_col)][1]]
                                plt.plot(x_values, y_values, color='black')

        plt.title("Hex Game Board")
        plt.show()

    def get_hex_neighbors(self, row, col):
        """
        Get the valid neighboring nodes of a given node in the hexagonal grid.
        """
        neighbors = []
        board_size = len(self.hex_game_data)

        # Define hexagonal neighbors (6 directions in a hex grid)
        neighbor_offsets = [
            (-1, 0),  # Top
            (1, 0),   # Bottom
            (0, -1),  # Left
            (0, 1),   # Right
            (-1, 1),  # Top-right
            (1, -1)   # Bottom-left
        ]

        for offset in neighbor_offsets:
            n_row = row + offset[0]
            n_col = col + offset[1]

            if 0 <= n_row < board_size and 0 <= n_col < board_size:
                neighbors.append((n_row, n_col))

        return neighbors
