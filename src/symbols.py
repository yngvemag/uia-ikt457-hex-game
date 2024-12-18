from enum import Enum

class Direction(Enum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3

HEXBOARD_VALID_DIRECTION = [(-1, 0),(0, -1), (0, 1), (1, 0), (1, -1), (-1, 1)]

PLAYER_SYMBOLS = {
    1: "X", 
    -1: "O", 
    0: "_"  
}


# # [(-1, 0),(0, -1), (0, 1), (1, 0), (1, 1), (-1, -1)
# DIRECTION_SYMBOLS = ["UP","DOWN","LEFT","RIGHT"]
# DIRECTIONS = {
#      (-1, 0): [DIRECTION_SYMBOLS[0]],   # UP
#      (-1, 1): [DIRECTION_SYMBOLS[0], DIRECTION_SYMBOLS[3]],    # UP + RIGHT
#      (0, -1): [DIRECTION_SYMBOLS[2]], # LEFT   
#      (0, 1): [DIRECTION_SYMBOLS[3]], # RIGHT      
#      (1, -1): [DIRECTION_SYMBOLS[1], DIRECTION_SYMBOLS[2]],    # DOWN + LEFT
#      (1, 0): [DIRECTION_SYMBOLS[1]]   # DOWN   
#  }

# # hexagonal neighbors
DIRECTIONS = {
    (-1, 0): ["NW"],    # Top Left
    (-1, 1): ["NE"],    # Top Right
    (0, -1): ["W"],     # Left
    (0, 1): ["E"],      # Right
    (1, -1): ["SW"],    # Bottom Left
    (1, 0): ["SE"]      # Bottom Right
}
DIRECTION_SYMBOLS = [ v[0] for k,v in DIRECTIONS.items()]

EDGE_SYMBOLS = {
    1: "X-EDGE-X",
    -1: "O-EDGE-O"
}

BLOCKED_SYMBOL = ['B']
NEIGHBOR_SYMBOLS = [f'{p}{n}' for p in PLAYER_SYMBOLS.values() for n in DIRECTION_SYMBOLS] 

ADVANTAGE_COUNT = 5
EDGE_TO_EDGE_ADVANTAGE = [f'E{i+1}E' for i in range(ADVANTAGE_COUNT)]

STEPS_COMPLETED_THRESHOLD = 1
def STEPS_COMPLETED(board_size: int) -> list[str]:
    steps_symbols = []    
    for j in [k for k,v in PLAYER_SYMBOLS.items() if '_' not in str(v)]: # both players        
        for i in range(STEPS_COMPLETED_THRESHOLD, board_size + 1):
            steps_symbols.append(f'{PLAYER_SYMBOLS[j]}{PLAYER_SYMBOLS[j]:{PLAYER_SYMBOLS[j]}<{i}}')
    return steps_symbols

def ALL_SYMBOLS(board_size: int) -> list[str]:
    return list(PLAYER_SYMBOLS.values()) \
        + STEPS_COMPLETED(board_size) \
        + NEIGHBOR_SYMBOLS \
        + BLOCKED_SYMBOL \
        + EDGE_TO_EDGE_ADVANTAGE \
        + list(EDGE_SYMBOLS.values()) 
        
