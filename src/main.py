from hexboard import HexBoard
from GraphTsetlinMachine.tm import MultiClassGraphTsetlinMachine
from sklearn.metrics import accuracy_score  # Import accuracy function from sklearn
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np
import os
from init_graphs import init_graphs, init_graphs_from
import argparse
from matplotlib import pyplot as plt

# Set the directory to this script's path
os.chdir(os.path.dirname(os.path.realpath(__file__)))

# Path to the hex game data                       
#hex_game_data_file = '../data/3x3_small-formatet.csv'
hex_game_data_file = '../data/hex_games_1_000_000_size_7_100.csv'
#hex_game_data_file = '../data/hex_games_1_000_000_size_7_10000.csv'
#hex_game_data_file = '../data/hex_games_1_000_000_size_7_100000.csv'
#hex_game_data_file = '../data/hex_games_1_000_000_size_7_10000_balanced.csv'
#hex_game_data_file = '../data/hex_games_100_000_balanced_80_20.csv'
#hex_game_data_file = '../data/hex_9x9_2moves.csv'
#hex_game_data_file = '../data/hex_9x9_5moves.csv'


def default_args(**kwargs):
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--number-of-clauses", default=10000, type=int) #optimal 10000
    parser.add_argument("--T", default=5000, type=int) #optimal 5000
    parser.add_argument("--s", default=2, type=float)
    parser.add_argument("--depth", default=2, type=int)
    parser.add_argument("--hypervector-size", default=128, type=int) #128
    parser.add_argument("--hypervector-bits", default=2, type=int)
    parser.add_argument("--message-size", default=512, type=int)
    parser.add_argument("--message-bits", default=2, type=int)
    parser.add_argument('--double-hashing', dest='double_hashing', default=False, action='store_true')
    parser.add_argument("--max-included-literals", default=200, type=int) #200


    args = parser.parse_args()
    for key, value in kwargs.items():
        if key in args.__dict__:
            setattr(args, key, value)
    return args

def display_boards(hexboards):
    while True:
        try:
            nr = int(input("Visual board nr: "))
            if nr < 0:
                break
            if nr > len(hexboards)-1:
                print("Board not found")
                continue

            hexboards[nr].visualize_board()
        except:
            break

if __name__ == '__main__':
    
    args = default_args()

    # If set to true we may inspect board visually
    is_display_board = True

    # test with random data
    is_using_random_data = True
    
    if not is_using_random_data: np.random.seed(42)
    
    print("Loading training and test data")

    # Load the training and test data as Graphs objects and labels
    train_hexboards, test_hexboards = HexBoard.load_training_test_data(
        hex_game_data_file,     # path to hex game data file
        0.2,                    # testsize
        True,                   # has header
        is_using_random_data)   # do not randomize -> get same result every time during test

    # visualize boards, enter a negative number to exit function
    if is_display_board: display_boards(train_hexboards)
    
    y_train = np.array([board.get_labels() for board in train_hexboards])
    y_test = np.array([board.get_labels() for board in test_hexboards])

    # setup graphs
    print("setup graphs")
    train_graphs = init_graphs(train_hexboards, hypervector_size=args.hypervector_size, hypervector_bits=args.hypervector_bits)    
    test_graphs = init_graphs_from(train_graphs, test_hexboards)

    print("Graphs initialized for training and test data.")
    print(f"Train graphs initialized with {len(train_graphs.graph_node_id)} graphs.")
    print(args)

    tm = MultiClassGraphTsetlinMachine(
        args.number_of_clauses,
        args.T,
        args.s,
        depth=args.depth,
        max_included_literals=args.max_included_literals,
        grid=(16*13,1,1),
        block=(128,1,1)
    )
        
    best_accuracy = 0
    best_f1 = 0
    best_recall = 0
    best_precision = 0
    sucsess = 0

    accuracy_history: list[float] = []
    recall_history: list[float] = []
    f1_history: list[float] = []
    precision_history: list[float] = []
    for epoch in range(args.epochs):
        if sucsess != 3:       
            tm.fit(train_graphs, y_train, epochs=1, incremental=True)  # Train for one epoch

            y_pred = tm.predict(test_graphs)  # Predict on test data
            #print(f"Epoch#{epoch+1} -- Accuracy train: {np.mean(y_train == tm.predict(train_graphs))}", end=' ')
            #print(f"-- Accuracy test: {np.mean(y_test == tm.predict(test_graphs))} ")
        
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            
            result_train = 100*(tm.predict(train_graphs) == y_train).mean()
            
            #accuracy_history.append(accuracy*100)
            accuracy_history.append(result_train)
            precision_history.append(precision*100)
            recall_history.append(recall*100)
            f1_history.append(f1*100)

            best_accuracy = accuracy if accuracy > best_accuracy else best_accuracy
            best_f1 = f1 if f1 > best_f1 else best_f1
            best_recall = recall if recall > best_recall else best_recall
            best_precision = precision if precision > best_precision else best_precision
            if result_train == 100:
                sucsess += 1
            else:
                sucsess = 0
            #print(f"Epoch:{epoch}: Accuracy: {accuracy * 100:.2f}%, F1: {f1 * 100:.2f}, Recall: {recall * 100:.2f}, precision: {precision * 100:.2f} (-- Accuracy test:{ result_train})")
            print(f"Epoch:{epoch}: Accuracy: {result_train}%")
             # Optionally, inspect internal state (clause weights, TA states) to understand training
            #ta_state, clause_weights, num_outputs, num_clauses, num_literals, depth, state_bits, ta_chunks, min_y, max_y = tm.get_state()

            # Print a sample of clause weights to get a sense of how they evolve
            #print("Sample clause weights (first 10):", clause_weights[:10])
        else: 
            break
        
   
    # Make predictions on the test set
    y_pred = tm.predict(test_graphs)

    # Calculate and print the accuracy
    accuracy = accuracy_score(y_test, y_pred)
    
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f"Test Accuracy: {accuracy * 100:.2f}% ({best_accuracy*100}%)")
    print(f"Precision (1): {precision:.2f} ({best_precision})")
    print(f"Recall (1): {recall:.2f} ({best_recall})")
    print(f"F1 Score (1): {f1:.2f} ({best_f1})")


    plt.plot(accuracy_history)
    #plt.plot(precision_history)
    #plt.plot(recall_history)
    #plt.plot(f1_history)
    plt.ylim(bottom=0, top=100)
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid(True)
    #plt.legend(['Accuracy', 'Precision', 'Recall', 'F1'])
    #plt.show()


 