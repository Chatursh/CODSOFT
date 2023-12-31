import random

# Function to print the Tic-Tac-Toe board
def print_board(board):
    for row in board:
        print(" | ".join(row))
        print("-" * 13)

# Function to check if a player has won
def check_winner(board, player):
    for row in board:
        if all(cell == player for cell in row):
            return True
    for col in range(3):
        if all(row[col] == player for row in board):
            return True
    if all(board[i][i] == player for i in range(3)) or all(board[i][2 - i] == player for i in range(3)):
        return True
    return False

# Function to check if the board is full (a draw)
def is_board_full(board):
    return all(cell != ' ' for row in board for cell in row)

# Function to get a list of empty cells on the board
def get_empty_cells(board):
    return [(i, j) for i in range(3) for j in range(3) if board[i][j] == ' ']

# Minimax algorithm with recursive evaluation of game states
def minimax(board, depth, maximizing_player):
    if check_winner(board, 'O'):
        return 1
    if check_winner(board, 'X'):
        return -1
    if is_board_full(board):
        return 0

    if maximizing_player:
        max_eval = float('-inf')
        for i, j in get_empty_cells(board):
            board[i][j] = 'O'
            eval = minimax(board, depth + 1, False)
            board[i][j] = ' '
            max_eval = max(max_eval, eval)
        return max_eval
    else:
        min_eval = float('inf')
        for i, j in get_empty_cells(board):
            board[i][j] = 'X'
            eval = minimax(board, depth + 1, True)
            board[i][j] = ' '
            min_eval = min(min_eval, eval)
        return min_eval

# Function to find the best move for the AI using the minimax algorithm
def best_move(board):
    best_val = float('-inf')
    best_move = None
    for i, j in get_empty_cells(board):
        board[i][j] = 'O'
        move_val = minimax(board, 0, False)
        board[i][j] = ' '
        if move_val > best_val:
            best_val = move_val
            best_move = (i, j)
    return best_move

# Main function to play the Tic-Tac-Toe game
def play_tic_tac_toe():
    board = [[' ' for _ in range(3)] for _ in range(3)]
    box_names = [['top-left', 'top-middle', 'top-right'],
                 ['middle-left', 'center', 'middle-right'],
                 ['bottom-left', 'bottom-middle', 'bottom-right']]
    
    print("Welcome to Tic-Tac-Toe!")
    print_board(board)

    while True:
        move = input("Enter your move (e.g., 'top-left', 'center', etc.): ").lower()
        box_mapping = {
            'top-left': (0, 0), 'top-middle': (0, 1), 'top-right': (0, 2),
            'middle-left': (1, 0), 'center': (1, 1), 'middle-right': (1, 2),
            'bottom-left': (2, 0), 'bottom-middle': (2, 1), 'bottom-right': (2, 2)
        }
        if move in box_mapping:
            i, j = box_mapping[move]
            if board[i][j] == ' ':
                board[i][j] = 'X'
            else:
                print("Invalid move. Try again.")
                continue
        else:
            print("Invalid input. Use box names like 'top-left', 'center', etc.")
            continue

        print_board(board)
        if check_winner(board, 'X'):
            print("Congratulations! You win!")
            break
        elif is_board_full(board):
            print("It's a draw!")
            break

        print("AI's turn:")
        ai_i, ai_j = best_move(board)
        board[ai_i][ai_j] = 'O'
        print(f"AI chose {box_names[ai_i][ai_j]}:")
        print_board(board)

        if check_winner(board, 'O'):
            print("AI wins! Better luck next time.")
            break
        elif is_board_full(board):
            print("It's a draw!")
            break

if __name__ == "__main__":
    play_tic_tac_toe()
