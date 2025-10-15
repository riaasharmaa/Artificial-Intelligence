import heapq

def get_manhattan_distance(from_state, to_state=[1, 2, 3, 4, 5, 6, 7, 0, 0]):
    """
    TODO: implement this function. This function will not be tested directly by the grader. 

    INPUT: 
        Two states (if the second state is omitted then it is assumed that it is the goal state)

    RETURNS:
        A scalar that is the sum of Manhattan distances for all tiles.
    """
    distance = 0
    for tile in range(1, 8):
        if tile in from_state:
            from_index = from_state.index(tile)
            to_index = to_state.index(tile)
            from_row, from_col = divmod(from_index, 3)
            to_row, to_col = divmod(to_index, 3)
            distance += abs(from_row - to_row) + abs(from_col - to_col)
    return distance

def print_succ(state):
    """
    TODO: This is based on get_successors function below, so should implement that function.

    INPUT: 
        A state (list of length 9)

    WHAT IT DOES:
        Prints the list of all the valid successors in the puzzle. 
    """
    succ_states = get_succ(state)

    for succ_state in succ_states:
        print(succ_state, "h={}".format(get_manhattan_distance(succ_state)))
        
def get_succ(state):
    """
    TODO: implement this function.

    INPUT: 
        A state (list of length 9)

    RETURNS:
        A list of all the valid successors in the puzzle (don't forget to sort the result as done below). 
    """
    succ_states = []
    empty_indices = [i for i, val in enumerate(state) if val == 0]
    for empty_index in empty_indices:
        moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        row, col = empty_index % 3, empty_index // 3
        for dx, dy in moves:
            new_row, new_col = row + dy, col + dx
            if 0 <= new_row < 3 and 0 <= new_col < 3:
                new_index = new_row * 3 + new_col
                new_state = state.copy()
                new_state[empty_index], new_state[new_index] = new_state[new_index], new_state[empty_index]
                if new_state not in succ_states:
                    succ_states.append(new_state)
    return sorted(succ_states)

    
def solve(state, goal_state=[1, 2, 3, 4, 5, 6, 7, 0, 0]):
    """
    TODO: Implement the A* algorithm here.

    INPUT: 
        An initial state (list of length 9)

    WHAT IT SHOULD DO:
        Prints a path of configurations from the initial state to the goal state along h values, the number of moves, and the max queue number in the format specified in the pdf.
    """

    priority_queue = []

    heapq.heappush(priority_queue, (get_manhattan_distance(state), tuple(state), 0, -1))
    visited = set()
    parents = {-1: None}
    info_for_state = {tuple(state): (0, get_manhattan_distance(state), -1)}
    max_length = 1

    while priority_queue:
        _, current_state_tuple, cost, parent_index = heapq.heappop(priority_queue)
        if current_state_tuple in visited:
            continue
        visited.add(current_state_tuple)
        current_state = list(current_state_tuple)
        if current_state == goal_state:
            break
        for succ_state in get_succ(current_state):
            succ_state_tuple = tuple(succ_state)
            heuristic = get_manhattan_distance(succ_state)
            if succ_state_tuple not in visited:
                parents[succ_state_tuple] = current_state_tuple
                info_for_state[succ_state_tuple] = (cost + 1, heuristic, current_state_tuple)
                heapq.heappush(priority_queue, (cost + 1 + heuristic, succ_state_tuple, cost + 1, current_state_tuple))
        if len(priority_queue) + 1 > max_length:
            max_length = len(priority_queue) + 1
   
    # This is a format helper.
    # build "state_info_list", for each "state_info" in the list, it contains "current_state", "h" and "move".
    # define and compute max length
    # it can help to avoid any potential format issue.

    path = []
    current_state_tuple = tuple(goal_state)

    while current_state_tuple is not None and current_state_tuple in parents:
        cost, heuristic, _ = info_for_state[current_state_tuple]
        path.append((list(current_state_tuple), heuristic, cost))
        current_state_tuple = tuple(parents[current_state_tuple])

    path.reverse()

    for current_state, heuristic, cost in path:
        print(current_state, "h={}".format(heuristic), "moves: {}".format(cost))

    print("Max queue length: {}".format(max_length))
    
if __name__ == "__main__":
    """
    Feel free to write your own test code here to exaime the correctness of your functions. 
    Note that this part will not be graded.
    """
    print_succ([3, 4, 6, 0, 0, 1, 7, 2, 5])
    print_succ([6, 0, 0, 3, 5, 1, 7, 2, 4])
    print_succ([0, 4, 7, 1, 3, 0, 6, 2, 5])
    solve([3, 4, 6, 0, 0, 1, 7, 2, 5])
    solve([6, 0, 0, 3, 5, 1, 7, 2, 4])
    solve([0, 4, 7, 1, 3, 0, 6, 2, 5])
