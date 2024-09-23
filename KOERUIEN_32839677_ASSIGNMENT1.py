"""

FIT2004 Assignment 1
By: Koe Rui En

"""

# import
from typing import Generic, TypeVar


"""
Question 1: Ultimate Fuse

"""

def fuse(fitmons: list[list[float]]) -> int:
    """
    Written by Koe Rui En

    Function Description:
        This fuse function is to fuse all of the given fitmons into only 1 fitmon and it returns the maximum cuteness score from fusing all of the fitmons.
        The maximum cuteness score of each 2 fitmons fused is calculated using bottom-up approach of Dynamic Programming. 
  
    Approach Description:
        - We firstly create a 2D matrix, called memo_array and it is defined as the maximum cuteness score of each 2 fitmons fuse. 
            memo_array[i][j] = {maxiumum cuteness score of 2 fitmons fuse}
        - The diagonal of the matrix is the maxiumum cuteness score for each 2 adjacent fitmons fuse.
        - The length of fitmons list is to determine the number of fitmons to be fused for each iteration. 
            - The memo_array is first filled diagonally with its cuteness_score when the length of fitmons list is 1 (base case). 
                memo_array[i][i] = fitmons[i][1]
        - Each possible start index (row, column) of fitmons is used to determine the leftmost fitmon and right fitmon to be fused.
            - left fitmon = fitmon[x] 
            - right fitmon = fitmon[x+1]
        - The cuteness score of the fused fitmon is calculated from each partition between leftmost and righmost fitmons.
        - The memo_array is updated with the maximum cuteness score of each fused fitmon.

    Input:
        fitmons: List of N fitmons, where N is a non-zero, positive integer, N>1 
                 Each fitmon in the list is a list of 3 values:
                    [affinity_left, cuteness_score, affinity_right]
          
    Return:
        cuteness_score: an integer where it is the maximum cuteness score from fusing all the N fitmons into single fitmon. 

    Time complexity: 
        Best case: 
            - O(N^3), where N is the number of items in the list fitmons.
        Best case analysis: 
            - (N(N-1))/2, as N(N-1) is the number of possible pairs of fitmons to be fused. 
               It is divided by 2 because the matrix is used half. 
            - There are N fitmons to be fused.
            - Thus, N*N(N-1)/2 = O(N^2)
        
        Worst case: 
            - O(N^3), where N is the number of items in the list fitmons.
        Worst case analysis: 
            - (N(N-1))/2, as N(N-1) is the number of possible pairs of fitmons to be fused. 
               It is divided by 2 because the matrix is used half. 
                - so its complexity is O((N^2-N)/2) = O(N^2)
            - There are N fitmons to be fused.
            - Thus, O(N) * O(N^2) = O(N^3)

    Space complexity: 
        Input space: 
            - O(N), where N is the number of items in the list fitmons.
        Input space analysis: 
            - This function takes a list of N fitmons where each fitmon is list with 3 values, so it is O(N).

        Auxilliary space: 
            - O(N^2), where N is the number of items in the list fitmons.
        Auxilliary space analysis:
            - This is due to additional matrix with size N*N to store the maximum cuteness score of the fused fitmon.
            - Thus, N*N = O(N^2)

    References:
    Reference is from the tutorial video FIT2004. https://youtu.be/DTpBszSWikA?si=MCIm-BxbScYjmJEy, 
    https://www.geeksforgeeks.org/matrix-chain-multiplication-dp-8/

    """

    # initialise cuteness score
    cuteness_score: int = 0 # O(1), assignment is constant

    # initialise memo array to store max cuteness score
    memo_array: list[list[int]] = [[0 for _ in range(len(fitmons))] for _ in range(len(fitmons))]
   
    # initalise base case
    # when there is only 1 fitmon, the cuteness score is the same as the fitmon
    for i in range(len(fitmons)): # O(N), where N is the number of fitmons
        memo_array[i][i] = fitmons[i][1] # O(1), assignment is constant
    
    # loop through memo array
    # fuse 2 fitmons at once, then increase fusion of fitmons till N fitmons (A,B,C -> AB -> ABC etc)
    for fitmon_number in range (2, len(fitmons)+1): # O(N), where N is the number of fitmons
        
        # find leftmost fitmon to start fusing
        for row in range(len(fitmons)-fitmon_number+1): # O(N), where N is the number of fitmons

            # find righmost fitmon to fuse with (last pos)
            # j = i+L-1, j<=n
            # row = 1, fitmon_number = 2, column = 1+2-1 = 1
            column: int = row + fitmon_number - 1 # O(1), arithmetic operation is constant

            # partition fitmons list into 2 parts (i<=x<=j)
            for x in range(row, column): # O(N), where N is the number of fitmons

                # everything is O(1), as integer comparision and assignment

                # get left fitmon affinity to fuse
                if x < (len(fitmons)-1):
                    affinity_left_fitmon: int = fitmons[x][2]
                else:
                    # affinity left of left fitmon is 0 when there is no fitmon on its left.  
                    affinity_left_fitmon: int = 0

                # get right fitmon affinity to fuse
                if x+1 < (len(fitmons)):
                    affinity_right_fitmon: int = fitmons[x+1][0]
                else: 
                    # affinity right of right fitmon is 0 when there is no fitmon on its right.
                    affinity_right_fitmon: int = 0 

                # compute cuteness score of the new fitmons
                # cuteness_score = fitmons[i][1] * fitmons[i][2]  + fitmons[i+1][1] * fitmons[i+1][0]
                #                = (cute_score_left_fitmon) * (aff_right_left_fitmon) + (cute_score_right_fitmon) * (aff_left_right_fitmon)

                new_cuteness_score: int = int(memo_array[row][x] * affinity_left_fitmon + memo_array[x+1][column] * affinity_right_fitmon)

                # compare cuteness score of two fused fitmons
                if new_cuteness_score > memo_array[row][column]:
                    memo_array[row][column] = new_cuteness_score
                else:
                    memo_array[row][column] = memo_array[row][column] 
                
    # get max cutness score
    cuteness_score = memo_array[0][len(fitmons)-1] # O(1), access array
    
    # return answer
    return cuteness_score # O(1), return is costant 

"""
Question 2: Delulu is not the Solulu

"""

class TreeMap:

    """

    TreeMap class is to represent the trees including Solulus in the forest and the roads between them of Delulu Forest
    using graph data structure.

    Written by Koe Rui En

    """

    def __init__(self, roads: list[tuple[int, int, int]], solulus: list[tuple[int, int, int]]) -> None:

        """
        Written by Koe Rui En

        Function Description:
        This __init__ function is the constructor of TreeMap, which used to initialise an instance of the TreeMap with the given roads and solulus lists.

        Approach Description:
        We need to make extra trees (vertices) for solulus to teleport to the specific destination tree before start, linked with original trees in the forest.
        I create extra trees and roads as we need to include the clay time of solulu tree and transport to the destinated tree.
        Hence, I create a multiverse tree map, which is exactly same paths with the original forest.

        The list of roads is looped through to find the maximum tree ID in the forest.
        The self.trees which is a list is created to store all the trees in the forest with the length of self.maxiumum_tree_id found.
        Since I create a multiverse tree map which have exact trees and paths as the original forest,
        I need to create a new list of roads, called new_road to store all the duplicated roads by increasing each element of tuple by the self.total_trees.
        The self.roads and new roads in the new_road are added to the respective tree in the self.trees array.

        Input:
            self: instance of TreeMap class

            roads: list of roads represented as a list of tuples (u, v, w), where 
                    u is the starting tree ID for the road.
                    v is the ending tree ID for the road.
                    w is the total time taken to travel down the road from tree-u to tree-v, and it is a non-negative integer.
            
            solulus: list of solulus represented as a list of tuples (x, y, z), where
                    x is the tree ID of the Solulu in the forest.
                    y is the time taken to clay and destroy the Solulu tree, and it is a non-negative integer.
                    z is the teleported tree ID if the Solulu tree is destroyed, and it is a non-negative integer.

        Return:
            None

        Time complexity: 
            Best case: 
                - O(|T|+|R|), 
                where, 
                    |T| is the set of unique trees in roads
                    |R| is R is the set of roads
            Best Case Analysis:
                - O(|T|), is to create list of arrays to store tree (vertices).
                - O(|R|), due to loop through the self.roads to add original roads to new_road array.
                - O(2*|R|) -> O(|R|), 
                    due to add_roads function is called twice to add road to respective tree, 
                    loop throuself.roads in the Tree class is the array to store all adjacenct roads 
                    for each position of the tree in self.trees array
                - Thus, O(|T|+ 3*|R|) = O(|R|)

            Worst case: 
                - O(|T|+|R|),
                where, 
                    |T| is the set of unique trees in roads
                    |R| is R is the set of roads
            Worst Case Analysis:
                - O(|T|), is to create list of arrays to store tree (vertices).
                - O(|R|), due to loop through the self.roads to add original roads to new_road array.
                - O(2*|R|) -> O(|R|), due to add_roads function is called to add road to respective tree. 
                - Thus, O(|T|+ 3*|R|) = O(|T| + |R|)

        Space complexity: 
            Input space: 
                - O(|T|+|R|),
                    where, 
                        |T| is the set of unique trees in roads
                        |R| is the set of roads
            Input space analysis:
                - O(|R|), due to list of roads 
                - O(|T|), due to list of solulus, since the worst size of solulus is |T|, so we can state O(|solulus|) = O(|T|), as it is bounded at T
                - Thus, O(|R|) + O(|T|) -> O(|T|+|R|)

            Auxiliary space: 
                - O(|T|+|R|),
                where, 
                    |T| is the set of unique trees in roads
                    |R| is is the set of roads
            Auxiliary space analysis:
                - O(|T|), create adjacency list, self.trees, to store all trees
                - O(|R|), due to self.roads list is created in the Tree class to store all adjacent roads that connected to each respective tree
                - Thus, O(|R|) + O(|T|) -> O(|T|+|R|)

        Reference:
        My reference is from the lecutre and tutorial FIT2004,
        https://youtu.be/_EVgZwKLfZg?si=uqX9xBLxgnZ4XoTA
        https://youtu.be/8S2jKSNC0BQ?si=KufLMQ7k2xfRDRjD

        """
        
        # list of roads in tuples (u, v, w)
        self.roads: list[tuple[int, int, int]] = roads # O(1), assignment is constant

        # list of solulus in tuples (x, y, z)
        self.solulus: list[tuple[int, int, int]] = solulus # O(1), assignment is constant

        # find max tree in tuple of road list
        self.maxiumum_tree_id: int = 0 # O(1), assignment is constant 

        # loop through roads to find max tree id
        for road in self.roads: # O(|R|), where R is the set of roads

            # extract road tuple
            u, v, _  = road # O(1), assignment is constant 

            # find max tree id
            self.maxiumum_tree_id = max(self.maxiumum_tree_id, u, v) # O(1), int comparision

        # find the total number of trees in the road
        self.total_trees: int = self.maxiumum_tree_id + 1 # O(1), assignment is constant 

        # array of Delulu trees (vertices)
        # build adjacency list of trees
        # create multiverse of graph by duplicating the graph
        self.trees = [None] * (self.total_trees * 2) # O(|T|), where |T| is set of unique trees in roads
    
        # initialise tree (vertex) in array
        # O(total_trees) -> O(1), constant loop 
        for i in range (self.total_trees*2): 
            self.trees[i] = Tree(i)

        # create new array to temporary store new roads/paths for graph duplication
        new_road: list[tuple[int, int, int]] = [] # O(1), create empty list

        # duplicate roads
        # tree id for the original road list is rised by total_trees when duplicated
        # O(|R|), where R is the set of roads
        for road in self.roads: 
            new_road.append((road[0] + self.total_trees, road[1] + self.total_trees, road[2])) # O(1), append is constant

        # add road to respective tree id, including the duplicated roads
        # time = O(|R|), where R is the set of roads
        # space = O(1)
        self.add_roads(self.roads) 
        self.add_roads(new_road)

    # add road (edges) to tree map
    def add_roads(self, argv_roads: list[tuple[int, int, int]]) -> None:
        """
        Written by Koe Rui En

        Function Description:
        This add_roads function is to add roads that connected to each tree in the self.trees array.

        Input:
            self: instance of TreeMap class
            argv_roads: list of roads represented as a list of tuples (u, v, w), where
                u is the starting tree ID for the road.
                v is the ending tree ID for the road.
                w is the total time taken to travel down the road from tree-u to tree-v, and it is a non-negative integer.

        Return:
            None

        Time complexity:
            Best case: 
                - O(|R|), where R is the set of roads
            Best case analysis: 
                - O(|R|), due to looping through the list of roads and add each road to the self.roads array for each tree instance in the self.trees array. 
                - O(1), due to assignment and also add_road function is O(1), as append is constant
                - Thus, O(|R|) + O(1) -> O(|R|)
    
            Worst case: 
                - O(|R|), where R is the set of roads
            Worst case analysis: 
                - O(|R|), due to looping through the list of roads and add roads to the self.roads array for each tree instance in the self.trees array. 
                - O(1), due to assignment and also add_road function is O(1), as append is constant
                - Thus, O(|R|) + O(1) -> O(|R|)
        
        Space complexity: 
            Input space:
                - O(|R|), where R is the set of roads
            Input space analysis:
                - O(|R|), due to the list of roads as input argument, argv_roads

            Auxiliary space:
                - O(1)
            Auxiliary space analysis:
                - O(1), only variables are defined, in-place, no additional space is created 

        Reference:
        My reference is from the lecutre and tutorial FIT2004,
        https://youtu.be/_EVgZwKLfZg?si=uqX9xBLxgnZ4XoTA
        https://youtu.be/8S2jKSNC0BQ?si=KufLMQ7k2xfRDRjD

        """

        for road in argv_roads: # O(|R|), where R is the set of roads
            
            # everything is O(1), assignment and constant space

            # get u, v, w from road
            u, v, w = road

            # create road (edge)
            current_road: Road = Road(u,v,w)
            # add road to each tree 
            current_tree: Tree = self.trees[u]
            current_tree.add_road(current_road)  # O(1)
        
    def escape(self, start: int, exits: list[int]) -> tuple[int, list[int]]:
        """
        Written by Koe Rui En

        Function Description:
        This escape function runs dijkstra algorithm to find the shortest path from the start tree to the exit tree in the forest including destroying Solulu trees. 
        It returns a tuple of (total_time, route) where including one fastest and optimal route from start tree to one of the exits trees including breaking the Solulu tree 
        and total time taken to escape from the forest.

        Approach Description (if the main function):
        There are 2 trees that we need to find. First is solulu tree and second is the exit tree.
        To escape from the forest, we need to break the solulu tree in prior.
        Before that, we loop through all the solulus to create a new road of each respective solulu tree in original map to the teleport tree in the multiverse
        with weight of time taken to claw the tree so that 2 graphs, original and multiverse, are connected.
        We also create a new tree (vertex) for all the exits to connect to it, so that we can run dijkstra from single source to single destination at once.
        We then run dijkstra to find the shortest time taken to reach the nearest exit tree from the start tree including time of destroying the solulu tree.
        After that, we backtrack the path to get the fastest route from the exit tree to the start tree.
        During backtracking, we will check the original tree is same as mutliverse tree, due to solulu can teleport us to itself in the multiverse.
        If the original tree is same as multiverse tree, we will append the original tree to the route list and skip the multiverse tree.
        There is also a special case that if no route exists, we will return None.

        Input:
            self: instance of TreeMap class
            start: an non-negative integer represents a tree ID in the forest to start escaping from the forest
            exits: a list of non-negative integers, where each integer represents a tree ID in the forest to escape to

        Return:
            A tuple of (total_time, route):
                total_time: the total time taken to escape from the forest
                route: a list of integers represents the tree IDs along the road, which is the shortest route to escape from the forest
            If no route exist, return None

        Time complexity: 
            Best case: 
            - O(|R|log|T|)
                where,
                    |R| is the set of roads
                    |T| is the set of unique trees in roads
            Best case analysis:
                - O(|T|), due to add_additional_tree, get_route and get_minimum_time_escape function
                - O(|R|log|T|), as we need to loop through all the adjacent roads (|R|) and 
                  perform min heap's operations, append, update, serve when performinng edge relaxation (log|T|) in the dijkstra function
                - Thus, O(|R|log|T|) + O(|T|) + O(log|T|) =  O(|R|log|T|)
    
           Worst case: 
            - O(|R|log|T|)
                where,
                    |R| is the set of roads
                    |T| is the set of unique trees in roads
            Worst case analysis:
                - O(|T|), due to add_additional_tree, get_route and get_minimum_time_escape function
                - O(|R|log|T|), as we need to loop through all the adjacent roads (|R|) and 
                  perform min heap's operations, append, update, serve when performinng edge relaxation (log|T|)
                - Thus, O(|R|log|T|) + O(|T|) + O(log|T|) =  O(|R|log|T|)

        Space complexity: 
            - Input space:
                -  O(|T|), where T is the set of unique trees in the roads
            - Input space analysis:
                - O(1), constant space for start as input parameter
                - O(|T|), due to add_additional_tree function and get_minimum_time_escape function
                - The size of exits_list is N, so O(N)
                - At worst, the size of exits_list is bounded at T, so it is O(|T|)
                - Thus, O(N) -> O(|T|)
                - In short, O(|T|) + O(|T|) + O(1) = 2*O(|T|) = O(|T|)
            
            - Auxiliary space: 
                - O(|T|+|R|), 
                where,
                    |T| is the set of unique trees in the roads
                    |R| is the set of roads
            - Auxiliary space analysis:
                - O(|T|), due to get_route and dijkstra function
                - O(|T| + |R|), due to we need to create new TreeMap instance before we can invoke the escape function,
                  so we had created an array to store all the trees and an array to store the roads that connect each respective tree instance  
                  in the _init_ function.
                - Thus, O(|T|) + O(|T| + |R|) = O(|T| + |R|)

        """

        # total time taken to travel
        total_time: int = 0 # O(1), assignment is constant 

        # to backtrack the entire route to source
        route: list[int] = [] # O(1), assignment is constant 

        # connect both graphs by creating new roads from solulu in the original map to teleport tree in multiuniverse
        # O(|T|), where T is the set of treees, at worst case, the size of solulus is |T|, it is bounded at T
        for solulu in self.solulus:

            # everything is O(1), assignment and constant space
            start_id, time_claw, transport_id = solulu
            new_road = Road(start_id, transport_id + self.total_trees, time_claw)
            self.trees[start_id].add_road(new_road)

        # create new vertex/tree to find shortest dist of each exit to each node
        new_start_vertex: int = self.add_additional_tree(exits)
        # time: O(|T|)
        # aux space: O(1)
        # input space:O(|T|)

        # run dijkstra from source to end 
        self.dijkstra(start, new_start_vertex)
        # time: O(|R| log |T|), where R is the set of roads, T is the set of trees
        # aux space: O(|T|)
        # input space: O(1)

        # get min time to escape
        nearest_tree, total_time = self.get_minimum_time_escape(exits)
        # time: O(|T|)
        # aux space: O(1)
        # input space: O(|T|)

        # no route exist, means does not exit from the forest
        # O(1), return and comparision is constant
        if (nearest_tree is None):
            return None

        # backtrack the path 
        route = self.get_route(nearest_tree, start)
        # time: O(|T|)
        # aux space: O(|T|)
        # input space: O(1)

        # return total time taken and route in tuple # O(1)
        return (total_time, route)
 
          
    def add_additional_tree(self, exits: list[int]) -> int:
        """
        Written by Koe Rui En

        Function Description:
        This add_additional_tree function is to create a additional tree for all the exits to connect to it, 
        and return new dummy tree ID as an integer.

        Input:
            self: instance of TreeMap class
            exits: a list of non-negative integers, where each integer represents a tree ID in the forest to escape to

        Return:
            dummy_tree.id: an integer represents the new tree ID which serves as destination for dijkstra. 

        Time complexity: 
            Best case: 
                -  O(|T|), where T is the set of unique trees in the roads
            Best case analysis:
                - The size of exits_list is N, so O(N)
                - We loop through the exits list to connect each exit to the new tree, dummy_tree.
                - At worst, the size of exits_list is bounded at T, so it is O(|T|)
                - Thus, O(N) -> O(|T|) 
    
           Worst case: 
                - O(|T|), where T is the set of unique trees in the roads
            Worst case analysis:
                - The size of exits_list is N, so O(N)
                - We loop through the exits list to connect each exit to the new tree, dummy_tree.
                - At worst, the size of exits_list is bounded at T, so it is O(|T|)
                - Thus, O(N) -> O(|T|) 
                
        Space complexity: 
            Input space: 
                - O(|T|), where T is the set of unique trees in the roads
            Input space analysis:
                - The size of exits_list is N, so O(N)
                - At worst, the size of exits_list is bounded at T, so it is O(|T|)
                - Thus, O(N) -> O(|T|)

            Auxiliary space: 
                - O(1)
            Auxiliary space analysis:
                - O(1), only variables is created, which is in-place and no additional space is created

        """

        # O(1), assignment is constant
        # create dummy tree/vertex to connect to all exits in the list with 0 weighted
        dummy_tree = Tree(len(self.trees))
        # add new vertex to tress list
        self.trees.append(dummy_tree)
        
        # connect each exit to new vertex in direction
        # O(N), where N is the size of exits list, at worst size of exits is |T| as all the trees can be assumed as exits, it bounded at T
        # -> Omitted complexity: O(|T|)
        for exit in exits: 
            
            # exit represent id of exit tree
            new_road = Road(exit, dummy_tree, 0) # O(1), assignment is constant
            dummy_tree.add_road(new_road) # O(1), append is constant

        # return the dummy tree id
        return dummy_tree.id # O(1)

    def dijkstra(self, source_tree: int, destination: int) -> None:
        """
        Written by Koe Rui En

        Function Description:
        This dijkstra function is find the shortest time travel taken to reach the new tree (vertex) as destination in TreeMap from the source tree.
        It will update the time_travel of each adjacent tree from the served tree during edge relaxation.

        Input:
            self: instance of TreeMap class
            source_tree: an non-integer starting tree ID in the forest to escape from
            destination: an integer represents the destination tree ID in the forest for dijkstra to reach

        Return:
            None

        Time complexity: 
            Best case:
                - O(|R|log|T|), 
                    where, 
                        |R| is the set of roads
                        |T| is the set of unique trees in roads
            Best case analysis: 
                - O(|T|), due to reset function as we need to loop and reset all trees to their original state.
                - O(log|T|), as we need to add source tree to min heap and rise to the top of the heap by comparing with its parent
                - O(|R|log|T|), as we need to loop through all the adjacent roads (|R|) and 
                  perform min heap's operations, append, update, serve when performinng edge relaxation (log|T|)
                - Thus, O(|R|log|T|) + O(|T|) + O(log|T|) =  O(|R|log|T|)

            Worst case:
                - O(|R| log |T|), 
                    where, 
                        |R| is the set of roads
                        |T| is the set of unique trees in roads
            Worst case analysis:
                - O(|T|), due to reset function as we need to loop and reset all trees to their original state.
                - O(log|T|), as we need to add source tree to min heap and rise to the top of the heap by comparing with its parent
                - O(|R|log|T|), as we need to loop through all the adjacent roads and 
                  perform min heap's operations, append, update, serve when performinng edge relaxation. 
                - Thus, O(|R|log|T|) + O(|T|) + O(log|T|) = O(|R|log|T|)

        Space complexity: 
            Input space:
                - O(1)
            Input space analysis:
                - O(1), in-place for source_tree and destination

            Auxiliary space:
                - O(|T|), where |T| is the set of unique trees in roads
            Auxiliary space analysis:
                - O(1), reset function is in-place, no additional space is created
                - O(1), all the operation of min heap (discovered) is in-place, no additional space is created
                - 2*O(|T|), since we create a min_heap list with size of self.trees and index_array list for index mapping in the MinHeap class
                - Thus, O(1) + O(1) + 2*O(|T|) =  2*O(|T|) =  O(|T|)

        Reference:
        My reference is from the lecutre and tutorial FIT2004,
        https://youtu.be/8Q_B7vly1g4?si=Biu2XtA8O4LF9j4p
        https://youtu.be/8S2jKSNC0BQ?si=KufLMQ7k2xfRDRjD

        """

        self.reset() # reset all vertices to original state
        # time: O(|T|)
        # aux space: O(1)

        # create min heap
        # O(1), assignment is constant
        discovered: MinHeap = MinHeap(len(self.trees)+1) # discovered == min heap
        source_tree: Tree = self.trees[source_tree] # source vertex
        source_tree.time_travel = 0 # source distance is 0
        discovered.append((source_tree.time_travel, source_tree))
        # time: worst = O(log |T|)
        # aux space: O(1)
    
        #  loop the minheap till empty
        while len(discovered) > 0: # O(len(discovered))

            # serve tree/vertex  from discovered queue
            # serve from heap means the vertex is visited distance is finalised
            current_tree: tuple[int, Tree] = discovered.serve() # worst = O(log |T|)
            # get served tree from tuple
            current_tree: Tree = current_tree[1] # O(1), array access
            # distance is finalised
            current_tree.visited = True  # O(1), assignment is constant

            if current_tree.id == destination:  # O(1)
                return

            # perfom edge relaxation (update time taken) on all adjacents of vertices
            for road in current_tree.roads: # O(|R|)
                
                # adjecent tree(v) of visted tree(u) been discovered or not
                # extract current adjecent vertex/tree 
                adjecent_tree: Tree = self.trees[road.v] # O(1)

                # distance is still inf
                if adjecent_tree.discovered == False:
                    # discovered, add to queue
                    adjecent_tree.discovered = True
                    # update distance of adjecent vertex
                    adjecent_tree.time_travel = current_tree.time_travel + road.w
                    # backtrack, to find the route from source
                    adjecent_tree.previous = current_tree
                    discovered.append((adjecent_tree.time_travel, adjecent_tree))
                    # time: worst = O(log |T|)
                    # aux space: O(1)
                
                # it is in heap (discovered), but not yet finalise (visited)
                # if i find a shorter route time, change it
                else: # v.visited == False
                    if adjecent_tree.time_travel > current_tree.time_travel + road.w:
                        # update distance
                        adjecent_tree.time_travel = current_tree.time_travel + road.w
                        # backtrack, to find the route from source
                        adjecent_tree.previous = current_tree
                        # update tree(vertex) in heap, with smaller distance; perform upheap
                        discovered.update((adjecent_tree.time_travel, adjecent_tree))
                        # time: worst = O(log |T|)
                        # aux space: O(1)
    
    # backtrack to find the shortest route to escape in prior to destroy solulu tree
    def get_route(self, exit_tree, start_tree: int) -> list[int]:
        """
        Written by Koe Rui En

        Function Description:
        This get_minimum_time_escape function is to find the shortest time to escape from the forest.

        Input:
            self: the instance of TreeMap class
            exit_tree: a tree object, which is the destination tree to escape
            start_tree: a non-negative integer, which is the starting tree ID in the forest to escape from

        Return:
            route: list of integers represents the tree IDs along the road, which is the shortest route to escape from the forest

        Time complexity: 
            Best case:
                - O(|T|), where |T| is the set of unique trees in roads
            Best case analysis: 
                - Since we create multiverse map, our exits are located in multiuniverse.
                - When break solulu tree, we will teleport to destinated trees in multiuniverse and so we can escape to multiverse exit (exit trees)
                - Thus, when we backtrack from the exit to source tree, 
                   we need to loop through from the exit tree which is located in multiuniverse to the source tree 
                   and find the fastest route, so it is O(|T|).
                - O(1), since assignment, append and insert operations is constant
                - Thus, O(|T|) + O(1) = O(|T|)

            Worst case:
                - O(|T|), where |T| is the set of unique trees in roads
            Worst case analysis:
                - Since we create multiverse map, our exits are located in multiuniverse.
                - When break solulu tree, we will teleport to destinated trees in multiuniverse and so we can escape to multiverse exit (exit trees)
                - Thus, when we backtrack from the exit to source tree, 
                   we need to loop through from the exit tree which is located in multiuniverse to the source tree 
                   and find the fastest route, so it is O(|T|).
                - O(1), since assignment, append and insert operations is constant
                - Thus, O(|T|) + O(1) = O(|T|)

        Space complexity:
            Input space: 
                - O(1) 
            Input space analysis:
                - O(1), constant space for start_tree and exit_tree as input parameters
            
            Auxiliary space:
                - O(|T|), where |T| is the set of unique trees in roads
            Aux space analysis:
                - O(|T|), as we need additional list to store the fastest route including breaks the Solulu trees.

        """
            
        # initialise empty route list
        route: list[int] = [] # O(1)

        # current tree is exit tree
        current_tree: Tree = exit_tree # O(1)

        # start from exit tree to the route
        # add exit tree to route, id-total_trees due to the graph duplicates
        original_tree_id: int = current_tree.id - self.total_trees # O(1)
        route.append(original_tree_id) # O(1)

        # backtrack to source tree
        while current_tree.id is not start_tree: # O(|T|), need to backtrack to the source tree
            
            # everything is O(1), insert assignment and constant space

            # check current tree > total_trees, means it is in multiverse
            if (current_tree.previous.id > self.total_trees):
                original_tree_previous_id = current_tree.previous.id - self.total_trees
                route.insert(0, original_tree_previous_id) 
            
            elif (route[0] != current_tree.previous.id):
                route.insert(0, current_tree.previous.id)
            
            current_tree = current_tree.previous 
        
        # return route of escape, O(1)
        return route 
    
    def get_minimum_time_escape(self, exits_list: list[int]):
        """
        Written by Koe Rui En

        Function Description:
        This get_minimum_time_escape function is to find the shortest time to escape from the forest.

        Input:
            self: the instance of TreeMap class
            exits_list: a list of non-negative integers, where each integer represents a tree ID in the forest to escape to

        Return:
            nearest_tree_exit: the nearest tree to escape from the forest
            minimum_time_escape: the minimum time taken to escape from the forest

        Time complexity: 
            Best case: 
                - O(|T|), where |T| is the set of unique trees in roads
            Best case analysis: 
                - O(N), where N is the size of exits_list
                - We loop through the size of exits_list to find the nearest exit
                - Since the size of exits_list at worst is |T| as all the trees can be assumed as exits, so it bounded at T
                - Hence, O(N) -> O(|T|)

            Worst case:
                - O(|T|), where |T| is the set of unique trees in roads
            Worst case analysis:
                - The size of exits_list is N, so O(N)
                - We loop through the size of exits_list to find the nearest exit
                - Since the size of exits_list at worst is |T| as all the trees can be assumed as exits, so it bounded at T
                - Hence, O(N) -> O(|T|)

        Space complexity: 
            Input space:
                - O(|T|), where |T| is the set of unique trees in roads
            Input space analysis:
                - The size of exits_list is N, so O(N)
                - At worst, the size of exits_list is bounded at T, so it is O(|T|)
                - Thus, O(N) -> O(|T|)

            Auxiliary space:
                - O(1)
            Auxiliary space analysis:
                - O(1), since only have variables, which is in-place, no additional space is created

        """
        
        # find the minimum time  
        minimum_time_escape: float = float('inf') # O(1), assignment is constant

        # find the nearest exit
        nearest_tree_exit: Tree = None # O(1), assignment is constant

        # loop through exit 
        # O(N), where N is the size of exits list
        # At worst size of exits is |T| as all the trees can be assumed as exits, it bounded at T
        #-> Omitted complexity: O(|T|)
        for exit in exits_list: 
            if self.trees[exit+self.total_trees].time_travel < minimum_time_escape:
                minimum_time_escape = self.trees[exit+self.total_trees].time_travel
                nearest_tree_exit = self.trees[exit+self.total_trees]
            
        # return the nearest tree exit
        return nearest_tree_exit, minimum_time_escape # O(1), return is constant
        
    # reset function
    def reset(self):
        """
        Written by Koe Rui En

        Function Description:
        This reset function is to reset all the roads(vertices) to its original state.

        Approach Description:
        Loop through all the roads that contained in the each position of self.road.
        Reset the value of the tree to its original state.

        Input:
            self: the instance of TreeMap class
        
        Return:
            None

        Time complexity: 
            Best case: 
                - O(|T|), where |T| is the set of unique trees in roads
            Best case analysis: 
                - O(|T|), we loop through the self.trees to reset all the trees to their original state.

            Worst case: 
                - O(|T|), where |T| is the set of unique trees in roads
            Worst case analysis: 
                - O(|T|), as we loop through the self.trees to reset all the trees to their original state.
                - O(1), as it is just reseting the value of the roads by assigning values
                - Thus, O(1) + O(|T|) -> O(|T|)
        
        Space complexity: 
            Input space: 
                - O(1)
            Input space analysis: 
                - O(1), in-place for self as parameter
            
            Auxiliary space: 
                - O(1)
            Auxiliary space analysis:
                - O(1), in-place, as no additional space is created 

            My reference is from the lecutre and tutorial recording FIT2004, 
            https://youtu.be/8S2jKSNC0BQ?si=KufLMQ7k2xfRDRjD

        """

        for tree in self.trees: # O(|T|), where T is the number of trees in the set
            
            # everything is O(1), as it is just assignment of value
            tree.discovered = False 
            tree.visited = False 
            tree.time_travel = float('inf')

# tree in TreeMap == vertex of graph
class Tree:

    def __init__(self, tree_id) -> None:
        """
        Written by Koe Rui En

        Function Description:
        This __init__ function is to initialise the attributes of an instance of Tree class.

        Input:
            self: the instance of TreeMap class
            tree_id: a non-negative integer that represents the tree ID in the forest
        
        Return:
            None

        Time complexity: 
            Best case:
                - O(1)
            Best case analysis:
                - O(1), since assignment of value and create empty list are constant

            Worst case: 
                - O(1)
            Worst case analysis:
                - O(1), since assignment of value and create empty list are constant

        Space complexity: 
            Input space: 
                - O(1)
            Input space analysis: 
                - O(1), in-place for tree_id

            Auxiliary space: 
                - O(1)
            Auxiliary space analysis: 
                - O(1), in-place for tree_id

        Reference:
            My reference is from the lecutre and tutorial recording FIT2004, 
            https://youtu.be/8S2jKSNC0BQ?si=KufLMQ7k2xfRDRjD

        """
        # everything is O(1), assignment is constant

        # tree id (vertex id)
        self.id = tree_id

        # list of roads(edges) (ori path)
        self.roads = []

        # check vertex is discovered/visited or not
        self.discovered = False
        self.visited = False

        # time travel along tree-u to tree-v for dijsktra
        self.time_travel = float('inf')

        # backtracking the previous tree
        self.previous = None

    # add road (edge) to the tree
    def add_road(self, road) -> None:
        """
        Written by Koe Rui En

        Function Description:
        This __init__ function is to initialise the attributes of an instance of Tree class.

        Input:
            self: the instance of Tree class
            road: a instance of Road class, which is the road that connected to the tree, which have 3 attributes, u, v, w.
        
        Return:
            None

        Time complexity: 
            Best case:
                - O(1)
            Best case analysis:
                - O(1), since append operation is constant

            Worst case: 
                - O(1)
            Worst case analysis:
                - O(1), since append operation is constant
                
        Space complexity: 
            Input space: 
                - O(1)
            Input space analysis: 
                - O(1), in-place for road as input argument

            Auxiliary space: 
                - O(1)
            Auxiliary space analysis: 
                - O(1), in-place for road as input argument
        
        Reference:
            My reference is from the lecutre and tutorial recording FIT2004, 
            https://youtu.be/8S2jKSNC0BQ?si=KufLMQ7k2xfRDRjD

        """
        # add ori paths of the graph
        self.roads.append(road) # O(1)
    
# road in TreeMap == edge of graph
class Road:

    def __init__(self, u: int, v: int, w: int) -> None:
        """""
        Written by Koe Rui En

        Function Description:
        This __init__ function is to initialise the road that connected to trees.

        Input:
            self: the instance of Road class
            u: a non-negative integer that represents the staring tree ID for the road is pointing from.
            v: a non-negative integer that represents the ending tree ID for the road is pointing to.
            w: a non-negative intger he weight of the road, which is the amount of time taken to travel down the road from tree-u to tree-v
        
        Return:
            None

        Time complexity: 
            Best case: 
                - O(1)
            Best case analysis: 
                - O(1), since assignment of value is constant
            
            Worst case:
                - O(1)
            Worst case analysis: 
                - O(1), since assignment of value is constant

        Space complexity: 
            Input space: 
                - O(1)
            Input space analysis: 
                - O(1), in-place for u, v, w
            
            Auxiliary space:
                - O(1)
            Auxiliary space analysis: 
                - O(1), in-place, no additional space is created
        
        Reference:
            My reference is from the lecutre and tutorial recording FIT2004, 
            https://youtu.be/8S2jKSNC0BQ?si=KufLMQ7k2xfRDRjD
        
        """

        # everything is O(1), as it is assignment

        self.u: int = u  # u - start tree id 
        self.v: int = v  # v - end tree id 
        self.w: int = w  # w - amount of time to travel the road from tree-u to tree-v

T = TypeVar('T')

class MinHeap(Generic[T]):

    """
    This is the MinHeap class.

    Reference: 
    All of the algorithms in the MinHeap class is refered from FIT1008 and did some modifications. 

    """

    def __init__(self, maximum_size: int) -> None:
        """""
        Written by Koe Rui En

        Function Description:
        This __init__ function is to initialise instance of MinHeap class.

        Input:
            self: the instance of MinHeap class
            maximum_size: maximum size of the min heap
        
        Return:
            None

        Time complexity: 
            Best case analysis: 
                - O(N), where N is the maxiumum size of self.heap_array and self.index_array 
                as we need O(N) time to create array of size N
            Worst case analysis: 
                - O(N), where N is the maxiumum size of self.heap_array and self.index_array 
                as we need O(N) time to create array of size N

        Space complexity: 
            Input space analysis: 
                - O(1)
            Auxiliary space analysis: 
                - O(N), where N is the maximum size of self.heap_array and self.index_array
                as we need O(N) space to create array of size N
        
        """

        # create priority queue (heap) 
        self.heap_array = [None] * (maximum_size) # O(N), where N is the maximum size of array
        
        # keep track length of heap/ # elements in heap
        self.length = 0 # O(1), assignment

        # index mapping of vertex in heap
        self.index_array = [None] * (maximum_size-1) # O(N), where N is the maximum size of array
    
    def __len__(self) -> int:
        """
        Written by Koe Rui En

        Function Description:
        This __len__ function returns the number of elements contain in the MinHeap

        Input:
            self: the instance of MinHeap class
        
        Return:
            self.length: number of elements in the MinHeap

        Time complexity: 
            Best case analysis: O(1)
            Worst case analysis: O(1)

        Space complexity: 
            Input space analysis: O(1)
            Auxiliary space analysis: O(1)

        """

        return self.length
    
    def append(self, element: T) -> None:

        """
        Written by Koe Rui En

        Function Description:
        Add the element into MinHeap, and invoke the rise function to swap this element to the correct position.

        Input:
            self: the instance of MinHeap class
            element: a tuple to be added to the MinHeap, where the first element is distance to reach the vertex
                     and the second element is the vertex 
        
        Return: 
            None

        Time Complexity: 
            Best case analysis:
                - O(1)
                depends on rise function, the element is larger than or equal to its parent
            Worst case analysis  
                - O(log N), where N is the number of elements in the MinHeap
                depends on rise function, the element rises all the way to the top of the heap when heap order is broken

        Space Complexity:
            Input space analysis: O(1)
            Auxiliary space analysis: O(1)

        """

        # O(1), assignment and arithmetic operation are constant

        # increase length of min heap
        self.length += 1

        # add new element in bottom of heap
        self.heap_array[self.length] = element

        # initialise index mapping arr for each vertex in heap
        self.index_array[element[1].id] = self.length

        # check if heap order broken, parent > new element
        # rise, swap new element with parent 
        self.rise(self.length)
        # best = O(1), worst = O(log N)
    
    # upheap (bubble up)
    def rise(self, k: int) -> None:
        """
        Written by Koe Rui En

        Function Description:
        Rise element at index k to its correct position

        Input: 
            self: the instance of MinHeap class
            k: an integer that represents the index of new added element 
        
        Return: 
            None

      Time Complexity: 
            Best case analysis:
                - O(1)
                when the element is larger than or equal to its parent
            Worst case analysis:
                - O(log N), where N is the number of elements in the MinHeap
                when the element rises all the way to the top of the heap to its correct position 
        
        Space Complexity:
            Input space analysis: O(1)
            Auxiliary space analysis: O(1)

        """

        # obtain element from heap arr at k index (key, data)
        element = self.heap_array[k]

        # check heap order, new element < parent node 
        while k > 1 and element[0] < self.heap_array[k//2][0]:
            # update index mapping arr
            self.swap(self.index_array, element[1].id, self.heap_array[k//2][1].id)
            # swap with parent if smaller than parent 
            self.swap(self.heap_array, k, k//2)
            # update the index 
            k = k//2
    
    # serve min element from heap
    def serve(self) -> T:
        """
        Written by Koe Rui En

        Function Description:
        Remove (and return) the minimum element from the heap.

        Input:
            self: the instance of MinHeap class
        
        Return:
            minimum_element: the smallest element with minimum integer in the tuple with index 1
        
        Time Complexity: 
            Best case analysis:
                - O(1)
                depends on sink function, the element is larger than or equal to its parent
            Worst case analysis  
                - O(log N), where N is the number of elements in the MinHeap
                depends on sink function, the element sinks all the way to the bottom of the heap to its correct position
        
        Space Complexity:
            Input space analysis: O(1)
            Auxiliary space analysis: O(1)

        """ 

        # get min element 
        minimum_element = self.heap_array[1]

        # decrease length of arr
        self.length -=1

        # swap min element with the last node
        if self.length > 0:
            # update index mapping arr
            self.swap(self.index_array, minimum_element[1].id, self.heap_array[self.length+1][1].id)
            # swap min element with last node
            self.swap(self.heap_array, 1, self.length+1)
            # sink the element to correct position
            self.sink(1)

        # return the smallest element 
        return minimum_element

    # downheap (bubble down)
    def sink(self, k: int) -> None:
        """ 
        Written by Koe Rui En

        Function Description:
        Make the element at index k sink to the correct position.

        Input: 
            self: the instance of MinHeap class
            k: an integer that represents the index of element in the array
        
        Return: 
            None

        Time Complexity: 
            Best case analysis:
                - O(1)
                when the element is larger than or equal to its parent
            Worst case analysis  
                - O(log N), where N is the number of elements in the MinHeap
                depends on sink function, the element sinks all the way to the bottom of the heap to its correct position
        
        Space Complexity:
            Input space analysis: O(1)
            Auxiliary space analysis: O(1)

        """

        # extract element at k index from heap arr
        element = self.heap_array[k]

        # swap with smallest element 
        while 2*k <= self.length:
            
            # found the smallest child node index
            min_child = self.smallest_child(k)

            # check if child node >= parent node 
            if self.heap_array[min_child][0] >= element[0]: 
                break
            # update index mapping arr
            self.swap(self.index_array, self.heap_array[k][1].id, self.heap_array[min_child][1].id)
            # swap parent node with smallest child node
            self.swap(self.heap_array, k, min_child)
            # update index
            k = min_child
        

    def smallest_child(self, k: int) -> int:
        """
        Written by Koe Rui En

        Function Description:
        Returns the index of k's child with smallest value.

        Input: 
            self: the instance of MinHeap class
            k: an integer that represents the index of element in the array

        Return:
            the child of k with minimum integer

        Time Complexity:
            Best case analysis: O(1)
            Worst case analysis: O(1)
            
        Space Complexity:
            Input space analysis: O(1)
            Auxiliary space analysis: O(1)
        
        """

        if 2*k == self.length or self.heap_array[2*k][0] < self.heap_array[2*k+1][0]:
            return 2*k
        else:
            return 2*k+1
        
   
    # update element in the min heap 
    def update(self, update_distance_vertex: T) -> None:
        """
        Written by Koe Rui En

        Function Description:
        Update vertex with new its distance value in heap and perform rise after performing edge relaxation of Dijkstra
       
        Input: 
            self: the instance of MinHeap class
            update_distance_vertex: vertex with smaller distance value in heap
        
        Return:
            None

        Time Complexity: 
            Best case analysis:
                - O(1)
                depends on rise function, the element is larger than or equal to its parent
            Worst case analysis:
                - O(log N), where N is the number of elements in the MinHeap
                depends on rise function, the element rises all the way to the top of the heap to its correct position 
        
        Space Complexity:
            Input space analysis: O(1)
            Auxiliary space analysis: O(1)

        """
            
        # find index of vertex in heap
        index = self.index_array[update_distance_vertex[1].id] # O(1)

        # update vertex value in heap
        self.heap_array[index] = update_distance_vertex # O(1)

        # perform rise
        # rise: worst = O(log N), rise all the way to top of heap
        self.rise(index)

    # swap function
    def swap(self, array: list[T], i: int, j: int) -> None:
        """
        Written by Koe Rui En

        Function Description:
        Swapping elements in the min heap. 

        Input:
            array: an array for index mapping of the Tree object in the min heap
            i: index of child vertex in index mapping array
            j: index of parent vertex in index mapping array
        
        Return:
            None
        
         Time Complexity:
            Best case analysis: O(1)
            Worst case analysis: O(1)
            
        Space Complexity:
            Input space analysis: O(1)
            Auxiliary space analysis: O(1)

        """

        # O(1), swapping is constant
        array[i], array[j] = array[j], array[i]