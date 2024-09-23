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
