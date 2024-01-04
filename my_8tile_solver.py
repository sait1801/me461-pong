import os
import numpy as np
import random

"""# A simple 8-tile puzzle implementation with animation support"""

class EightTile():
    '''
    This class implements a basic 8-tile board when instantiated
    You can shuffle it using shuffle() to generate a puzzle
    After shuffling, you can use manual moves using ApplyMove()
    '''
    # class level variables for image and animation generation
    cellSize = 50 #cell within which a single char will be printed
    xpadding = 13
    ypadding = 5
    fontname = '/usr/share/fonts/truetype/liberation/LiberationMono-Bold.ttf'
    fontsize = 45
    simSteps = 10 # number of intermediate steps in each digit move

    # a class level function
    def GenerateImage(b, BackColor = (255, 125, 60), ForeColor = (0,0,0)):
        '''
        Generates an image given a board numpy array
        0s are simply neglected in the returned image

        if b is not a board object but a string, a single char image is generated
        '''
        cellSize = EightTile.cellSize
        xpadding, ypadding = EightTile.xpadding, EightTile.ypadding
        font = EightTile.font

        if isinstance(b, str): # then a single character is expected, no checks
            img = Image.new('RGB', (cellSize, cellSize), BackColor ) # blank image
            imgPen = ImageDraw.Draw(img) # pen to draw on the blank image
            imgPen.text((xpadding, ypadding), b, font=font, fill=ForeColor) # write the char to img
        else: # the whole board is to be processed
            img = Image.new('RGB', (3*cellSize, 3*cellSize), BackColor ) # blank image
            imgPen = ImageDraw.Draw(img) # pen to draw on the blank image
            for row in range(3): # go row by row
                y = row * cellSize + ypadding
                for col in range(3): # then columns
                    x = col * cellSize + xpadding
                    txt = str(b[row, col]).replace('0', '')
                    # now that position of the current cell is fixed print into it
                    imgPen.text((x, y), txt, font=font, fill=ForeColor) # write the character to board image
        # finally return whatever desired
        return np.array(img) # return image as a numpy array

    def GenerateAnimation(board, actions, mName = 'puzzle', fps=15, debugON = False):
        # using each action collect images
        framez = []
        for action in actions: # for every action generate animation frames
            frm = board.ApplyMove(action, True, debugON)
            EightTile.print_debug(f'frame:{len(frm)} for action {action}', debugON)
            framez += frm
        imageio.mimsave(mName + ".gif", framez, fps=fps) #Creates gif out of list of images
        return framez

    def print_debug(mess, whether2print):
        # this is a simple conditional print,
        # you should prefer alternative methods for intensive printing
        if whether2print:
            print(mess)

    # object level stuff
    def __init__(me):
        # board is a numpy array
        me.__board = np.array([[1,2,3],[4,5,6],[7,8,0]])
        me.__winner = me.__board.copy() # by default a winning board is givenq
        # keep track of where 0 is, you can also use np.where, but I like it better this way
        me.__x, me.__y = 2, 2 # initially it is at the bottom right corner


    def shuffle(me, n = 1, debugON = False):
        '''
        randomly moves the empty tile, (i.e. the 0 element) around the gameboard
        n times and returns the moves from the initial to the last move
        Input:
            n: number of shuffles to performe, defaults to 1
        Output:
            a list of n entries [ [y1, x2], [y2, x2], ... , [yn, xn]]
            where [yi, xi] is the relative move of the empty tile at step i
            ex: [-1, 0] means that empty tile moved left horizontally
                note that:
                    on the corners there 2 moves
                    on the edges there are 3 moves
                    only at the center there are 4 moves

        Hence if you apply the negative of the returned list to the board
        step by step the puzzle should be solved!
        Check out ApplyMove()
        '''
        # depending on the current index possible moves are listed
        # think of alternative ways of achieving this, without using if conditions
        movez = [[0,1], [-1,0,1], [-1,0]]
        trace = []
        dxold, dyold = 0, 0 # past moves are tracked to be avoided, at first no such history
        for i in range(n):
            # note that move is along either x or y, but not both!
            # also no move at all is not accepted
            # we should also avoid the last state, i.e. an opposite move is not good
            dx, dy = 0, 0 # initial move is initialized to no move at all
            while ( dx**2 + dy**2 != 1 ) or (dx == -dxold and dy == -dyold): # i.e. it is either no move or a diagonal move
                dx = random.choice(movez[me.__x])
                dy = random.choice(movez[me.__y])
            # now that we have the legal moves, we also have the new coordinates
            xn, yn = me.__x+dx, me.__y+dy # record new coordinates
            trace.append([dy, dx]) # just keeping track of the move not the absolute position tomato, tomato
            me.__board[me.__y, me.__x], me.__board[yn, xn] = me.__board[yn, xn], me.__board[me.__y, me.__x]
            # enable print if debug is desired
            EightTile.print_debug(f'shuffle[{i}]: {me.__y},{me.__x} --> {yn},{xn}\n{me}\n', debugON)
            # finally update positions as well
            me.__x, me.__y = xn, yn

            dxold, dyold = dx, dy # keep track of old moves to avoid oscillations

        # finally return the sequence of shuffles
        # note that if negative of trace is applied to the board in reverse order board should reset!
        return trace

    def ApplyMove(me, move, generateAnimation = False, debugON = False):
        '''
        applies a single move to the board and updates it
        move is a list such that [deltaY, deltaX]
        this is manual usage, so it does not care about the previous moves
        if generateAnimation is set, a list of images will be returned that animates the move
        '''
        dy, dx = move[0], move[1]
        xn, yn = me.__x+dx, me.__y+dy # record new coordinates
        img = None
        imList = []
        if ( dx**2 + dy**2 == 1 and 0<=xn<=2 and 0<=yn<=2 ): # then valid
            if generateAnimation:
                # the value at the target will move to the current location
                cellSize = EightTile.cellSize
                simSteps = EightTile.simSteps
                c = me.__board[yn, xn]
                EightTile.print_debug(f'{c} is moving', debugON)
                # generate a template image
                temp = me.Board # copy board
                temp[yn, xn] = 0 # blank the moving number as well which is kept in c
                tempimg = EightTile.GenerateImage(temp) # get temp image
                tempnum = EightTile.GenerateImage(str(c)) # get the image for moving number
                # now at every animation step generate a new image
                # image will move in either along x or y, but linspace does not care
                xPos = np.linspace(xn * cellSize, me.__x * cellSize, simSteps+1, dtype=int)
                yPos = np.linspace(yn * cellSize, me.__y * cellSize, simSteps+1, dtype=int)
                Pos = np.vstack((yPos, xPos)).T # position indices in target image are in rows
                EightTile.print_debug(f'Position', debugON)
                # go over each pos pair to generate new images
                for p in range(Pos.shape[0]):
                    frm = tempimg.copy() # generate a template image
                    xi, yi = Pos[p,1], Pos[p,0]
                    EightTile.print_debug(f'moving to {yi}:{xi}', debugON)
                    #'''
                    frm[yi:yi+50, xi:xi+50, :] = tempnum # set image
                    EightTile.print_debug(f'frm = {frm.shape}, tempnum = {tempnum.shape}', debugON)
                    # finally add image to list
                    imList.append(frm)
            me.__board[me.__y, me.__x], me.__board[yn, xn] = me.__board[yn, xn], me.__board[me.__y, me.__x]
            me.__x, me.__y = xn, yn
            return imList
        else:
            return None
    @property
    def Position(me):
        # returns the position of the empty cell
        return [me.__y, me.__x]

    @property
    def Board(me):
        # returns a numpy array stating the current state of the board
        return me.__board.copy() # return a copy of the numpy array

    @property
    def isWinner(me):
        # returns true, if current board is a winner
        return np.array_equal(me.__winner, me.__board)

    @property
    def Winner(me):
        # returns true, if current board is a winner
        return me.__winner

    @property
    def BoardImage(me):
        return EightTile.GenerateImage(me.Board)


"""# Finally, write your own solver :)"""
from copy import deepcopy
import math

class Solve8():
    """
    This class is designed to solve the 8-puzzle problem using the A* algorithm.
    """

    def __init__(self):
        """
        Initializes the PuzzleSolver class with the possible directions of movement and the goal state.
        """
        self.MOVEMENTS = {"w": [-1, 0], "s": [1, 0], "a": [0, -1], "d": [0, 1]}
        self.GOAL = [[1, 2, 3], [4, 5, 6], [7, 8, 0]]

    class PuzzleTile:
        """
        This inner class represents a node in the search tree.
        """
        def __init__(self, current_state, previous_state, g, h, direction):
            """
            Initializes the PuzzleTile class with the current state, previous state, cost to reach the current state, heuristic cost, and direction of movement.
            """
            self.current_state = current_state
            self.previous_state = previous_state
            self.g = g
            self.h = h
            self.direction = direction

        def total_cost(self):
            """
            Returns the total cost of the node which is the sum of the cost to reach the current state and the heuristic cost.
            """
            return self.g + self.h

    def find_positions(self, state, element):
        """
        Returns the position of a given element in the current state.
        """
        for row in range(len(state)):
            if element in state[row]:
                return (row, state[row].index(element))

    def manhattan_distance(self, state):
        """
        Returns the Manhattan distance cost of the current state.
        """
        total_cost = 0
        for row in range(len(state)):
            for col in range(len(state[0])):
                position = self.find_positions(self.GOAL, state[row][col])
                total_cost += abs(row - position[0]) + abs(col - position[1])
        return total_cost

    def get_neighbouring_tiles(self, tile):
        """
        Returns the adjacent nodes of the given node.
        """
        neighbouring_tiles = []
        empty_position = self.find_positions(tile.current_state, 0)

        for direction in self.MOVEMENTS.keys():
            new_position = (empty_position[0] + self.MOVEMENTS[direction][0], empty_position[1] + self.MOVEMENTS[direction][1])
            if 0 <= new_position[0] < len(tile.current_state) and 0 <= new_position[1] < len(tile.current_state[0]):
                new_state = deepcopy(tile.current_state)
                new_state[empty_position[0]][empty_position[1]] = tile.current_state[new_position[0]][new_position[1]]
                new_state[new_position[0]][new_position[1]] = 0
                neighbouring_tiles.append(self.PuzzleTile(new_state, tile.current_state, tile.g + 1, self.manhattan_distance(new_state), direction))

        return neighbouring_tiles

    def get_optimal_tile(self, open_set):
        """
        Returns the node with the lowest total cost in the open set.
        """
        first_iteration = True

        for tile in open_set.values():
            if first_iteration or tile.total_cost() < best_cost:
                first_iteration = False
                optimal_tile = tile
                best_cost = optimal_tile.total_cost()
        return optimal_tile

    def construct_path(self, closed_set):
        """
        Returns the path from the start state to the goal state.
        """
        tile = closed_set[str(self.GOAL)]
        path = list()

        while tile.direction:
            path.append({
                'dir': tile.direction,
                'node': tile.current_state
            })
            tile = closed_set[str(tile.previous_state)]
        path.append({
            'dir': '',
            'node': tile.current_state
        })
        path.reverse()

        return path

    def Solve(self, puzzle):
        """
        This function takes a puzzle as input and returns the solution.
        """
        # Convert numpy.ndarray to list of lists
        puzzle = puzzle.Board
        puzzle_list = puzzle.tolist()

        # Solve the puzzle
        solution = self.main(puzzle_list)

        # Return the solution
        movessss = []
        for json_obj in solution:
            movessss.append(json_obj['dir'])
        del movessss[0]
        return movessss

    def main(self, puzzle_list):
        """
        The main function that solves the puzzle.
        """
        open_set = {str(puzzle_list): self.PuzzleTile(puzzle_list, puzzle_list, 0, self.manhattan_distance(puzzle_list), "")}
        closed_set = {}

        while True:
            current_tile = self.get_optimal_tile(open_set)
            closed_set[str(current_tile.current_state)] = current_tile

            if current_tile.current_state == self.GOAL:
                return self.construct_path(closed_set)

            neighbouring_tiles = self.get_neighbouring_tiles(current_tile)
            for tile in neighbouring_tiles:
                if str(tile.current_state) in closed_set.keys() or str(tile.current_state) in open_set.keys() and open_set[
                    str(tile.current_state)].total_cost() < tile.total_cost():
                    continue
                open_set[str(tile.current_state)] = tile

            del open_set[str(current_tile.current_state)]
