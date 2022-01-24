import sys

from crossword import *
from queue import Queue
import copy


class CrosswordCreator():

    def __init__(self, crossword):
        """
        Create new CSP crossword generate.
        """
        self.crossword = crossword
        self.domains = {
            var: self.crossword.words.copy()
            for var in self.crossword.variables
        }

    def letter_grid(self, assignment):
        """
        Return 2D array representing a given assignment.
        """
        letters = [
            [None for _ in range(self.crossword.width)]
            for _ in range(self.crossword.height)
        ]
        for variable, word in assignment.items():
            direction = variable.direction
            for k in range(len(word)):
                i = variable.i + (k if direction == Variable.DOWN else 0)
                j = variable.j + (k if direction == Variable.ACROSS else 0)
                letters[i][j] = word[k]
        return letters

    def print(self, assignment):
        """
        Print crossword assignment to the terminal.
        """
        letters = self.letter_grid(assignment)
        for i in range(self.crossword.height):
            for j in range(self.crossword.width):
                if self.crossword.structure[i][j]:
                    print(letters[i][j] or " ", end="")
                else:
                    print("█", end="")
            print()

    def save(self, assignment, filename):
        """
        Save crossword assignment to an image file.
        """
        from PIL import Image, ImageDraw, ImageFont
        cell_size = 100
        cell_border = 2
        interior_size = cell_size - 2 * cell_border
        letters = self.letter_grid(assignment)

        # Create a blank canvas
        img = Image.new(
            "RGBA",
            (self.crossword.width * cell_size,
             self.crossword.height * cell_size),
            "black"
        )
        font = ImageFont.truetype("assets/fonts/OpenSans-Regular.ttf", 80)
        draw = ImageDraw.Draw(img)

        for i in range(self.crossword.height):
            for j in range(self.crossword.width):

                rect = [
                    (j * cell_size + cell_border,
                     i * cell_size + cell_border),
                    ((j + 1) * cell_size - cell_border,
                     (i + 1) * cell_size - cell_border)
                ]
                if self.crossword.structure[i][j]:
                    draw.rectangle(rect, fill="white")
                    if letters[i][j]:
                        w, h = draw.textsize(letters[i][j], font=font)
                        draw.text(
                            (rect[0][0] + ((interior_size - w) / 2),
                             rect[0][1] + ((interior_size - h) / 2) - 10),
                            letters[i][j], fill="black", font=font
                        )

        img.save(filename)

    def solve(self):
        """
        Enforce node and arc consistency, and then solve the CSP.
        """
        self.enforce_node_consistency()
        self.ac3()
        return self.backtrack(dict())

    def enforce_node_consistency(self):
        """
        Update `self.domains` such that each variable is node-consistent.
        (Remove any values that are inconsistent with a variable's unary
         constraints; in this case, the length of the word.)

        Recall that node consistency is achieved when, for every variable,
        each value in its domain is consistent with  the variable’s unary constraints.
        In the case of a crossword puzzle, this means making sure that every value in
        a variable’s domain has the same number of letters as the variable’s length.
        """
        domains = copy.deepcopy(self.domains)  # to avoid Set changed size during iteration Error

        for variable in domains:
            for value in domains[variable]:
                if variable.length != len(value):
                    self.domains[variable].remove(value)

        overlaps = []

        for x, y in self.crossword.overlaps:
            if self.crossword.overlaps[x, y]:
                overlaps.append((x, y))

    def revise(self, x, y):
        """
        Make variable `x` arc consistent with variable `y`.
        To do so, remove values from `self.domains[x]` for which there is no
        possible corresponding value for `y` in `self.domains[y]`.

        Return True if a revision was made to the domain of `x`; return
        False if no revision was made.
        """
        domains = copy.deepcopy(self.domains)  # to avoid Set changed size during iteration Error
        revised = False

        if not self.crossword.overlaps[x, y]:
            return revised

        """
        every choice in X's domain has a possible choice for Y 
        if I choose a value in X, is there a value for Y that satisfies the binary constraint 
        """
        for x_value in domains[x]:
            if not self.satisfies_constraint_for_y(domains, x, x_value, y):
                self.domains[x].remove(x_value)
                revised = True

        return revised

    def satisfies_constraint_for_y(self, domains, x, x_value, y):
        for y_value in domains[y]:
            i, j = self.crossword.overlaps[x, y]
            # todo: should I check that the two words are not equal too ??
            if x_value[i] == y_value[j] and (x_value != y_value):
                return True
        return False

    def ac3(self, arcs=None):
        """
        Update `self.domains` such that each variable is arc consistent.
        If `arcs` is None, begin with initial list of all arcs in the problem.
        Otherwise, use `arcs` as the initial list of arcs to make consistent.

        Return True if arc consistency is enforced and no domains are empty;
        return False if one or more domains end up empty.
        """
        queue = Queue()

        if arcs:
            # put arcs here
            queue.put(arcs)
        else:
            # put all arcs here
            for arc in self.get_all_arcs():
                queue.put(arc)

        while not queue.empty():
            x, y = queue.get()
            if self.revise(x, y):
                if len(self.domains[x]) == 0:
                    # no solution
                    return False
                for neighbor in self.crossword.neighbors(x):
                    if neighbor != y:
                        queue.put((neighbor, x))
        return True

    def get_all_arcs(self):
        arcs = []
        # todo: I should get x -> y . but also y -> x
        for x, y in self.crossword.overlaps:
            if self.crossword.overlaps[x, y]:
                arcs.append((x, y))
                arcs.append((y, x))

        return arcs

    def assignment_complete(self, assignment):
        """
        Return True if `assignment` is complete (i.e., assigns a value to each
        crossword variable); return False otherwise.

        An assignment is a dictionary where the keys are Variable objects and
        the values are strings representing the words those variables will take on.

        An assignment is complete if every crossword variable is assigned to a value
        (regardless of what that value is).

        The function should return True if the assignment is complete and return False otherwise.
        """
        return len(assignment.keys()) == len(self.crossword.variables)

    def consistent(self, assignment):
        """
        Return True if `assignment` is consistent (i.e., words fit in crossword
        puzzle without conflicting characters); return False otherwise.

        An assignment is a dictionary where the keys are Variable objects and the values are
        strings representing the words those variables will take on. Note that the assignment
        may not be complete: not all variables will necessarily be present in the assignment.

        An assignment is consistent if it satisfies all of the constraints of the problem:
        that is to say, all values are distinct, every value is the correct length,
        and there are no conflicts between neighboring variables.
        """
        if assignment is None:
            return False

        # all values are distinct
        if len(assignment.values()) != len(set(assignment.values())):
            return False

        # every value is the correct length
        for variable, value in assignment.items():
            if variable.length != len(value):
                return False

        # no conflicts between neighboring variables
        for variable in assignment:
            for neighbor in self.crossword.neighbors(variable):
                if neighbor in assignment:
                    i, j = self.crossword.overlaps[variable, neighbor]
                    x_value = assignment[variable]
                    y_value = assignment[neighbor]
                    if x_value[i] != y_value[j] or (x_value == y_value):
                        return False

        return True

    def order_domain_values(self, variable, assignment):
        """
        Return a list of values in the domain of `variable`, in order by
        the number of values they rule out for neighboring variables.
        The first value in the list, for example, should be the one
        that rules out the fewest values among the neighbors of `var`.
        """
        # todo: try least constraining values first. a variable that rules out fewer options for neighbors variables
        return self.domains[variable]

    def select_unassigned_variable(self, assignment):
        """
        Return an unassigned variable not already part of `assignment`.
        Choose the variable with the minimum number of remaining values
        in its domain. If there is a tie, choose the variable with the highest
        degree. If there is a tie, any of the tied variables are acceptable
        return values.
        """
        # todo: minimum remaining values (MRV) heuristic = variable with the smallest domain ()
        #       degree heuristic = most constrained variable (high degree)

        unassigned_variables = list(self.crossword.variables - assignment.keys())
        sorted_by_minimum_values = sorted(unassigned_variables, key=lambda variable: len(self.domains[variable]))

        return sorted_by_minimum_values[0]

    def backtrack(self, assignment: dict):
        """
        Using Backtracking Search, take as input a partial assignment for the
        crossword and return a complete assignment if possible to do so.

        `assignment` is a mapping from variables (keys) to words (values).

        If no assignment is possible, return None.
        """
        if self.assignment_complete(assignment):
            return assignment

        variable = self.select_unassigned_variable(assignment)

        for value in self.order_domain_values(variable, assignment):
            if self.is_value_consistent_with_assignment(variable, value, assignment):
                assignment[variable] = value
                # call ac3 with all arcs (Y, variable) where Y is a neighbor of X
                result = self.backtrack(assignment)
                if result is not None:
                    return result

            # backtrack (the current value didn't work, let's try another value)
            if variable in assignment:
                assignment.pop(variable)

        # failure
        return None

    def is_value_consistent_with_assignment(self, variable, value, assignment):
        copy_assignment = copy.deepcopy(assignment)
        copy_assignment[variable] = value
        return self.consistent(copy_assignment)


def main():
    # Check usage
    if len(sys.argv) not in [3, 4]:
        sys.exit("Usage: python generate.py structure words [output]")

    # Parse command-line arguments
    structure = sys.argv[1]
    words = sys.argv[2]
    output = sys.argv[3] if len(sys.argv) == 4 else None

    # Generate crossword
    crossword = Crossword(structure, words)
    creator = CrosswordCreator(crossword)
    assignment = creator.solve()

    # Print result
    if assignment is None:
        print("No solution.")
    else:
        creator.print(assignment)
        if output:
            creator.save(assignment, output)


if __name__ == "__main__":
    main()
