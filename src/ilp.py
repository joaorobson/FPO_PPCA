from ortools.linear_solver import pywraplp


class SummaryGenerator: 
    def get_summary(self, rel, red, lengths, K):
        """
        Implements the ILP formulation using OR-Tools.

        Args:
            rel: Dictionary {i: Rel(i)} representing the relevance of each textual unit.
            red: Dictionary {(i, j): Red(i, j)} representing redundancy between units.
            lengths: Dictionary {i: l(i)} representing the length of each textual unit.
            K: Integer, maximum total length allowed.

        Returns:
            Selected textual units as a list of indices.
        """
        solver = pywraplp.Solver.CreateSolver('SCIP')
        if not solver:
            raise Exception("SCIP solver is not available.")

        # Define variables
        alpha = {i: solver.BoolVar(f'alpha_{i}') for i in rel.keys()}
        alpha_ij = {(i, j): solver.BoolVar(f'alpha_{i}_{j}') for (i, j) in red.keys()}

        # Objective function: Maximize sum(Rel(i) * alpha_i) - sum(Red(i,j) * alpha_ij)
        objective = solver.Objective()
        for i in rel.keys():
            objective.SetCoefficient(alpha[i], rel[i])
        for (i, j), redundancy in red.items():
            objective.SetCoefficient(alpha_ij[(i, j)], float(-redundancy))
        objective.SetMaximization()

        # Constraints
        # (1) α_i, α_ij ∈ {0,1} -> Implicit in OR-Tools by using BoolVar()

        # (2) Sum of selected unit lengths must not exceed K
        solver.Add(sum(alpha[i] * lengths[i] for i in rel.keys()) <= K)

        # (3) α_ij - α_i ≤ 0  <=> α_ij ≤ α_i
        for (i, j) in red.keys():
            solver.Add(alpha_ij[(i, j)] <= alpha[i])

        # (4) α_ij - α_j ≤ 0  <=> α_ij ≤ α_j
        for (i, j) in red.keys():
            solver.Add(alpha_ij[(i, j)] <= alpha[j])

        # (5) α_i + α_j - α_ij ≤ 1
        for (i, j) in red.keys():
            solver.Add(alpha[i] + alpha[j] - alpha_ij[(i, j)] <= 1)

        # Solve
        status = solver.Solve()

        if status == pywraplp.Solver.OPTIMAL:
            selected_units = [i for i in rel.keys() if alpha[i].solution_value() > 0]
            return selected_units
        else:
            print("No optimal solution found.")
            return []

    # Example input (Replace with real values)
    #rel = {0: 0.9, 1: 0.8, 2: 0.7}  # Relevance scores
    #red = {(0, 1): 0.3, (0, 2): 0.2, (1, 2): 0.4}  # Redundancy penalties
    #lengths = {0: 10, 1: 15, 2: 5}  # Lengths of units
    #K = 20  # Length constraint

    ## Run ILP model
    #selected = ilp_global_inference(rel, red, lengths, K)
    #print("Selected textual units:", selected)

