def build_full_boundary_matrix(filtered_graph):
    """
    Constructs a full boundary matrix for a filtered graph.

    Parameters:
    - filtered_graph: A list of tuples where each tuple consists of a tuple of
    vertices and a filtration value. We assume the list is ordered using its
    filtration value.

    Returns:
    - A full boundary matrix (NumPy array).
    """
    # Number of elements in the filtered graph
    n = len(filtered_graph)

    # Initialize the boundary matrix with False
    D = np.full((n, n), False, dtype=int)

    # Fill the boundary matrix
    for i, (x, _) in enumerate(filtered_graph):
        if len(x) == 2:  # Only for edges
            D[x[0], i] = -1
            D[x[1], i] = 1

    return D

def _pivot(column):
    """
    Find the pivot position (row index) of a column.

    Args:
        column: A 1D NumPy array.

    Returns:
        The index of the last non-zero element or None if the column is all zeros.
    """
    if np.any(column):  # Check if there are any non-zero elements
        return np.max(np.nonzero(column)[0])
    return None

def reduce_matrix(matrix):
    """
    Perform column reduction on a binary matrix (mod 2) and maintain the transformation matrix.

    Args:
        matrix: A binary (boolean) NumPy array representing the matrix to be reduced.

    Returns:
        reduced: The reduced matrix.
        triangular: The transformation matrix that tracks column operations.
    """
    n = matrix.shape[1]  # Number of columns in the matrix
    reduced = np.array(matrix)  # Copy the input matrix to avoid modifying the original
    triangular = np.eye(n, dtype=bool)  # Initialize the transformation matrix as an identity matrix

    # Iterate through each column `j` of the matrix
    for j in range(n):
        i = j
        # Compare column `j` with all preceding columns
        while i > 0:
            i -= 1
            # If the current column `j` is all zeros, stop processing it
            if not np.any(reduced[:, j]):
                break
            else:
                # Compute pivot positions for columns `j` and `i`
                piv_j = _pivot(reduced[:, j])
                piv_i = _pivot(reduced[:, i])

                # If the pivots are the same, perform column reduction
                if piv_i == piv_j:
                    # Update column `j` using the column operation `C_j = C_j + C_i (mod 2)`
                    reduced[:, j] = np.logical_xor(reduced[:, i], reduced[:, j])
                    # Update the transformation matrix to reflect the same column operation
                    triangular[:, j] = np.logical_xor(triangular[:, i], triangular[:, j])
                    # Reset `i` to `j` to ensure consistency in the reduction process
                    i = j

    # Return the reduced matrix and the transformation matrix
    return reduced, triangular

def compute_index_barcode(reduced):
    """
    Computes the barcode of a filtered graph from its reduced boundary matrix,
    using the indices of the basis elements instead of their filtration values.

    Args:
        reduced: A NumPy array representing the reduced boundary matrix.

    Returns:
        A list of tuples, where each tuple represents a bar [birth, death) in
        the basis order, not the filtration values.
    """
    n = reduced.shape[0]  # Number of basis elements
    barcode = []  # Initialize the list to store the barcode
    used_indices = set()  # Use a set for efficient membership checks

    # Add finite bars
    for j in range(n):
        if np.any(reduced[:, j]):  # Check if column j has any non-zero entries
            i = _pivot(reduced[:, j])  # Find the pivot row for column j
            barcode.append((i, j))  # Add the finite bar [birth, death)
            used_indices.update([i, j])  # Mark the indices as used for finite bars

    # Add infinite bars
    # Iterate over unused indices and check if they represent cycles
    infinite_indices = set(range(n)) - used_indices
    for i in infinite_indices:
        if not np.any(reduced[:, i]):  # Check if column i is all zeros
            barcode.append((i, np.inf))  # Add the infinite bar [birth, ∞)

    return sorted(barcode)  # Return the barcode sorted by birth times


def compute_barcode(reduced, filtration):
    """
    Computes the barcode of a filtered graph using the filtration values
    and dimensions associated with the basis elements.

    Args:
        reduced: A NumPy array representing the reduced boundary matrix.
        filtration: A list of tuples, where each tuple contains:
                    - A tuple of basis elements (vertices or edges).
                    - A filtration value.

    Returns:
        A list of tuples, where each tuple represents a bar [birth, death, dimension].
    """
    # Extract dimensions and filtration values from the filtration input
    dims = [len(pair[0]) - 1 for pair in filtration]  # Dimension of the basis elements
    values = [pair[1] for pair in filtration]  # Filtration values

    # Compute the index-based barcode
    index_barcode = compute_index_barcode(reduced)

    # Map indices to filtration values and dimensions
    barcode = [(values[b], values[d], dims[b]) if d != np.inf else (values[b], np.inf, dims[b])
               for b, d in index_barcode]

    return [tup for tup in barcode if tup[0] != tup[1]] # Remove empty bars

def compute_hardness(filtered_graph):
    """
    Computes the reduced degree 0 barcode from a filtered graph.

    Parameters:
    - filtered_graph (list): A list of tuples representing the filtered graph,
      where each tuple contains a simplex and its filtration value.

    Returns:
    - deg_0_barcode (list): A list of tuples representing the reduced degree 0 barcode.
    """
    # Build the full boundary matrix
    D = build_full_boundary_matrix(filtered_graph)

    # Reduce the boundary matrix
    reduced, _ = reduce_matrix(D)

    # Compute the full barcode
    full_barcode = compute_barcode(reduced, filtered_graph)

    # Extract degree 0 barcodes
    deg_0_barcode = [(bar[0], bar[1]) for bar in full_barcode if bar[2] == 0]

    # Remove the first element for reduced homology
    deg_0_barcode.pop(0)

    return deg_0_barcode