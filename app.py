import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt

# Configure the page
st.set_page_config(page_title="Tensor Data Processing: All Topics", layout="wide")
st.title("Tensor Data Processing: Simple Examples for All Topics")

# Create tabs for each topic
tab_names = [
    "1. Data Manipulation",
    "2. Vector",
    "3. Matrices",
    "4. Tensors",
    "5. Tensor Constructors",
    "6. Tensor Operators",
    "7. Dot Product",
    "8. Matrix-Vector Multiplication",
    "9. Matrix Multiplication",
    "10. Norms",
    "11. Broadcasting",
    "12. Indexing & Slicing",
    "13. Saving Memory",
    "14. Converting Tensors"
]
tabs = st.tabs(tab_names)

# ----------------------------------------------------------------------
# 1. Data Manipulation
with tabs[0]:
    st.header("1. Data Manipulation")
    st.markdown("""
    **Data Manipulation** involves changing or transforming data.  
    Here, we create a random tensor and normalize it between 0 and 1.
    """)
    # Simple example: create a 3x3 tensor and normalize it.
    tensor = torch.randn(3, 3)
    tensor_min = torch.min(tensor)
    tensor_max = torch.max(tensor)
    tensor_normalized = (tensor - tensor_min) / (tensor_max - tensor_min)
    st.write("Original Tensor:", tensor)
    st.write("Normalized Tensor:", tensor_normalized)
    st.code("""
tensor = torch.randn(3, 3)
tensor_min = torch.min(tensor)
tensor_max = torch.max(tensor)
tensor_normalized = (tensor - tensor_min) / (tensor_max - tensor_min)
""", language="python")

# ----------------------------------------------------------------------
# 2. Vector
with tabs[1]:
    st.header("2. Vector")
    st.markdown("""
    A **vector** is a 1-dimensional tensor (a list of numbers).
    """)
    simple_vector = torch.tensor([1, 2, 3, 4])
    st.write("Vector:", simple_vector)
    st.code("""
simple_vector = torch.tensor([1, 2, 3, 4])
""", language="python")

# ----------------------------------------------------------------------
# 3. Matrices
with tabs[2]:
    st.header("3. Matrices")
    st.markdown("""
    A **matrix** is a 2-dimensional tensor (a table of numbers).
    """)
    simple_matrix = torch.tensor([[1, 2], [3, 4]])
    st.write("Matrix:", simple_matrix)
    st.code("""
simple_matrix = torch.tensor([[1, 2],
                              [3, 4]])
""", language="python")

# ----------------------------------------------------------------------
# 4. Tensors
with tabs[3]:
    st.header("4. Tensors")
    st.markdown("""
    A **tensor** is a multi-dimensional array.  
    - A scalar is a 0-D tensor.
    - A vector is a 1-D tensor.
    - A matrix is a 2-D tensor.
    """)
    # Example of a 0-D tensor (scalar)
    scalar = torch.tensor(5)
    st.write("Scalar (0-D Tensor):", scalar)
    st.code("""
scalar = torch.tensor(5)
""", language="python")

# ----------------------------------------------------------------------
# 5. Common Tensor Constructors
with tabs[4]:
    st.header("5. Tensor Constructors")
    st.markdown("""
    Common tensor constructors include:
    - `torch.zeros()`: Tensor of zeros.
    - `torch.ones()`: Tensor of ones.
    - `torch.randn()`: Tensor with random numbers.
    """)
    zeros_tensor = torch.zeros(2, 2)
    ones_tensor = torch.ones(2, 2)
    rand_tensor = torch.randn(2, 2)
    st.write("Zeros Tensor:", zeros_tensor)
    st.write("Ones Tensor:", ones_tensor)
    st.write("Random Tensor:", rand_tensor)
    st.code("""
zeros_tensor = torch.zeros(2, 2)
ones_tensor  = torch.ones(2, 2)
rand_tensor  = torch.randn(2, 2)
""", language="python")

# ----------------------------------------------------------------------
# 6. Common Tensor Operators
with tabs[5]:
    st.header("6. Tensor Operators")
    st.markdown("""
    Tensor operators perform element-wise operations such as addition, subtraction, etc.
    """)
    # Define a and b as float tensors for consistency in later operations.
    a = torch.tensor([1, 2, 3], dtype=torch.float)
    b = torch.tensor([4, 5, 6], dtype=torch.float)
    sum_tensor = a + b
    product_tensor = a * b
    st.write("Tensor a:", a)
    st.write("Tensor b:", b)
    st.write("a + b:", sum_tensor)
    st.write("a * b:", product_tensor)
    st.code("""
a = torch.tensor([1, 2, 3], dtype=torch.float)
b = torch.tensor([4, 5, 6], dtype=torch.float)
sum_tensor = a + b      # Element-wise addition
product_tensor = a * b  # Element-wise multiplication
""", language="python")

# ----------------------------------------------------------------------
# 7. Dot Product
with tabs[6]:
    st.header("7. Dot Product")
    st.markdown("""
    The **dot product** multiplies two vectors and sums the results.
    """)
    dot_product = torch.dot(a, b)
    st.write("Dot product of a and b:", dot_product)
    st.code("""
dot_product = torch.dot(a, b)
""", language="python")

# ----------------------------------------------------------------------
# 8. Matrix-Vector Multiplication
with tabs[7]:
    st.header("8. Matrix-Vector Multiplication")
    st.markdown("""
    Multiply a matrix by a vector.
    """)
    matrix_mv = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float)
    vector_mv = torch.tensor([1, 0, -1], dtype=torch.float)
    result_mv = torch.mv(matrix_mv, vector_mv)
    st.write("Matrix:", matrix_mv)
    st.write("Vector:", vector_mv)
    st.write("Matrix-Vector Result:", result_mv)
    st.code("""
matrix = torch.tensor([[1, 2, 3],
                       [4, 5, 6]], dtype=torch.float)
vector = torch.tensor([1, 0, -1], dtype=torch.float)
result = torch.mv(matrix, vector)
""", language="python")

# ----------------------------------------------------------------------
# 9. Matrix Multiplication
with tabs[8]:
    st.header("9. Matrix Multiplication")
    st.markdown("""
    Multiply two matrices.  
    Note: The number of columns in the first must equal the number of rows in the second.
    """)
    matrix_a = torch.tensor([[1, 2], [3, 4]], dtype=torch.float)
    matrix_b = torch.tensor([[5, 6], [7, 8]], dtype=torch.float)
    matrix_product = matrix_a @ matrix_b
    st.write("Matrix A:", matrix_a)
    st.write("Matrix B:", matrix_b)
    st.write("A @ B:", matrix_product)
    st.code("""
matrix_a = torch.tensor([[1, 2],
                         [3, 4]], dtype=torch.float)
matrix_b = torch.tensor([[5, 6],
                         [7, 8]], dtype=torch.float)
matrix_product = matrix_a @ matrix_b
""", language="python")

# ----------------------------------------------------------------------
# 10. Norms
with tabs[9]:
    st.header("10. Norms")
    st.markdown("""
    **Norms** measure the size of a tensor.  
    The L2 norm (Euclidean norm) is commonly used.
    """)
    # Convert 'a' to float (if it wasn't already) to ensure proper computation.
    l2_norm = torch.norm(a, p=2)
    st.write("L2 Norm of a:", l2_norm)
    st.code("""
l2_norm = torch.norm(a, p=2)  # Ensure 'a' is a float tensor
""", language="python")

# ----------------------------------------------------------------------
# 11. Broadcasting Mechanism
with tabs[10]:
    st.header("11. Broadcasting")
    st.markdown("""
    **Broadcasting** automatically expands the dimensions of tensors during operations.  
    For example, adding a scalar to every element of a tensor.
    """)
    broadcast_example = tensor + 1  # Adds 1 to every element of 'tensor'
    st.write("Tensor + 1:", broadcast_example)
    st.code("""
broadcast_example = tensor + 1
""", language="python")

# ----------------------------------------------------------------------
# 12. Tensor Indexing and Slicing
with tabs[11]:
    st.header("12. Indexing & Slicing")
    st.markdown("""
    **Indexing and slicing** allow you to extract parts of a tensor.
    """)
    # For simplicity, use the previously created normalized tensor (3x3)
    sub_tensor = tensor_normalized[:2, :2]
    st.write("Sliced Tensor (first 2 rows and columns):", sub_tensor)
    st.code("""
sub_tensor = tensor_normalized[:2, :2]
""", language="python")

# ----------------------------------------------------------------------
# 13. Saving Memory
with tabs[12]:
    st.header("13. Saving Memory")
    st.markdown("""
    In-place operations modify tensors without creating new ones, saving memory.
    """)
    tensor_for_memory = tensor.clone()  # Clone to preserve the original tensor
    tensor_for_memory.add_(2)  # In-place addition: add 2 to each element
    st.write("Tensor after in-place addition (tensor.add_(2)):", tensor_for_memory)
    st.code("""
tensor_for_memory = tensor.clone()
tensor_for_memory.add_(2)  # In-place addition
""", language="python")

# ----------------------------------------------------------------------
# 14. Converting Tensors to Other Objects
with tabs[13]:
    st.header("14. Converting Tensors")
    st.markdown("""
    You can convert tensors to other objects, like NumPy arrays, for further use (e.g., plotting).
    """)
    tensor_np = tensor_normalized.numpy()
    st.write("Tensor converted to a NumPy array:", tensor_np)
    
    # Simple visualization using matplotlib
    fig, ax = plt.subplots()
    cax = ax.matshow(tensor_np, cmap='viridis')
    fig.colorbar(cax)
    st.pyplot(fig)
    
    st.code("""
tensor_np = tensor_normalized.numpy()
""", language="python")

st.markdown("---")
st.markdown("This app demonstrates simple examples of tensor operations using PyTorch. Each tab covers one topic with minimal code to help you understand the fundamentals. Enjoy exploring tensors!")
