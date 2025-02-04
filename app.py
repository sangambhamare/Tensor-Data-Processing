import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Tensor Data Processing", layout="wide")
st.title("Tensor Data Processing Teaching App")

# Create tabs for each topic
tabs = st.tabs([
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
])

# -----------------------------------------------
# 1. Data Manipulation
with tabs[0]:
    st.header("1. Data Manipulation")
    st.markdown("""
    **What It Is:**  
    Data manipulation involves transforming, reshaping, or modifying data so that it can be used effectively.
    """)
    
    st.markdown("**Example:** Creating a tensor and normalizing it between 0 and 1.")
    # Create a random tensor
    tensor = torch.randn(10, 10)
    st.write("Original Tensor:", tensor)
    
    # Normalize the tensor
    tensor_min = torch.min(tensor)
    tensor_max = torch.max(tensor)
    tensor_normalized = (tensor - tensor_min) / (tensor_max - tensor_min)
    st.write("Normalized Tensor:", tensor_normalized)
    
    st.code("""
# Create a random tensor
tensor = torch.randn(10, 10)
# Normalize the tensor between 0 and 1
tensor_min = torch.min(tensor)
tensor_max = torch.max(tensor)
tensor_normalized = (tensor - tensor_min) / (tensor_max - tensor_min)
""", language="python")

# -----------------------------------------------
# 2. Vector
with tabs[1]:
    st.header("2. Vector")
    st.markdown("""
    **What It Is:**  
    A vector is a 1-dimensional tensor, essentially a list of numbers.
    """)
    
    vector = torch.tensor([1.0, 2.0, 3.0])
    st.write("Vector:", vector)
    
    st.code("""
# Create a vector (1-D tensor)
vector = torch.tensor([1.0, 2.0, 3.0])
""", language="python")

# -----------------------------------------------
# 3. Matrices
with tabs[2]:
    st.header("3. Matrices")
    st.markdown("""
    **What It Is:**  
    A matrix is a 2-dimensional tensor, similar to a table or grid of numbers.
    """)
    
    matrix = torch.tensor([[1, 2, 3], [4, 5, 6]])
    st.write("Matrix:", matrix)
    
    st.code("""
# Create a matrix (2-D tensor)
matrix = torch.tensor([[1, 2, 3], [4, 5, 6]])
""", language="python")

# -----------------------------------------------
# 4. Tensors
with tabs[3]:
    st.header("4. Tensors")
    st.markdown("""
    **What It Is:**  
    A tensor is a generalization of scalars, vectors, and matrices to higher dimensions.
    - A scalar is a 0-D tensor.
    - A vector is a 1-D tensor.
    - A matrix is a 2-D tensor.
    - Higher-dimensional tensors (e.g., 3-D, 4-D) are used for more complex data.
    """)
    
    tensor_3d = torch.randn(3, 4, 5)
    st.write("Example of a 3-D Tensor:", tensor_3d)
    
    st.code("""
# Create a 3-D tensor
tensor_3d = torch.randn(3, 4, 5)
""", language="python")

# -----------------------------------------------
# 5. Commonly-used Tensor Constructors
with tabs[4]:
    st.header("5. Commonly-used Tensor Constructors")
    st.markdown("""
    **What They Are:**  
    Functions to create tensors such as:
    - `torch.zeros(shape)`: Tensor filled with zeros.
    - `torch.ones(shape)`: Tensor filled with ones.
    - `torch.randn(shape)`: Tensor with random numbers from a normal distribution.
    - `torch.arange(start, end)`: Tensor with a range of numbers.
    """)
    
    zeros_tensor = torch.zeros(3, 3)
    ones_tensor = torch.ones(3, 3)
    rand_tensor = torch.randn(3, 3)
    st.write("Zeros Tensor:", zeros_tensor)
    st.write("Ones Tensor:", ones_tensor)
    st.write("Random Tensor:", rand_tensor)
    
    st.code("""
zeros_tensor = torch.zeros(3, 3)
ones_tensor = torch.ones(3, 3)
rand_tensor = torch.randn(3, 3)
""", language="python")

# -----------------------------------------------
# 6. Common Tensor Operators
with tabs[5]:
    st.header("6. Common Tensor Operators")
    st.markdown("""
    **What They Are:**  
    Element-wise operations such as addition, subtraction, multiplication, and division.
    """)
    
    a = torch.tensor([1, 2, 3])
    b = torch.tensor([4, 5, 6])
    sum_tensor = a + b
    product_tensor = a * b
    st.write("Tensor a:", a)
    st.write("Tensor b:", b)
    st.write("Sum (a + b):", sum_tensor)
    st.write("Product (a * b):", product_tensor)
    
    st.code("""
a = torch.tensor([1, 2, 3])
b = torch.tensor([4, 5, 6])
sum_tensor = a + b      # Element-wise addition
product_tensor = a * b  # Element-wise multiplication
""", language="python")

# -----------------------------------------------
# 7. Dot Product
with tabs[6]:
    st.header("7. Dot Product")
    st.markdown("""
    **What It Is:**  
    The dot product takes two equal-length vectors and returns a single number—a measure of their similarity.
    """)
    
    dot_product = torch.dot(a, b)
    st.write("Dot Product of a and b:", dot_product)
    
    st.code("""
dot_product = torch.dot(a, b)
""", language="python")

# -----------------------------------------------
# 8. Matrix-Vector Multiplication
with tabs[7]:
    st.header("8. Matrix-Vector Multiplication")
    st.markdown("""
    **What It Is:**  
    Multiplying a matrix by a vector. Often used in linear transformations.
    """)
    
    # Create a matrix (2x3) and a vector (3 elements)
    matrix_mv = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float)
    vector_mv = torch.tensor([1, 0, -1], dtype=torch.float)
    result_mv = torch.mv(matrix_mv, vector_mv)
    st.write("Matrix:", matrix_mv)
    st.write("Vector:", vector_mv)
    st.write("Matrix-Vector Multiplication Result:", result_mv)
    
    st.code("""
matrix = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float)
vector = torch.tensor([1, 0, -1], dtype=torch.float)
result = torch.mv(matrix, vector)
""", language="python")

# -----------------------------------------------
# 9. Matrix Multiplication
with tabs[8]:
    st.header("9. Matrix Multiplication")
    st.markdown("""
    **What It Is:**  
    Multiplying two matrices. The number of columns in the first must match the number of rows in the second.
    """)
    
    matrix_a = torch.tensor([[1, 2], [3, 4]], dtype=torch.float)
    matrix_b = torch.tensor([[5, 6], [7, 8]], dtype=torch.float)
    matrix_product = matrix_a @ matrix_b  # or torch.matmul(matrix_a, matrix_b)
    st.write("Matrix A:", matrix_a)
    st.write("Matrix B:", matrix_b)
    st.write("Matrix Multiplication (A @ B):", matrix_product)
    
    st.code("""
matrix_a = torch.tensor([[1, 2], [3, 4]], dtype=torch.float)
matrix_b = torch.tensor([[5, 6], [7, 8]], dtype=torch.float)
matrix_product = matrix_a @ matrix_b
""", language="python")

# -----------------------------------------------
# 10. Norms
with tabs[9]:
    st.header("10. Norms")
    st.markdown("""
    **What They Are:**  
    Norms measure the “length” or “size” of vectors and tensors. The L2 norm (Euclidean norm) is very common.
    """)
    
    l2_norm = torch.norm(a, p=2)
    st.write("L2 Norm of vector a:", l2_norm)
    
    st.code("""
l2_norm = torch.norm(a, p=2)
""", language="python")

# -----------------------------------------------
# 11. Broadcasting Mechanism
with tabs[10]:
    st.header("11. Broadcasting Mechanism")
    st.markdown("""
    **What It Is:**  
    Broadcasting allows operations on tensors of different shapes by automatically expanding smaller tensors.
    """)
    
    broadcast_example = tensor_normalized + 0.5
    st.write("Tensor after Broadcasting Addition (tensor_normalized + 0.5):", broadcast_example)
    
    st.code("""
broadcast_example = tensor_normalized + 0.5
""", language="python")

# -----------------------------------------------
# 12. Tensor Indexing and Slicing
with tabs[11]:
    st.header("12. Tensor Indexing and Slicing")
    st.markdown("""
    **What It Is:**  
    Extracting or modifying parts of a tensor by indexing and slicing.
    """)
    
    sub_tensor = tensor_normalized[:5, :5]
    st.write("Sliced Tensor (first 5 rows and columns):", sub_tensor)
    
    st.code("""
sub_tensor = tensor_normalized[:5, :5]
""", language="python")

# -----------------------------------------------
# 13. Saving Memory
with tabs[12]:
    st.header("13. Saving Memory")
    st.markdown("""
    **What It Is:**  
    Using memory-saving techniques like in-place operations (operations that modify data without creating a new tensor).  
    In PyTorch, in-place operations have an underscore suffix (e.g., `add_()`).
    """)
    
    # In-place addition: adds 1 to every element without creating a new tensor
    tensor_copy = tensor.clone()  # create a clone to preserve the original tensor for display
    tensor_copy.add_(1)
    st.write("Tensor after in-place addition (tensor.add_(1)):", tensor_copy)
    
    st.code("""
# In-place operation example:
tensor.add_(1)
""", language="python")

# -----------------------------------------------
# 14. Converting Tensors to Other Objects
with tabs[13]:
    st.header("14. Converting Tensors to Other Objects")
    st.markdown("""
    **What It Is:**  
    Converting tensors to other formats (e.g., NumPy arrays) is often necessary for visualization or interoperability with other libraries.
    """)
    
    tensor_np = tensor_normalized.numpy()
    st.write("Tensor converted to a NumPy array:", tensor_np)
    
    # Show a simple visualization using matplotlib
    fig, ax = plt.subplots()
    cax = ax.matshow(tensor_np, cmap='viridis')
    fig.colorbar(cax)
    st.pyplot(fig)
    
    st.code("""
# Convert a tensor to a NumPy array for visualization:
tensor_np = tensor_normalized.numpy()
""", language="python")

st.markdown("---")
st.markdown("This app demonstrates various tensor operations using PyTorch, with interactive examples organized in separate tabs. Each tab provides a simple explanation along with example code, so you can see how these concepts are implemented in practice.")

