import streamlit as st
import torch  # Replace with tensorflow if you prefer
import numpy as np
import matplotlib.pyplot as plt

# Title of the Streamlit app
st.title("Tensor Data Processing Demo")

# Section: Create a sample tensor
st.header("1. Creating a Tensor")
# Create a random tensor (e.g., a 2D tensor representing an image or a matrix)
tensor = torch.randn(10, 10)
st.write("Sample Tensor:", tensor)

# Section: Process the Tensor (e.g., normalization)
st.header("2. Processing the Tensor")
# Normalize the tensor between 0 and 1
tensor_min = torch.min(tensor)
tensor_max = torch.max(tensor)
tensor_normalized = (tensor - tensor_min) / (tensor_max - tensor_min)
st.write("Normalized Tensor:", tensor_normalized)

# Section: Visualization
st.header("3. Visualizing the Tensor")

# Convert tensor to numpy array for plotting
tensor_np = tensor_normalized.numpy()

fig, ax = plt.subplots()
cax = ax.matshow(tensor_np, cmap='viridis')
fig.colorbar(cax)
ax.set_title("Normalized Tensor Heatmap")
st.pyplot(fig)

# Additional interactive widget: Adjusting transformation parameters (example)
st.header("4. Interactive Transformation")
scale_factor = st.slider("Select a scale factor", min_value=0.1, max_value=5.0, value=1.0)
tensor_scaled = tensor_normalized * scale_factor
st.write("Scaled Tensor:", tensor_scaled)

# Plot the scaled tensor
tensor_scaled_np = tensor_scaled.numpy()
fig2, ax2 = plt.subplots()
cax2 = ax2.matshow(tensor_scaled_np, cmap='plasma')
fig2.colorbar(cax2)
ax2.set_title("Scaled Tensor Heatmap")
st.pyplot(fig2)
