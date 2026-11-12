# Method of moments tool for electromagnetic simulation

Method of moments is a mathematical method to compute electromagnetic simulation. Currently only conductive objects are allowed to use

Functions of library (sequential)
- `create_edge_tables(vertices, faces, num_of_handles)` (create edge tables from vertices and faces)
- `L_mn(vertices, faces, edges, edges_curr, frequency)` (create matrix EFIE $\mathbf{Z^{EJ}}$ and nMFIE $\mathbf{Z^{H}}$ )
- `V_m(vertices, faces, edges, edges_curr, k, frequency, polarization)`(create impulse for vector for edges based on polarized TEM wave with direction $k$ )
- `E_s(r, I_E, verticies, faces, edges, face_curr, frequency)`(Calculate electromagnetic field in positions r, given resulted current matrix I_E  )
  
Functions of library (parallel, CUDA used)
- `L_mn_cuda(vertices, faces, edges, edges_curr, frequency)`
- `L_mn_cuda_16_batches(vertices, faces, edges, edges_curr, frequency)`
- `E_s_cuda(r, I_E, verticies, faces, edges, face_curr, frequency)`

See the ```example.ipynb``` file to see how to use library. 

## How to create edge and face tables? 
1. Create object you want to simulate in blender and select it
2. Change G to number of handles (holes) in design - it affects the number of vertices and faces
3. Run script ```blender_vertices_faces.py``` from Blender Scripting tool

## What is polarization?
You are allowed to used six types of polarization (of electric field):
1. Pez (Perpendicular to k and z) $\frac{\mathbf{\hat{z}} \times \mathbf{\hat{k}}}{|\mathbf{\hat{z}} \times \mathbf{\hat{k}}|}$
2. Paz (Parallel to k and z) $\frac{\mathbf{\hat{k}}\times (\mathbf{\hat{z}} \times \mathbf{\hat{k}})}{|\mathbf{\hat{k}}\times (\mathbf{\hat{z}} \times \mathbf{\hat{k}})|}$
3. Pey (Perpendicular to k and y) $\frac{\mathbf{\hat{y}} \times \mathbf{\hat{k}}}{|\mathbf{\hat{y}} \times \mathbf{\hat{k}}|}$
4. Pay (Parallel to k and y) $\frac{\mathbf{\hat{k}}\times (\mathbf{\hat{y}} \times \mathbf{\hat{k}})}{|\mathbf{\hat{k}}\times (\mathbf{\hat{y}} \times \mathbf{\hat{k}})|}$
5. Pex (Perpendicular to k and x) $\frac{\mathbf{\hat{x}} \times \mathbf{\hat{k}}}{|\mathbf{\hat{x}} \times \mathbf{\hat{k}}|}$
5. Pax (Parallel to k and x) $\frac{\mathbf{\hat{k}}\times (\mathbf{\hat{x}} \times \mathbf{\hat{k}})}{|\mathbf{\hat{k}}\times (\mathbf{\hat{x}} \times \mathbf{\hat{k}})|}$

## Effects:
### Simulated:
<img width="200" height="200" alt="symulacja_MoM" src="https://github.com/user-attachments/assets/2121e890-fff2-4477-97e2-e8904fc276f4" />

### Example from book 'The Method of Moments in Electromagnetics'
<img width="200" height="200" alt="screen_z_MoM" src="https://github.com/user-attachments/assets/73cf7ce1-8d1d-48b2-9338-09be7580a9a6" />

### 3D:
<img width="200" height="200" alt="sphere_3D_3645_1_2_A" src="https://github.com/user-attachments/assets/bc2f6cc6-8930-41ed-8aa0-05943423c106" />

## Installation
1. You need cMake, nvcc compiler, >python3.9, pybind library
2. Compile python library (Ubuntu compatible)
3. Use compiled .so library '''ThreeD_MoMs.cpython-312-x86_64-linux-gnu.so''' as a python library

### Based on
Walton C. Gibson, The Method of Moments in Electromagnetics, Third Edition 2022
