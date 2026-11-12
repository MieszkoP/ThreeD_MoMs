# Method of moments tool for electromagnetic simulation

The method of moments is a numerical method for electromagnetic simulation. Currently, only conductive objects are supported.

## Library functions (sequential)
- `create_edge_tables(vertices, faces, num_of_handles)` – creates edge tables from vertices and faces.
- `L_mn(vertices, faces, edges, face_curr, frequency)` – computes the EFIE matrix $\mathbf{Z^{EJ}}$ and the nMFIE matrix $\mathbf{Z^{H}}$.
- `V_m(vertices, faces, edges, face_curr, k, frequency, polarization)` – computes the excitation vector for the edges, based on a polarized TEM plane wave propagating in direction $k$.
- `E_s(r, I_E, vertices, faces, edges, face_curr, frequency)` – computes the scattered electric field at points `r`, given the current vector `I_E`.
  
## Library functions (parallel, CUDA used)
- `L_mn_cuda(vertices, faces, edges, edges_curr, frequency)`
- `L_mn_cuda_16_batches(vertices, faces, edges, edges_curr, frequency)`
- `E_s_cuda(r, I_E, verticies, faces, edges, face_curr, frequency)`

See `example.ipynb` for how to use the library.

## How to create edge and face tables?

1. Create a 3D object in Blender. The object must be (a) composed only of triangles and (b) closed, i.e. every edge belongs to exactly two faces.
2. Select it.
3. Open `blender_vertices_faces.py` in Blender's Scripting tab.
4. Set `G` to the number of handles (the genus) of the design; together with the number of vertices it determines the number of edges.
5. Set the output paths of the `.pkl` files.
6. Run `blender_vertices_faces.py` from Blender's Scripting tab.

## What is polarization?
You can choose one of six polarizations of the electric field:
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

1. You need CMake (≥ 3.18), a C++20 compiler, the `nvcc` compiler, Python ≥ 3.9, and pybind11.
2. Build the Python module (tested on Ubuntu).
3. Import the compiled `ThreeD_MoMs.cpython-312-x86_64-linux-gnu.so` as a Python module.

## Based on

Gibson, W. C., *The Method of Moments in Electromagnetics*, 3rd ed., 2022.
