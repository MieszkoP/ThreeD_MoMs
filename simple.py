import ThreeD_MoMs
import pickle
with open("faces_simple.pkl", "rb") as file:
    faces = pickle.load(file)
with open("vert_simple.pkl", "rb") as file:
    vertices = pickle.load(file)
fn, fn_pm = ThreeD_MoMs.create_edge_tables(vertices, faces, 0)
ThreeD_MoMs.L_mn(vertices, faces, fn, fn_pm, 3.5e9)
