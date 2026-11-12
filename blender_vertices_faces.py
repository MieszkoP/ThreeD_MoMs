import bpy
import numpy as np
import pickle
# Aktywny obiekt
obj = bpy.context.active_object

# Sprawdzenie czy to mesh
from mathutils import Matrix

if obj and obj.type == 'MESH':

    mesh = obj.data
    scale_matrix = obj.matrix_world.to_3x3()
    normal_matrix = scale_matrix.inverted().transposed()

    # =========================
    # WIERZCHOŁKI
    # =========================

    print("=== VERTICES ===")
    vertices = np.zeros((1, 3))
    faces = np.zeros((1,7))
    for vertex in mesh.vertices:
        print(
            vertex.index,
            vertex.co.x,
            vertex.co.y,
            vertex.co.z
        )
        global_co = obj.matrix_world @ vertex.co
        vertices = np.vstack((vertices, [global_co.x, global_co.y, global_co.z]))

    # =========================
    # ŚCIANY
    # =========================

    print("\n=== FACES ===")
    
    for face in mesh.polygons:
        #print(face.normal)
        area =  face.area * scale_matrix.to_3x3().determinant() ** (2/3)
        normal = (normal_matrix @ face.normal).normalized()
        #print("area: "+ str(area))
        faces = np.vstack((faces, [face.vertices[0]+1, face.vertices[1]+1,
         face.vertices[2]+1, area, normal.x, normal.y, normal.z]))
         
    with open("faces_simple.pkl", 'wb') as f: ## YOU NEED TO SET YOUR FILENAME
        pickle.dump(faces, f)
        
    with open("vert_simple.pkl", 'wb') as f:  ## YOU NEED TO SET YOUR FILENAME
        pickle.dump(vertices,f)