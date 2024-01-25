import open3d as o3d
import numpy as np
import torch
import os
import matplotlib.pyplot as plt
from sklearn.neighbors import KDTree
import json
import trimesh

def read_txt_obj_ls(path, ret_mesh=False, use_tri_mesh=False):
    # In some cases, trimesh can change vertex order
    if use_tri_mesh:
        tri_mesh_loaded_mesh = trimesh.load_mesh(path, process=False)
        vertex_ls = np.array(tri_mesh_loaded_mesh.vertices)
        tri_ls = np.array(tri_mesh_loaded_mesh.faces)+1
    else:
        f = open(path, 'r')
        vertex_ls = []
        tri_ls = []
        #vertex_color_ls = []
        while True:
            line = f.readline().split()
            if not line: break
            if line[0]=='v':
                vertex_ls.append(list(map(float,line[1:4])))
                #vertex_color_ls.append(list(map(float,line[4:7])))
            elif line[0]=='f':
                tri_verts_idxes = list(map(str,line[1:4]))
                if "//" in tri_verts_idxes[0]:
                    for i in range(len(tri_verts_idxes)):
                        tri_verts_idxes[i] = tri_verts_idxes[i].split("//")[0]
                tri_verts_idxes = list(map(int, tri_verts_idxes))
                tri_ls.append(tri_verts_idxes)
            else:
                continue
        f.close()

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertex_ls)
    mesh.triangles = o3d.utility.Vector3iVector(np.array(tri_ls)-1)
    mesh.compute_vertex_normals()

    norms = np.array(mesh.vertex_normals)

    vertex_ls = np.array(vertex_ls)
    output = [np.concatenate([vertex_ls,norms], axis=1)]

    if ret_mesh:
        output.append(mesh)
    return output

def get_colored_mesh(mesh, label_arr):
    palte = np.array([
        [255,153,153],

        [153,76,0],
        [153,153,0],
        [76,153,0],
        [0,153,153],
        [0,0,153],
        [153,0,153],
        [153,0,76],
        [64,64,64],

        [255,128,0],
        [153,153,0],
        [76,153,0],
        [0,153,153],
        [0,0,153],
        [153,0,153],
        [153,0,76],
        [64,64,64],
    ])/255
    palte[9:] *= 0.4
    label_arr = label_arr.copy()
    label_arr %= palte.shape[0]
    label_colors = np.zeros((label_arr.shape[0], 3))
    for idx, palte_color in enumerate(palte):
        label_colors[label_arr==idx] = palte[idx]
    mesh.vertex_colors = o3d.utility.Vector3dVector(label_colors)
    return mesh

def load_json(file_path):
    with open(file_path, "r") as st_json:
        return json.load(st_json)

def np_to_pcd(arr, color=[1,0,0]):
    arr = np.array(arr)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(arr[:,:3])
    if arr.shape[1] >= 6:
        pcd.normals = o3d.utility.Vector3dVector(arr[:,3:6])
    pcd.colors = o3d.utility.Vector3dVector([color]*len(pcd.points))
    return pcd
def print_3d(*data_3d_ls):
    data_3d_ls = [item for item in data_3d_ls]
    for idx, item in enumerate(data_3d_ls):
        if type(item) == np.ndarray:
            data_3d_ls[idx] = np_to_pcd(item)
    o3d.visualization.draw_geometries(data_3d_ls, mesh_show_wireframe = True, mesh_show_back_face = True)