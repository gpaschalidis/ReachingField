# -*- coding: utf-8 -*-
#
# Copyright (C) 2022 Universiteit van Amsterdam (UvA).
# All rights reserved.
#
# Universiteit van Amsterdam (UvA) is holder of all proprietary rights
# on this computer program. You can only use this computer program if you have closed a license agreement
# with UvA or you get the right to use the computer program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and liable to prosecution.
# Contact: g.paschalidis@uva.nl
#


import numpy as np
import open3d as o3d
import os


def transform_points(points, origin, cos_rot_angles, sin_rot_angles):
    points = points - origin    
    points_x = points[...,0][...,None]
    points_y = points[...,1][...,None]
    points_z = np.repeat(points[...,2][..., None],10,-1)
        
    p_x = points_x * cos_rot_angles[None] + points_y * sin_rot_angles[None]
    p_y = points_y * cos_rot_angles[None] - points_x * sin_rot_angles[None]
    points = np.concatenate((
        p_x[...,None],
        p_y[...,None],
        points_z[...,None]
    ), axis=-1)

    points = points + origin
    return points  


def visualize_rays_after_scanning(clean_mask, i,new_origins,targets,new_clean_points, obj_center,obj_mesh,rec_mesh,mesh_box):
    clean_mask = clean_mask.reshape(-1, 5,10)
    clean_mask = clean_mask[:,:,0:i+1].reshape(-1,5*(i+1))
    final_mask = clean_mask.sum(1) == 5*(i+1)
    final_clean_points = new_clean_points[final_mask]
    #new_origins = new_origins[:,:,0:i+1,:][final_mask]
    new_origins = new_origins[final_mask]
    #targets = targets[:,:,0:i+1,:][final_mask]
    targets = targets[final_mask]
    line_set_new = create_line_set(new_origins.reshape(-1,3), targets.reshape(-1,3), [0.5, 0.1, 0.2], multiple_origins=True)
    line_set_new_4 = create_line_set(obj_center[None], final_clean_points, [1, 0, 1])
    o3d.visualization.draw_geometries([rec_mesh, obj_mesh, mesh_box, line_set_new,line_set_new_4])


def define_obj_key(obj_name, list_per_obj, rec_key):
    comb = list_per_obj[0]
    comb_parts = comb.split("_")
    obj_where = "_".join((comb_parts[1],comb_parts[2]))
    obj_key = "_".join((obj_name, rec_key, obj_where))
    return obj_key


def sphere_around_obj(obj_height, obj_center):
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=2*obj_height)
    sp_verts = np.array(sphere.vertices)
    sp_verts = sp_verts + obj_center
    return sp_verts

    
def read_o3d_mesh(path):
    return o3d.io.read_triangle_mesh(path)


def create_o3d_mesh(verts, faces, color):
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    mesh.paint_uniform_color(color)
    mesh.compute_vertex_normals()
    return mesh


def create_o3d_box_mesh(rec_verts):
    mesh_box = o3d.geometry.TriangleMesh.create_box(width=4,height=4,depth=0.005)
    mesh_box_verts = np.array(mesh_box.vertices)
    mesh_box_center = (mesh_box_verts.max(0)+mesh_box_verts.min(0))/2
    mesh_box_verts -= mesh_box_center
    mesh_box_verts[:,0] += rec_verts.mean(0)[0]
    mesh_box_verts[:,1] += rec_verts.mean(0)[1]
    mesh_box.vertices = o3d.utility.Vector3dVector(mesh_box_verts)
    mesh_box.compute_vertex_normals()
    mesh_box.paint_uniform_color([0.2, 0.2, 0.2])
    return mesh_box


def create_o3d_box_mesh_vertical_y(rec_verts):
    mesh_box = o3d.geometry.TriangleMesh.create_box(width=4,height=4,depth=0.005)
    mesh_box_verts = np.array(mesh_box.vertices)
    mesh_box_center = (mesh_box_verts.max(0)+mesh_box_verts.min(0))/2
    mesh_box_verts -= mesh_box_center
    Rx = np.eye(3)
    Rx[1][1] = 0 
    Rx[2][2] = 0 
    Rx[1][2] = -1
    Rx[2][1] = 1 
    mesh_box_verts = (Rx.T @ mesh_box_verts.T).T

    mesh_box_verts[:,0] += rec_verts.mean(0)[0]
    mesh_box_verts[:,2] += rec_verts.mean(0)[2]
    mesh_box.vertices = o3d.utility.Vector3dVector(mesh_box_verts)
    mesh_box.compute_vertex_normals()
    mesh_box.paint_uniform_color([0.2, 0.2, 0.2])
    return mesh_box

def make_background_white(o3d_image):
    o3d_image = np.array(o3d_image)
    mask = o3d_image >= 234
    o3d_image[mask] = 255
    o3d_image = o3d.geometry.Image(o3d_image)
    return o3d_image


def create_point_cloud(points):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.paint_uniform_color(list(np.random.rand(1,3)[0]))
    return pcd 


def define_plane(center):
    point1 = np.array([0, 1, center[2]])
    point2 = np.array([1, 2, center[2]])

    vector1 = point1 - center
    vector2 = point2 - center

    vector1 = vector1 / np.sqrt((vector1**2).sum())
    vector2 = vector2 / np.sqrt((vector2**2).sum())

    normal_plane = np.cross(vector1, vector2)
    return normal_plane / np.sqrt((normal_plane**2).sum())


def create_line_set(origin, target_points, color, multiple_origins=False):
    points = np.concatenate((origin, target_points), 0)
    step = len(target_points) if multiple_origins else 1
    u = 1 if multiple_origins else 0
    lines = [[idp*u, idp+step] for idp in range(len(target_points))]
    colors = [color for k in range(len(lines))]
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)
    return line_set


def cast_rays(obstacle_mesh_list, origin, norm_dirs, multiple_origins=False):
    scene = o3d.t.geometry.RaycastingScene()
    obstacles = [
        o3d.t.geometry.TriangleMesh.from_legacy(mesh) for mesh in obstacle_mesh_list
    ]
    obstacles_id = [scene.add_triangles(ob) for ob in obstacles]
    u = 1 if multiple_origins else 0
    ray_list = [list(origin[jd*u]) + list(norm_dirs[jd]) for jd in range(len(norm_dirs))]
    rays = o3d.core.Tensor(ray_list, dtype=o3d.core.Dtype.Float32)
    results = scene.cast_rays(rays)
    return results


def mask_rays(far_target_points, rays_results):
    inter_dists = np.array([
        rays_results["t_hit"][ju].item() for ju in range(len(far_target_points))
    ])
    clean_mask = inter_dists == np.inf
    return clean_mask


def read_params(params):
    return params[0], params[1]



