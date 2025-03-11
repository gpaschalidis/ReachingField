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
import torch

from reachingfield.utils import read_o3d_mesh, create_o3d_mesh, define_plane, cast_rays,\
                     mask_rays, transform_points


class ReachingField:
    def __init__(self, inter_obj_path: str) -> None:
        self.inter_obj_path = inter_obj_path
        self.inter_obj_mesh = None
        self.inter_obj_verts = None
        self.inter_obj_faces = None
        self.inter_obj_height = None
        self.inter_obj_center = None
        self.posed_object = False 
        self.inter_data()
    

    def create_mesh_from_path(self, path: str):
        return read_o3d_mesh(path)


    def create_mesh_from_verts_and_faces(self, 
                                         verts: np.ndarray, 
                                         faces: np.ndarray,
                                         color: list = [1.0, 0.0, 0.0]):
        
        return create_o3d_mesh(verts, faces, color)


    def inter_data(self):
        self.inter_obj_mesh = self.create_mesh_from_path(self.inter_obj_path)
        self.inter_obj_verts = np.array(self.inter_obj_mesh.vertices)
        self.inter_obj_faces = np.array(self.inter_obj_mesh.triangles)
        self.inter_obj_center, self.inter_obj_height = self.define_obj_metadata(
                    self.inter_obj_verts
        )


    def define_obj_metadata(self, verts: np.ndarray):
        center = (verts.max(0) + verts.min(0)) / 2
        obj_dims = verts.max(0) - verts.min(0)
        obj_height = obj_dims[2]
        return center, obj_height

         

    def pose_obj_mesh(self, 
                      obj_transl: np.ndarray,
                      global_orient: np.ndarray):       
        self.inter_obj_verts = self.inter_obj_verts @ global_orient.T + obj_transl
        self.inter_obj_center, self.inter_obj_height = self.define_obj_metadata(
                    self.inter_obj_verts
        )
        self.inter_obj_mesh = create_o3d_mesh(
                self.inter_obj_verts, self.inter_obj_faces, [1,0,0]
        )
        self.posed_object = True

        
    def create_sphere_around_inter_obj(self):
        sphere = o3d.geometry.TriangleMesh.create_sphere(
            radius=2*self.inter_obj_height
        )
        sp_verts = np.array(sphere.vertices)
        target_points = sp_verts + self.inter_obj_center
        return target_points
   

    def respecify_target_points(self, target_points, origin, factor, projection=False):
        # We elongate the rays using a scaling factor
        dirs = target_points - origin
        if projection:
            normal_plane = define_plane(origin)
            dirs = dirs - (np.dot(dirs, normal_plane)[..., None] * normal_plane) / np.sqrt((normal_plane**2).sum())
            # In case there  is a vertical direction, its projection to the plane that
            # we define with the normal is going to be the zero vector. Therefore for such cases
            # we exclude these vectors and return the mask as well so we exclude the corresponding 
            # values from other arrays as well
            mask = ((dirs == np.zeros(3)).sum(1) != 3)
            dirs = dirs[mask]

        norm_dirs = dirs / (np.sqrt((dirs**2).sum(-1))[..., None])
        far_target_points = origin + factor * norm_dirs
        if projection:
            return far_target_points, norm_dirs, mask
        else:
            return far_target_points, norm_dirs

 
    def create_reachingfield(self, 
                             obj_transl=np.zeros(3), 
                             global_orient=np.eye(3),
                             obstacle_meshes: list = [],
                             grasp_type: str = "right"
    ):
        if not self.posed_object:
            self.pose_obj_mesh(obj_transl, global_orient)
        target_points = self.create_sphere_around_inter_obj()
        far_target_points, norm_dirs = self.respecify_target_points(
                target_points,
                self.inter_obj_center,
                factor=2,
                projection=False
        )
        
    
        rays_dict = {}
        rays_dict["first_step"] = {}
        rays_dict["first_step"]["rays_directions"] = norm_dirs

        # We cast rays from the center of the object towards norm_dirs
        results = cast_rays(
            obstacle_meshes, self.inter_obj_center[None], norm_dirs
        )

        # We store which rays intersect with the surrounding objects and 
        # which are not
        rays_dict["first_step"]["no_intersection"] = (
            results["t_hit"] == np.inf
        ).numpy()
        clean_mask = mask_rays(far_target_points, results)


        far_target_points = far_target_points[clean_mask]
        norm_dirs = norm_dirs[clean_mask]
        rays_dict["second_step"] = {}
        
        # These are the remaining rays after the first step
        rays_dict["second_step"]["rays_directions"] = norm_dirs

        normal_plane = np.array([0,0,1])

    
        # We project the remaining rays to the horizontal plane and check if 
        # these horizontal rays intersect with objects. hor_tar_p refers to 
        # the Horizontal Target Points, while hor_norm_dirs to the horizontal 
        # normalized directions that go from the object centroid to the Horizontal 
        # Target Points

        hor_tar_p, hor_norm_dirs, mask = self.respecify_target_points(
                far_target_points,
                self.inter_obj_center,
                factor=2,
                projection=True
        )
        
        far_target_points = far_target_points[mask]
        norm_dirs = norm_dirs[mask]
        
        rays_dict["third_step"] = {}

        rays_dict["third_step"]["rays_directions"] = hor_norm_dirs

        results = cast_rays(
            obstacle_meshes, self.inter_obj_center[None], hor_norm_dirs
        )
        
        rays_dict["third_step"]["no_intersection"] = (
            results["t_hit"] == np.inf
        ).numpy()
        
        clean_mask = mask_rays(hor_norm_dirs, results)
        
        # For the case, where all the horizontal rays intersect with the 
        #surrounding objects as when we have an object inside a drawer, 
        # we keep all rays
        if clean_mask.sum() != 0:
            hor_tar_p = hor_tar_p[clean_mask]
            hor_norm_dirs = hor_norm_dirs[clean_mask]
        
        remaining_dirs = norm_dirs if clean_mask.sum() == 0 else norm_dirs[clean_mask]
        
        far_target_points = far_target_points if clean_mask.sum() == 0 else far_target_points[clean_mask]

        dis = 0.6
        d = np.linspace(dis, 2, 5).reshape(5, 1, 1)
        new_origin_points = self.inter_obj_center + (d * hor_norm_dirs[None]).transpose(1,0,2)
        target_points = new_origin_points.copy()
        target_points[:, :, 2] = -1

        target_points, norm_dirs = self.respecify_target_points(
                target_points, new_origin_points, 2
        )
        
        target_z = target_points[:, :, 2].mean()

        if grasp_type == "right":
            rot_angles = np.linspace(0, np.pi/6, 10)
        else:
            rot_angles = np.linspace(0, -np.pi/6, 10)


        cos_rot_angles = np.cos(rot_angles)
        sin_rot_angles = np.sin(rot_angles)

        new_origin_points = transform_points(new_origin_points, 
                                             self.inter_obj_center,
                                             cos_rot_angles,
                                             sin_rot_angles
        )


        target_points = new_origin_points.copy()
        target_points[:, :, :, 2] = target_z

        rot_points = hor_tar_p.copy()
        rot_points = transform_points(rot_points, 
                                      self.inter_obj_center,
                                      cos_rot_angles,
                                      sin_rot_angles
        )
        rotating_rays = {}
        hor_rays_results = []
        for i in range(10):
            rotating_rays["step_{}".format(i)] = {}

            horizontal_directions = rot_points[:, i, :] - self.inter_obj_center[None]
            horizontal_directions = horizontal_directions / np.sqrt((horizontal_directions**2).sum(1))[...,None]
            rotating_rays["step_{}".format(i)]["horizontal_directions"] = horizontal_directions

            results = (cast_rays(obstacle_meshes, 
                                  self.inter_obj_center[None], 
                                  horizontal_directions)["t_hit"] == np.inf).numpy()
            
            hor_rays_results.append(results)
            rotating_rays["step_{}".format(i)]["start_points"] = new_origin_points[:, :, i,:]
            rotating_rays["step_{}".format(i)]["target_points"] = target_points[:, :, i, :]

        hor_rays_results = np.array(hor_rays_results).transpose(1,0).reshape(-1, 1, 10)
        
        if clean_mask.sum() == 0:
            # This is for the case where all the projected rays are discarded
            # (eg. an object inside a drawer). While we rotate those horizontal rays 
            # all are discarded for the same reason. That's why we keep all of them
            hor_rays_results = ~hor_rays_results       

        norm_dirs = np.repeat(norm_dirs[...,None],10,3).transpose(0,1,3,2)
        results = cast_rays(obstacle_meshes, new_origin_points.reshape(-1,3) , norm_dirs.reshape(-1,3), multiple_origins=True)
        clean_mask = mask_rays(norm_dirs.reshape(-1,3), results).reshape(-1, 5, 10)

        # We combine the filtering from the vertical and the horizontal rays with the
        # following command
        clean_mask = clean_mask * hor_rays_results
        rotating_rays["no_intersection"] = (clean_mask.reshape(-1, 5, 10).sum(1) == 5)

        clean_mask = clean_mask.reshape(-1, 50)
        final_mask = clean_mask.sum(1) == 50


        final_clean_points = hor_tar_p[final_mask]
        final_proj_dirs = hor_norm_dirs[final_mask]    
        final_proj_dirs = final_proj_dirs / np.sqrt((final_proj_dirs**2).sum(1))[...,None]
        new_origin_points = new_origin_points[final_mask]
        target_points = target_points[final_mask]


        final_dirs = remaining_dirs[final_mask]
        final_init_points = far_target_points[final_mask]

        rays_dict["final_step"] = {} 
        rays_dict["final_step"]["rays_directions"] = final_dirs
        rays_dict["final_step"]["horizontal_rays"] = final_proj_dirs
        
        return rays_dict, rotating_rays 
       
 
    def sample(self, 
               obj_transl=np.zeros(3), 
               global_orient=np.eye(3),
               obstacle_meshes: list = [],
               grasp_type: str = "right",
               num_samples: int = 1):

        rays_dict, _ = self.create_reachingfield(
             obj_transl, global_orient, obstacle_meshes, grasp_type)

        final_dirs = rays_dict["final_step"]["rays_directions"]
        final_hor_dirs = rays_dict["final_step"]["horizontal_rays"]
        
        # We calculate first the angle of the remaining rays with the +z axis
        angles1 = np.arccos(np.dot(final_dirs, np.array([0, 0, 1])))
        
        # We do the same for the -z axis
        angles2 = np.arccos(np.dot(final_dirs,np.array([0, 0, 1])*(-1)))
        
        # The following mask tells us which rays are directed upwards
        mask = angles1 < angles2

        # For each ray we keep then the smallest of the two angles
        final_angles = np.min((angles1,angles2),0)
        
        # We use a treshold T = 0.7
        if self.inter_obj_center[2] < 0.7:
            final_angles[mask] *= -1
        else:
            final_angles[~mask] *= -1  

        probs = np.exp(-1/final_angles)
        probs = torch.tensor(probs / probs.sum())
        ray_index = torch.multinomial(probs, num_samples, replacement=True)
        
        ray_dirs = np.atleast_2d(final_dirs[ray_index])
        hor_ray_dirs = np.atleast_2d(final_hor_dirs[ray_index])
        return ray_dirs, hor_ray_dirs
         
