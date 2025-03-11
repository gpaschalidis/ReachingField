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

import argparse
import os
import numpy as np
import open3d as o3d

from reachingfield.reachingfield import ReachingField
from reachingfield.utils import create_o3d_mesh, create_o3d_box_mesh,\
        create_line_set 


def main():
    parser = argparse.ArgumentParser(
        description="Create a ReachingField for a given object"
    )   

    parser.add_argument(
        "--obj_rec_conf",
        required=True,
        help="The name of the receptacle configuration that is going to be used."
    )   
    parser.add_argument(
        "--grasp_type",
        default="right",
        choices=["right", "left"],
        help="The type of grasp which will be used to calculate the ReachingField."
    )   
    parser.add_argument(
        "--vis",
        action="store_true",
        help="The type of grasp which will be used to calculate the ReachingField."
    )   
    parser.add_argument(
        "--save_path",
        default=None,
        help="The path where the reachingfield will be stored"
    )   

    args = parser.parse_args()
    obj_rec_conf = args.obj_rec_conf


    if args.save_path:
        os.makedirs(args.save_path, exist_ok=True)


    obj_name = obj_rec_conf.split("_")[0]
    rec_name = "_".join(obj_rec_conf.split("_")[1:-2])
    inter_obj_path = os.path.join(os.getcwd(),"data/contact_meshes", "{}.ply".format(obj_name))
    metadata_path = os.path.join(os.getcwd(),"data/replicagrasp", "dset_info.npz")
    metadata = dict(np.load(metadata_path, allow_pickle=True))[obj_rec_conf]
    obj_transl = metadata[0]
    global_orient = metadata[1]

    rec_data_path = os.path.join(os.getcwd(),"data/replicagrasp", "receptacles.npz")
    rec_data = dict(np.load(rec_data_path, allow_pickle=True))[rec_name]
    rec_verts = rec_data[0][0]
    rec_faces = rec_data[0][1]
    
    rec_mesh = create_o3d_mesh(rec_verts, rec_faces, (0.3,0.1,0.5))
    floor_mesh = create_o3d_box_mesh(rec_verts)

    reachingfield = ReachingField(inter_obj_path)
    rays_dict, rotating_rays = reachingfield.create_reachingfield(
            obj_transl, global_orient, [rec_mesh], grasp_type=args.grasp_type
    )
    
    if args.save_path:
        np.savez(args.save_path,
                 rays_dict=rays_dict,
                 rotating_rays=rotating_rays
        )


    if args.vis:
        o3d.visualization.draw_geometries(
            [rec_mesh, floor_mesh, reachingfield.inter_obj_mesh]
        )
       
        ### First step 
        first_step_rays = create_line_set(
            reachingfield.inter_obj_center[None],
            reachingfield.inter_obj_center + 2 * rays_dict["first_step"]["rays_directions"],
            [0, 1, 0],
            multiple_origins=False
        )
        
        o3d.visualization.draw_geometries(
            [rec_mesh, floor_mesh, reachingfield.inter_obj_mesh, first_step_rays]
        )

        not_inter_mask = rays_dict["first_step"]["no_intersection"]
        
        # We visualize with green color the rays that do not intersect and with red
        # those that intersect.
        
        combined_first = []
        
        if not_inter_mask.sum() != 0:
            first_no_inter = create_line_set(
                reachingfield.inter_obj_center[None],
                reachingfield.inter_obj_center + 2 * rays_dict["first_step"]["rays_directions"][not_inter_mask],
                [0, 1, 0],
                multiple_origins=False
            )
            combined_first.append(first_no_inter)
        
        if (~not_inter_mask.sum()) != 0:
            first_inter = create_line_set(
                reachingfield.inter_obj_center[None],
                reachingfield.inter_obj_center + 2 * rays_dict["first_step"]["rays_directions"][~not_inter_mask],
                [1, 0, 0],
                multiple_origins=False
            )
            combined_first.append(first_inter)

        o3d.visualization.draw_geometries(
            [rec_mesh, floor_mesh, reachingfield.inter_obj_mesh] + combined_first
        )

        
        ### Second step 
        second_step_rays = create_line_set(
            reachingfield.inter_obj_center[None],
            reachingfield.inter_obj_center + 2 * rays_dict["second_step"]["rays_directions"],
            [0, 1, 0],
            multiple_origins=False
        )
        
        o3d.visualization.draw_geometries(
            [rec_mesh, floor_mesh, reachingfield.inter_obj_mesh, second_step_rays]
        )
        
        ### Third step 
        third_step_rays = create_line_set(
            reachingfield.inter_obj_center[None],
            reachingfield.inter_obj_center + 2 * rays_dict["third_step"]["rays_directions"],
            [0, 1, 0],
            multiple_origins=False
        )
        
        o3d.visualization.draw_geometries(
            [rec_mesh, floor_mesh, reachingfield.inter_obj_mesh, third_step_rays]
        )

        not_inter_mask = rays_dict["third_step"]["no_intersection"]
        
        combined_third = []
        if not_inter_mask.sum() != 0:
            third_no_inter = create_line_set(
                reachingfield.inter_obj_center[None],
                reachingfield.inter_obj_center + 2 * rays_dict["third_step"]["rays_directions"][not_inter_mask],
                [0, 1, 0],
                multiple_origins=False
            )
            combined_third.append(third_no_inter)
        
        if (~not_inter_mask).sum() != 0:
            third_inter = create_line_set(
                reachingfield.inter_obj_center[None],
                reachingfield.inter_obj_center + 2 * rays_dict["third_step"]["rays_directions"][~not_inter_mask],
                [1, 0, 0],
                multiple_origins=False
            )
            combined_third.append(third_inter)

        o3d.visualization.draw_geometries(
            [rec_mesh, floor_mesh, reachingfield.inter_obj_mesh] + combined_third
        )
        
        ### Rotating steps
        rot_not_inter = rotating_rays["no_intersection"] 
        for i in range(10):
            step = "step_{}".format(i)
            horizontal_directions = rotating_rays[step]["horizontal_directions"]
        
            not_inter_mask = rot_not_inter[:, i]
            combined = []
            if not_inter_mask.sum() != 0:
                hor_no_inter = create_line_set(
                    reachingfield.inter_obj_center[None],
                    reachingfield.inter_obj_center + 2 * horizontal_directions[not_inter_mask],
                    [0, 1, 0],
                    multiple_origins=False
                )
                ver_no_inter = create_line_set(
                    rotating_rays[step]["start_points"][not_inter_mask].reshape(-1,3),
                    rotating_rays[step]["target_points"][not_inter_mask].reshape(-1,3),
                    [0, 1, 0],
                    multiple_origins=True
                )
                combined = combined + [hor_no_inter, ver_no_inter]
            
            if (~not_inter_mask).sum() != 0:
                hor_inter = create_line_set(
                    reachingfield.inter_obj_center[None],
                    reachingfield.inter_obj_center + 2 * horizontal_directions[~not_inter_mask],
                    [1, 0, 0],
                    multiple_origins=False
                )
                
                ver_inter = create_line_set(
                    rotating_rays[step]["start_points"][~not_inter_mask].reshape(-1,3),
                    rotating_rays[step]["target_points"][~not_inter_mask].reshape(-1,3),
                    [1, 0, 0],
                    multiple_origins=True
                )
                combined = combined + [hor_inter, ver_inter]

            o3d.visualization.draw_geometries(
                [rec_mesh, floor_mesh, reachingfield.inter_obj_mesh] + combined
            )
                    

        ### Final step 
        final_step_rays = create_line_set(
            reachingfield.inter_obj_center[None],
            reachingfield.inter_obj_center + 2 * rays_dict["final_step"]["rays_directions"],
            [0, 1, 0],
            multiple_origins=False
        )
        
        o3d.visualization.draw_geometries(
            [rec_mesh, floor_mesh, reachingfield.inter_obj_mesh, final_step_rays]
        )


if __name__ == "__main__":
    main()
