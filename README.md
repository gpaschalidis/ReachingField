# ReachingField

This repository is the official implementation of the ReachingField model from 
our __3DV 2025__ paper:

**3D Whole-body Grasp Synthesis with Directional Controllability**
(https://gpaschalidis.github.io/cwgrasp/).

<a href="">
<img src="img/ray_distribution_visualization.jpg" alt="Logo" width="100%">
</a>

<p align="justify">
ReachingField is a probabilistic 3D vector field conditioned on a 3D object. 
Specifically, it defines a distribution over the directions
from which a body's arm and hand can reach an object, without penetrating 
obstacles around the object. Below, we provide visualizations illustrating how
ReachingField is formed for objects placed at various heights, ranging from low to high, 
on different receptacle objects.
</p>

<p align="center">
    <img width="360" alt="" src="img/reachingfield1.gif"/>
    <img width="360" alt="" src="img/reachingfield2.gif"/>
    <img width="360" alt="" src="img/reachingfield3.gif"/>
    <img width="360" alt="" src="img/reachingfield4.gif"/>
</p>



## Installation & Dependencies
Clone the repository using:

```bash
git clone git@github.com:gpaschalidis/reachingfield.git
cd reachingfield
```
Run the following commands:
```bash
conda create -n reachingfield python=3.9 -y
conda activate reachingfield
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia
pip install -r requirements.txt
 ```

## Installing Dataset
To try ReachingField you can use any data you want. However, for our paper we use 
the grasping objects from GRAB dataset and the ReplicaGrasp dataset.

- Download first the GRAB object mesh (`.ply`) files from the [GRAB website](https://grab.is.tue.mpg.de/).
- Download the ReplicaGrasp dataset from [FLEX github page](https://github.com/purvaten/FLEX).
- Create a folder and paste both datasets inside it, so that they follow the structure below:

```bash
      data
       │
       ├── contact_meshes
       │    ├── airplane.ply
       │    └── ...
       │
       └── replicagrasp
            ├── dset_info.npz
            └── receptacles.npz
```
 

## Testing ReachingField
To try ReachingField and generate visualizations for each of its building steps, use the following command:

```bash
python test_reachingfield.py --obj_rec_conf cup_receptacle_aabb_WC1_Top3frl_apartment_wall_cabinet_01_all_1 --grasp_type "right" --vis

```

<p align="justify" style="word-wrap: break-word;">
Inside the file <strong>dset_info.npz</strong> you will find a dictionary containing all the configurations of the
ReplicaGrasp dataset. Each dictionary's key represents a distinct configuration. You can experiment with any configuration
by replacing  <code>cup_receptacle_aabb_WC1_Top3frl_apartment_wall_cabinet_01_all_1</code> in the command
above with the corresponding key.
</p>


## Citation
If you found this work influential or helpful for your research, please cite:
```
@inproceedings{paschalidis2025cwgrasp,
  title     = {{3D} {W}hole-Body Grasp Synthesis with Directional Controllability},
  author    = {Paschalidis, Georgios and Wilschut, Romana and Anti\'{c}, Dimitrije and Taheri, Omid and Tzionas, Dimitrios},
  booktitle = {{International Conference on 3D Vision (3DV)}},
  year      = {2025}
 }
```
