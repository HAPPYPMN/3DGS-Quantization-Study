import os
import torch
import json
import random
import numpy as np


from utils.camera_utils import camera_to_JSON, cameraList_from_camInfos
from utils.system_utils import searchForMaxIteration

from scene.gaussian_model import GaussianModel
from scene.dataset_readers import sceneLoadTypeCallbacks


def loaddata(source_path, model_path, gaussians, eval = False, shuffle = False):
    if os.path.exists(os.path.join(source_path, "sparse")):
        scene_info = sceneLoadTypeCallbacks["Colmap"](source_path, "images", eval)
        
    loaded_iter = searchForMaxIteration(os.path.join(model_path, "point_cloud"))  
    
    if not loaded_iter:
        with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(model_path, "input.ply") , 'wb') as dest_file:
            dest_file.write(src_file.read())  
            json_cams = []
            camlist = []
            camlist.extend(scene_info.train_cameras)
            
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)
        
    if shuffle:
        random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
        
    views = cameraList_from_camInfos(scene_info.train_cameras, 1.0)
        
    cameras_extent = scene_info.nerf_normalization["radius"]

    if loaded_iter:
        gaussians.load_ply(os.path.join(model_path,
                                                    "point_cloud",
                                                    "iteration_" + str(loaded_iter),
                                                    "point_cloud.ply"))
    else:
        gaussians.create_from_pcd(scene_info.point_cloud, cameras_extent)
        
    return views, loaded_iter

def save_gaussian_data(save_path, gaussians):
    os.makedirs(save_path, exist_ok=True)

    np.save(os.path.join(save_path, "means3D.npy"), gaussians.get_xyz.detach().cpu().numpy())
    np.save(os.path.join(save_path, "opacity.npy"), gaussians.get_opacity.detach().cpu().numpy())
    np.save(os.path.join(save_path, "scales.npy"), gaussians.get_scaling.detach().cpu().numpy())
    np.save(os.path.join(save_path, "rotations.npy"), gaussians.get_rotation.detach().cpu().numpy())
    np.save(os.path.join(save_path, "sh.npy"), gaussians.get_features.detach().cpu().numpy())
    print(f"Saved Gaussian attributes to {save_path}")
    
def save_view_info(save_path, views):
    os.makedirs(save_path, exist_ok=True)
    
    # 假设所有视图高度宽度一致
    image_height = int(views[0].image_height)
    image_width  = int(views[0].image_width)
    np.save(os.path.join(save_path, "image_hw.npy"), np.array([image_height, image_width]))

    view_matrices = []
    proj_matrices = []
    camera_centers = []

    for view in views:
        view_matrices.append(view.world_view_transform.detach().cpu().numpy())  # shape (4, 4)
        proj_matrices.append(view.full_proj_transform.detach().cpu().numpy())   # shape (4, 4)
        camera_centers.append(view.camera_center.detach().cpu().numpy())        # shape (3,)

    np.save(os.path.join(save_path, "view_matrices.npy"), np.stack(view_matrices))       # (N, 4, 4)
    np.save(os.path.join(save_path, "proj_matrices.npy"), np.stack(proj_matrices))       # (N, 4, 4)
    np.save(os.path.join(save_path, "camera_centers.npy"), np.stack(camera_centers))     # (N, 3)

    print(f"Saved view information to {save_path}")
    
if __name__ == "__main__":
    source_path = '/data/peiminnan/3DGS-Arch/data'
    model_path = '/data/peiminnan/3DGS-Arch/model'
    save_path = '/data/peiminnan/3DGS-Quantization-Study/data'

    dataset = 'Palace'
    source_path = os.path.join(source_path, dataset)
    model_path = os.path.join(model_path, dataset)
    save_path = os.path.join(save_path, dataset)

    sh_degree = 3
    gaussians = GaussianModel(sh_degree)
    views, loaded_iter = loaddata(source_path, model_path, gaussians)

    save_gaussian_data(save_path, gaussians)
    save_view_info(save_path, views)