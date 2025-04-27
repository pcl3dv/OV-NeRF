'''TensoRF.

The source code is adopted from:
https://github.com/apchenstu/TensoRF

Reference:
[1] Chen A, Xu Z, Geiger A, et al.
    Tensorf: Tensorial radiance fields. European Conference on Computer Vision
'''

import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import os
from PIL import Image
from torchvision import transforms as T
import pdb
from .ray_utils import *
import json
import glob 
import cv2
import random
import torch.nn.functional as F
import torchvision.transforms.functional as TF

def get_json_content(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data

def circle(radius=3.5, h=0.0, axis='z', t0=0, r=1):
    if axis == 'z':
        return lambda t: [radius * np.cos(r * t + t0), radius * np.sin(r * t + t0), h]
        # return lambda t: [radius * np.sin(r * t + t0), radius * np.cos(r * t + t0), h]
    elif axis == 'y':
        return lambda t: [radius * np.cos(r * t + t0), h, radius * np.sin(r * t + t0)]
    else:
        return lambda t: [h, radius * np.cos(r * t + t0), radius * np.sin(r * t + t0)]


def cross(x, y, axis=0):
    T = torch if isinstance(x, torch.Tensor) else np
    return T.cross(x, y, axis)


def normalize(x, axis=-1, order=2):
    if isinstance(x, torch.Tensor):
        l2 = x.norm(p=order, dim=axis, keepdim=True)
        return x / (l2 + 1e-8), l2

    else:
        l2 = np.linalg.norm(x, order, axis)
        l2 = np.expand_dims(l2, axis)
        l2[l2 == 0] = 1
        return x / l2,


def cat(x, axis=1):
    if isinstance(x[0], torch.Tensor):
        return torch.cat(x, dim=axis)
    return np.concatenate(x, axis=axis)


def look_at_rotation(camera_position, at=None, up=None, inverse=False, cv=False):
    """
    This function takes a vector 'camera_position' which specifies the location
    of the camera in world coordinates and two vectors `at` and `up` which
    indicate the position of the object and the up directions of the world
    coordinate system respectively. The object is assumed to be centered at
    the origin.
    The output is a rotation matrix representing the transformation
    from world coordinates -> view coordinates.
    Input:
        camera_position: 3
        at: 1 x 3 or N x 3  (0, 0, 0) in default
        up: 1 x 3 or N x 3  (0, 1, 0) in default
    """

    if at is None:
        at = torch.zeros_like(camera_position)
    else:
        at = torch.tensor(at).type_as(camera_position)
        
    if up is None:
        up = torch.zeros_like(camera_position)
        up[2] = -1
    else:
        up = torch.tensor(up).type_as(camera_position)

    z_axis = normalize(at - camera_position)[0]
    x_axis = normalize(cross(up, z_axis))[0]
    y_axis = normalize(cross(z_axis, x_axis))[0]

    R = cat([-x_axis[:, None], y_axis[:, None], -z_axis[:, None]], axis=1)
    return R


def gen_path(pos_gen, at=(0, 0, 0), up=(0, -1, 0), frames=180):
    c2ws = []
    for t in range(frames):
        c2w = torch.eye(4)
        cam_pos = torch.tensor(pos_gen(t * (360.0 / frames) / 180 * np.pi))
        #print("{cam_pos}:", t, cam_pos)
        cam_rot = look_at_rotation(cam_pos, at=at, up=up, inverse=False, cv=True)
        c2w[:3, 3], c2w[:3, :3] = cam_pos, cam_rot
        c2ws.append(c2w)
    return torch.stack(c2ws)


class scannetDataset(Dataset):
    def __init__(self, datadir, patch_size=256, patch_stride=1, split='train', downsample=1.0, wh=[1296, 968], 
                 is_stack=False, skip_every_for_val_split=10, clip_input=1., loaded_center=[0., 0., 0.]):
        
        self.clip_input = clip_input
        self.root_dir = datadir
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.split = split
        self.is_stack = is_stack
        self.downsample = downsample        
        self.skip_every_for_val_split = skip_every_for_val_split
        self.define_transforms()            
        
        # self.blender2opencv = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        self.white_bg = False                
        self.img_wh = (int(wh[0]/downsample),int(wh[1]/downsample))
        
        self.near_far = [0.05, 2.5]
        self.scene_bbox = torch.tensor([[-1.5, -1.5, -1.5], [1.5, 1.5, 1.5]]) # or 1.0 
        print("near_far:", self.near_far)
        print("scene_bbox:", self.scene_bbox)

        self.center = torch.mean(self.scene_bbox, axis=0).float().view(1, 1, 3)
        self.radius = (self.scene_bbox[1] - self.center).float().view(1, 1, 3)
        self.invradius = 1.0 / (self.scene_bbox[1] - self.center).float().view(1, 1, 3)
        self.loaded_center = loaded_center
        
        self.read_meta()

    def read_meta(self):
        self.intrinsics = np.loadtxt(os.path.join(self.root_dir, "intrinsics.txt"))
        self.intrinsics[:2] *= (np.array(self.img_wh)/ np.array([1296, 968])).reshape(2, 1)
        
        img_dir_sorted = sorted(glob.glob(os.path.join(self.root_dir, 'rgb/*')))
        seg_dir_sorted = sorted(glob.glob(os.path.join(self.root_dir, 'semantic/*')))  
        pose_dir_sorted = sorted(glob.glob(os.path.join(self.root_dir, 'pose/*')))
        sammask_dir_sorted = sorted(glob.glob(os.path.join(self.root_dir, 'sam_mask/*'))) 
        clipsam_dir_sorted = sorted(glob.glob(os.path.join(self.root_dir, 'clipsam/*')))   
        print("The first label of the pseudo data:", clipsam_dir_sorted[0])
        
        self.pose_l, self.img_path_l, self.seg_path_l = [], [], []
        self.clipsam, self.maskclip = [], []
        self.sammask = []
        train_pose_l, val_pose_l = [], []
        tarin_img_path_l, val_img_path_l = [], []
        train_seg_path_l, val_seg_path_l = [], []
        train_clipsam, val_clipsam = [], []
        train_maskclip, val_maskclip = [], []
        train_sammask, val_sammask = [], []

        for ii in range(len(img_dir_sorted)):
            pose = np.loadtxt(pose_dir_sorted[ii])
            pose = np.array(pose).reshape(4, 4)

            if (ii) % self.skip_every_for_val_split:        
                train_pose_l.append(pose)
                tarin_img_path_l.append(img_dir_sorted[ii])
                train_seg_path_l.append(seg_dir_sorted[ii])
                train_clipsam.append(clipsam_dir_sorted[ii])
                train_sammask.append(sammask_dir_sorted[ii])
            else:
                val_pose_l.append(pose)
                val_img_path_l.append(img_dir_sorted[ii])    
                val_seg_path_l.append(seg_dir_sorted[ii])
                val_clipsam.append(clipsam_dir_sorted[ii])
                val_sammask.append(sammask_dir_sorted[ii])
        
        if self.split == 'train':
            self.pose_l = np.stack(train_pose_l, 0)
            self.img_path_l = np.stack(tarin_img_path_l, 0) 
            self.seg_path_l = np.stack(train_seg_path_l, 0)  
            self.clipsam = train_clipsam
            self.sammask = train_sammask
        else:
            self.pose_l = np.stack(val_pose_l, 0) 
            self.img_path_l = np.stack(val_img_path_l, 0)  
            self.seg_path_l = np.stack(val_seg_path_l, 0)   
            self.clipsam = val_clipsam
            self.sammask = val_sammask
            
        print("Maximum value before pose scale: ", np.max(np.abs(self.pose_l[:, :3, 3])))
        scale_factor = 1.0 
        scale_factor /= float(np.max(np.abs(self.pose_l[:, :3, 3])))
        self.pose_l[..., 3] *= scale_factor        # abs_max = 1.0
        print("Maximum value after pose scale: ", np.max(np.abs(self.pose_l[:, :3, 3])))

        # ray directions for all pixels, same for all images (same H, W, focal)
        self.directions = get_ray_directions(self.img_wh[1], self.img_wh[0], 
                                             [self.intrinsics[0, 0], self.intrinsics[1, 1]], 
                                             center = self.intrinsics[:2, 2])  # (h, w, 3) 
        self.directions = self.directions / torch.norm(self.directions, dim=-1, keepdim=True)   
        

        ########################## select some img from the img_list ##########################
        # print(len(self.img_path_l))   
        if self.clip_input != 1.:   
            # Calculate the number of indices to select
            num_indices = int(self.clip_input * len(self.img_path_l))
            # Randomly select the indices
            selected_indices = random.sample(range(len(self.img_path_l)), num_indices)
            # Retrieve the corresponding elements
            self.img_path_l = [self.img_path_l[i] for i in selected_indices]
            self.pose_l = [self.pose_l[i] for i in selected_indices]
            self.seg_path_l = [self.seg_path_l[i] for i in selected_indices]
            self.clipsam = [self.clipsam[i] for i in selected_indices]
            self.sammask = [self.sammask[i] for i in selected_indices]
        # print(len(self.img_path_l))  
        ########################## select some img from the img_list ##########################

        self.poses = []
        self.all_rays = []
        self.all_rgbs = []
        self.SAM_images = [] 
        self.all_segs = []
        self.all_clipsams = []
        self.all_sammasks = []
        
        assert len(self.pose_l) == len(self.img_path_l) == len(self.seg_path_l) == len(self.clipsam) == len(self.sammask)
        for img_fname, pose_fname, seg_fname, clipsam_fname, sammask_fname in tqdm(zip(self.img_path_l, self.pose_l, self.seg_path_l, self.clipsam, self.sammask), desc=f'Loading data {self.split} ({len(self.img_path_l)})'):
            image_path = os.path.join(img_fname)
            img = Image.open(image_path)
            seg_path = os.path.join(seg_fname)
            seg_img = Image.open(seg_path)

            clipsam_path = os.path.join(clipsam_fname)
            clipsam_img = Image.open(clipsam_path)            

            sammask_path = os.path.join(sammask_fname)
            sam_masks = torch.load(sammask_path, map_location=torch.device('cpu'))    # lists 
            
            if self.downsample!=1.0:
                img = img.resize(self.img_wh, Image.LANCZOS)
                seg_img = seg_img.resize(self.img_wh, Image.LANCZOS)
                clipsam_img = clipsam_img.resize(self.img_wh, Image.LANCZOS)
                
            img = self.transform(img)  # (4, h, w)
            img = img.view(img.shape[0], -1).permute(1, 0)  # (h*w, 4) RGBA
            if img.shape[-1] == 4:
                img = img[:, :3] * img[:, -1:] + (1 - img[:, -1:])  # blend A to RGB
            
            seg_img = (np.array(seg_img)).astype(np.int8)  # [H, W]
            seg_img = torch.from_numpy(seg_img)            # tensor: [H, W]
            seg_img = seg_img.reshape(-1).long()           # [H*W]
            
            clipsam_img = (np.array(clipsam_img)).astype(np.int8)  # [H, W]
            clipsam_img = torch.from_numpy(clipsam_img)            # tensor: [H, W]
            clipsam_img = clipsam_img.reshape(-1).long()
            
            self.all_rgbs.append(img)
            self.all_segs.append(seg_img)
            self.all_clipsams.append(clipsam_img)
            self.all_sammasks.append(sam_masks)    
            
            ########## SAM_image load ##########
            SAM_image = cv2.imread(img_fname)
            SAM_image = cv2.cvtColor(SAM_image, cv2.COLOR_BGR2RGB)
            self.SAM_images.append(SAM_image)
            ########## SAM_image load ##########
                    
            c2w = pose_fname# @ cam_trans
            # c2w[0:3, 1:3] *= -1             # opencv=>opengl
            c2w = torch.FloatTensor(c2w)
            self.poses.append(c2w)  # C2W

            rays_o, rays_d = get_rays(self.directions, c2w)  # both (h*w, 3)
            self.all_rays += [torch.cat([rays_o, rays_d], 1)]  # (h*w, 6)
            
        
        # import pdb; pdb.set_trace()
        self.poses = torch.stack(self.poses)
        radius = 0.005                                                   
        up = torch.mean(self.poses[:, :3, 1], dim=0).tolist()            
        pos_gen = circle(radius=radius, h=-0.001, axis='z')              
        self.render_path = gen_path(pos_gen, up=None, frames=10)         
        self.render_path[:, :3, 3] += torch.tensor(self.loaded_center)   

        all_rays = self.all_rays
        all_rgbs = self.all_rgbs
        all_segs = self.all_segs
        all_clipsams = self.all_clipsams        
        
        # import pdb; pdb.set_trace()
        self.all_rays = torch.cat(self.all_rays, 0) # (len(self.meta['frames'])*h*w,6)
        self.all_rgbs = torch.cat(self.all_rgbs, 0) # (len(self.meta['frames'])*h*w,3)
        self.all_segs = torch.cat(self.all_segs, 0) # (len(self.meta['frames'])*h*w,1)
        self.all_clipsams = torch.cat(self.all_clipsams, 0)  # (len(self.meta['frames'])*h*w,1)
        
        if self.is_stack:
            self.all_rays_stack = torch.stack(all_rays, 0).reshape(-1,*self.img_wh[::-1], 6)  # (len(self.meta['frames']),h,w,6)
            self.all_rgbs_stack = torch.stack(all_rgbs, 0).reshape(-1,*self.img_wh[::-1], 3)  # (len(self.meta['frames']),h,w,3)
            self.all_segs_stack = torch.stack(all_segs, 0).reshape(-1,*self.img_wh[::-1], 1)  # (len(self.meta['frames']),h,w,1)
            # self.all_nvs_all_rays = torch.stack(nvs_all_rays, 0).reshape(-1,*self.img_wh[::-1], 6)  # (nvs_number,h,w,6)

        self.all_rays = self.all_rays.cuda()
        self.all_rgbs = self.all_rgbs.cuda()
        self.all_segs = self.all_segs.cuda()
        self.all_clipsams = self.all_clipsams.cuda()
        
    def read_classes_names(self):
        # read class names
        with open(os.path.join(self.root_dir, 'classes.txt'), 'r') as f:
            lines = f.readlines()
            self.classes = [line.strip() for line in lines]
            # print("Classes before sort: ", self.classes)
            self.classes[0] = ' ' 
            # self.classes.sort()   
            # print("Classes after sort: ", self.classes)
            print("Classes: ", self.classes)
    
    
    def read_segmentation_maps_train(self):
        seg_maps, seg_maps_HW = [], []
        for img_seg in tqdm(self.seg_path_l):     
            img_seg = Image.open(img_seg).convert('L')
            if self.downsample != 1.0:
                img_seg = img_seg.resize(self.img_wh, Image.NEAREST) # [W, H]  
            
            img_seg = (np.array(img_seg)).astype(np.int8)            # [H, W]
            seg_maps_HW.append(img_seg)
            img_seg = img_seg.flatten()      
            
            seg_for_one_image = [] 
            for i in range(len(self.classes)): 
                seg_for_one_class = np.zeros_like(img_seg)       # [H*W]
                seg_for_one_class[img_seg == i] = 1              # [H*W]
                seg_for_one_image.append(seg_for_one_class)
                
            #import pdb; pdb.set_trace()
            seg_for_one_image = np.stack(seg_for_one_image, axis=0)   # [classes, H*W] 
            seg_for_one_image = seg_for_one_image.transpose(1, 0)     # [H*W, classes] 
            seg_maps.append(seg_for_one_image)     
            
        self.seg_maps_train = np.stack(seg_maps, axis=0)
        self.seg_maps_HW_train = np.stack(seg_maps_HW, axis=0)
        
        
    def read_segmentation_maps(self):
        seg_maps = []
        seg_maps_HW = []
        
        for img_seg in tqdm(self.seg_path_l):     
            img_seg = Image.open(img_seg).convert('L')
            if self.downsample != 1.0:
                img_seg = img_seg.resize(self.img_wh, Image.NEAREST) # [W, H]  
            img_seg = (np.array(img_seg)).astype(np.int8)            # [H, W]
            seg_maps_HW.append(img_seg)
            img_seg = img_seg.flatten()      
            
            seg_for_one_image = [] 
            for i in range(len(self.classes)):  
                seg_for_one_class = np.zeros_like(img_seg)       # [H*W]
                seg_for_one_class[img_seg == i] = 1              # [H*W]
                seg_for_one_image.append(seg_for_one_class)
                
            seg_for_one_image = np.stack(seg_for_one_image, axis=0)   # [classes, H*W] 
            seg_for_one_image = seg_for_one_image.transpose(1, 0)     # [H*W, classes] 
            seg_maps.append(seg_for_one_image)     
            
        self.seg_maps = np.stack(seg_maps, axis=0)
        self.seg_maps_HW = np.stack(seg_maps_HW, axis=0)

        
    
    @torch.no_grad()
    def read_clip_features_and_relevancy_maps(self, feature_dir, text_features, test_prompt=None):
        '''
        the input text_features are already normalized
        '''

        feature_paths = sorted(glob.glob(f'{feature_dir}/*'))
        features, relevancy_maps = [], []
        view_idx = 0 
        
        print(f'Reading CLIP feaures from {feature_paths[0]}...')
        for i in tqdm(self.img_path_l):   
            if test_prompt is not None and view_idx != int(test_prompt): continue   
            feature_path = os.path.join(feature_dir, i.split('/')[-1][:-4] + '.pth') 
            feature = torch.load(feature_path, map_location='cuda') # [scale(=3), dim, H, W]
            if feature.size(-2) != self.img_wh[1] or feature.size(-1) != self.img_wh[0]:
                feature = TF.resize(feature, size=(self.img_wh[1], self.img_wh[0]))
            feature = F.normalize(feature, dim=1) 
            features.append(feature.cpu())
            
            # ################## save the relevancy map as image ##################
            # if test_prompt is not None:
            #     text_prompt_save_path = 'clip_features/clip_relevancy_maps_'
            #     os.makedirs(text_prompt_save_path, exist_ok=True)
            #     for i in range(relevancy_map.size(0)):
            #         for j in range(relevancy_map.size(1)):
                        
            #             img = relevancy_map[i,j].cpu().numpy()
            #             img = (img * 255).astype(np.uint8)
            #             img = Image.fromarray(img)
            #             img.save(f'{text_prompt_save_path}/{i}_{self.classes[j]}.png')
            #     exit()
            
        features = torch.stack(features, dim=0) # [n_frame, scale(=3), dim, H, W]
        self.clip_featureMap = features  # [n_frame, scale(=3), dim, H, W]
        features = features.permute(0,3,4,1,2) # [n_frame, H, W, scale(=3), dim]
        self.all_features = torch.reshape(features, (-1, features.size(-2), features.size(-1))) # [-1, scale, dim]

    
    def define_transforms(self):
        self.transform = T.ToTensor()
        
    def __len__(self):
        return len(self.all_rays_stack)


    def __getitem__(self, i):
        '''
        Only used for patch sampling
        '''

        h_idx = random.randint(0, self.img_wh[1]-self.patch_size)   
        w_idx = random.randint(0, self.img_wh[0]-self.patch_size)
        
        # print("#######################", self.all_rays_stack.shape, i)
        ray_sample = self.all_rays_stack[i]
        rays = ray_sample[h_idx:h_idx+self.patch_size:self.patch_stride, w_idx:w_idx+self.patch_size:self.patch_stride, :]  

        avg_pool = torch.nn.AvgPool2d(8, ceil_mode=True)
        rays = avg_pool(rays.permute(2,0,1)[None,...]).squeeze(0).permute(1,2,0)       
        # print("After pooling, rays.shape:",  rays.shape)

        rgb_sample = self.all_rgbs_stack[i]
        rgbs = rgb_sample[h_idx:h_idx+self.patch_size:self.patch_stride, w_idx:w_idx+self.patch_size:self.patch_stride, :]
        
        return {
            'rays': rays, # [patch_size//8, patch_size//8, 6]
            'rgbs': rgbs  # [pathc_size, patch_size, 3]
        }    
    