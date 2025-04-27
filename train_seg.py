import os
from tqdm.auto import tqdm
from opt import config_parser
from sklearn.decomposition import PCA

from models.DINO_extractor import VitExtractor
from models.losses import CorrelationLoss
from renderer import *
from funcs import *

from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
import datetime

from dataLoader import dataset_dict
import sys
from pathlib import Path


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
renderer = OctreeRender_trilinear_fast


def InfiniteSampler(n):
    i = n - 1
    order = np.random.permutation(n)
    while True:
        yield order[i]
        i += 1
        if i >= n:
            np.random.seed()
            order = np.random.permutation(n)
            i = 0

class InfiniteSamplerWrapper(torch.utils.data.sampler.Sampler):
    def __init__(self, data_source):
        self.num_samples = len(data_source)

    def __iter__(self):
        return iter(InfiniteSampler(self.num_samples))

    def __len__(self):
        return 2 ** 31

class SimpleSampler:
    def __init__(self, total, batch):
        self.total = total
        self.batch = batch
        self.curr = total
        self.ids = None

    def nextids(self):
        self.curr+=self.batch
        if self.curr + self.batch > self.total:
            self.ids = torch.LongTensor(np.random.permutation(self.total))
            self.curr = 0
        return self.ids[self.curr:self.curr+self.batch]

    
    
@torch.no_grad()
def render_test(args):
    # init dataset
    dataset = dataset_dict[args.dataset_name]
    # test_dataset = dataset(args.datadir, split='test', downsample=args.downsample_test, is_stack=True)
    test_dataset = dataset(args.datadir, split='train', downsample=args.downsample_test, is_stack=True)
    test_dataset.read_classes_names()
    if args.has_segmentation_maps:
        test_dataset.read_segmentation_maps()
    c2ws = test_dataset.render_path
    classes = test_dataset.classes
    ndc_ray = args.ndc_ray
    
    if not os.path.exists(args.ckpt):
        print('the ckpt path does not exists!!')
        return

    ckpt = torch.load(args.ckpt, map_location=device)
    kwargs = ckpt['kwargs']
    kwargs.update({'device': device})

    tensorf = eval(args.model_name)(**kwargs)           # tensorf 
    tensorf.change_to_feature_mode(device)
    tensorf.load(ckpt)
    tensorf.eval()

    logfolder = os.path.dirname(args.ckpt)     
    data_name = os.path.basename(args.datadir) 
    
    # import pdb; pdb.set_trace()
    if args.render_seg_test:    
        if args.has_segmentation_maps:
            os.makedirs(f'{logfolder}/segmentations_test', exist_ok=True)
            evaluation_segmentation_test(test_dataset, tensorf, renderer, f'{logfolder}/segmentations_test/', N_vis=-1, N_samples=-1, classes=classes, ndc_ray=ndc_ray,device=device)
        else:
            raise ValueError('the dataset does not have segmentation maps!!')
        
    if args.render_seg_path:   
        os.makedirs(f'{logfolder}/segmentations_path', exist_ok=True)
        evaluation_segmentation_path(test_dataset, tensorf, c2ws, renderer, f'{logfolder}/segmentations_path/',
                                        N_vis=-1, N_samples=-1, classes=classes, ndc_ray=ndc_ray,device=device)

    if args.render_seg_train:   
        train_dataset = dataset(args.datadir, split='train', downsample=args.downsample_test, is_stack=True)
        os.makedirs(f'{logfolder}/segmentations_train', exist_ok=True)
        evaluation_segmentation_train(train_dataset, tensorf, renderer, f'{logfolder}/segmentations_train/', f'{logfolder}/segmentations_train_vis', N_vis=-1, N_samples=-1, classes=classes, ndc_ray=ndc_ray,device=device)

    # depth map
    if args.render_seg_depth:
        os.makedirs(f'{logfolder}/segmentations_depth', exist_ok=True)
        evaluation_segmentation_depth(test_dataset, tensorf, renderer, f'{logfolder}/segmentations_depth/',
                                        N_vis=-1, N_samples=-1, classes=classes, ndc_ray=ndc_ray,device=device)

    # feature pca & text feature 
    if args.render_feature:
        text = args.reference_text
        if text is None:            
            os.makedirs(f'{logfolder}/feature_pca', exist_ok=True)
            evaluation_feature_pca_path(test_dataset, tensorf, c2ws, renderer, f'{logfolder}/feature_pca/',
                                        N_vis=-1, N_samples=-1, ndc_ray=ndc_ray,device=device)
        else:
            os.makedirs(f'{logfolder}/feature_text_{text}', exist_ok=True)
            evaluation_feature_text_activation_path(test_dataset, tensorf, c2ws, renderer, f'{logfolder}/feature_text_{text}/',
                                        N_vis=-1, N_samples=-1, ndc_ray=ndc_ray, text=text, device=device)

    # selection vector
    if args.render_select:
        os.makedirs(f'{logfolder}/select', exist_ok=True)
        evaluation_select_path(test_dataset, tensorf, c2ws, renderer, f'{logfolder}/select/',
                                    N_vis=-1, N_samples=-1, ndc_ray=ndc_ray,device=device)


        
        
def reconstruction(args):

    ################################## init dataset #################################
    dataset = dataset_dict[args.dataset_name]   
    
    
    ##################### feature_train_dataset #####################
    feature_train_dataset = dataset(args.datadir, split='train', patch_size=args.patch_size, downsample=args.ray_downsample_train, is_stack=True, clip_input=args.clip_input) 
    
    feature_train_dataset.read_classes_names()  # read class names
    if args.has_segmentation_maps:              
        # feature_train_dataset.read_segmentation_maps_train()  # read segmentation maps   
        feature_train_dataset.read_segmentation_maps()
        
    ndc_ray = args.ndc_ray      
    
    ################################ get text features #################################  
    classes = feature_train_dataset.classes
    
    clip_model, _ = clip.load("model/ViT-B-16.pt", device=device)
    text = clip.tokenize(classes).to(device)
    with torch.no_grad():
        text_features = clip_model.encode_text(text).float() # [N, 512]
        text_features = F.normalize(text_features, dim=1)
    del clip_model
    
    ################################# init log file #################################
    if args.add_timestamp:
        logfolder = f'{args.basedir}/{args.expname}{datetime.datetime.now().strftime("-%Y%m%d-%H%M%S")}'
    else:
        logfolder = f'{args.basedir}/{args.expname}'
    
    os.makedirs(logfolder, exist_ok=True)
    summary_writer = SummaryWriter(logfolder)
    init_logger(Path(logfolder))
    logger.info(args)
    logger.info(f'classes: {classes}')

    os.makedirs(f'{logfolder}/imgs_vis', exist_ok=True)

    
        
    ################################## get pre-computed image CLIP features #################################
    feature_train_dataset.read_clip_features_and_relevancy_maps(args.feature_dir, text_features, args.test_prompt)
    # return the gt labels from pre-compute clip feature by using clip model 
    # self.all_features: [n_frame*H*W, scale, dim]
    # self.all_relevancies: [n_frame*H*W, scale, num_classes]


    ################################## init parameters#################################
    aabb = feature_train_dataset.scene_bbox.to(device)     
    reso_cur = N_to_reso(args.N_voxel_init, aabb)
    nSamples = min(args.nSamples, cal_n_samples(reso_cur,args.step_ratio))


    ################################## load pre-trained nerf #################################
    assert args.ckpt is not None, 'Have to be pre-trained to get the density field!'

    ckpt = torch.load(args.ckpt, map_location=device)
    kwargs = ckpt['kwargs']
    kwargs.update({'device':device})
    tensorf = eval(args.model_name)(**kwargs)
    tensorf.load(ckpt)
   
    tensorf.change_to_feature_mode(device)


    ################################## init DINO #################################
    dino = VitExtractor(model_name='dino_vitb8', device=device)


    ################################## init CorrelationLoss #################################
    correlation_loss = CorrelationLoss(args)

    
    ################################## init Loss #################################
    # CrossEntropyLoss = torch.nn.CrossEntropyLoss(ignore_index = 0)
    CrossEntropyLoss = torch.nn.CrossEntropyLoss()
    crossentropy_loss = lambda logit, label: CrossEntropyLoss(logit, label)  
    
    
    ################################## set optimizer #################################
    grad_vars = tensorf.get_optparam_groups_feature_mod(args.lr_init, args.lr_basis)
    if args.lr_decay_iters > 0:
        lr_factor = args.lr_decay_target_ratio**(1/args.lr_decay_iters)
    else:
        args.lr_decay_iters = args.n_iters
        lr_factor = args.lr_decay_target_ratio**(1/args.n_iters)
    
    optimizer = torch.optim.Adam(grad_vars, betas=(0.9, 0.99))


    ################################## prepare training data #################################
    allrays, allrgbs, allfeatures = feature_train_dataset.all_rays, feature_train_dataset.all_rgbs, feature_train_dataset.all_features
    # allrays: [rays_o, rays_d],  (len(self.meta['frames'])*h*w,6)
    # allrgbs: images, (len(self.meta['frames'])*h*w,3)
    # all_features: [n_frame*H*W, scale, dim]
    
    allrelevancies = feature_train_dataset.all_relevancies 
    allclipsams = feature_train_dataset.all_clipsams
    
    trainingSampler = SimpleSampler(allrays.shape[0], args.batch_size)  

    ################################## training option #################################
    out_rgb = False
    dc = DistinctColors()
    

    ################################## training loop #################################
    torch.cuda.empty_cache()
    pbar = tqdm(range(args.n_iters), miniters=args.progress_refresh_rate, file=sys.stdout, bar_format='{l_bar}{r_bar}')
    
    
    ################################## update pseudo step #################################
    update_pseudo_step_list = [i for i in range(10000, 15000, args.update_pseudo_step)]   
    print("update_pseudo_step_list:", update_pseudo_step_list)
    
    
    # # ##################### patch_train_dataset #####################
    patch_train_dataset = dataset(args.datadir, split='train', patch_size=args.patch_size, downsample=args.patch_downsample_train, is_stack=True)
    patch_train_loader = iter(torch.utils.data.DataLoader(patch_train_dataset, 
                                                          batch_size=args.patch_num, 
                                                          sampler=InfiniteSamplerWrapper(patch_train_dataset), 
                                                          num_workers=4, 
                                                          pin_memory=True))
    
    
    for iteration in pbar:    
        ################ change to feature and rgb joint training mode ################
        if iteration == args.joint_start_iter:
            out_rgb = True
            #original_tensorf = tensorf
            tensorf.change_to_feature_rgb_mode()
            # new optimizer
            grad_vars = tensorf.get_optparam_groups_feature_rgb_mode(args.lr_init, args.lr_basis)
            optimizer = torch.optim.Adam(grad_vars, betas=(0.9, 0.99))
        ################ change to feature and rgb joint training mode ################
        

        ################
        ##feature_loss##
        ################
        ray_idx = trainingSampler.nextids()
        # labels
        rays_train, features_train = allrays[ray_idx], allfeatures[ray_idx].to(device)
        relevancies_train = allrelevancies[ray_idx].to(device)
        
        
        # outputs from the field  
        feature_map, select_map, rgb_map = renderer(rays_train, tensorf, chunk=args.chunk_size, N_samples=nSamples, 
                            ndc_ray=ndc_ray, is_train=True, render_feature=True, out_rgb=out_rgb, device=device)
        
        
        ## feature loss
        selected_features = torch.sum(features_train * select_map[...,None], dim=1)
        selected_features = F.normalize(selected_features, dim=-1)
        feature_loss = 1 - F.cosine_similarity(feature_map, selected_features, dim=1).mean()
        
        
        ## segmentation probability from predited feature_map
        feature_map_normalized = F.normalize(feature_map, dim=1)
        relevancy_map = torch.mm(feature_map, text_features.T)
        relevancy_map = F.log_softmax(relevancy_map, dim=-1)  # [N, N_classes]
        
        
        if iteration in update_pseudo_step_list:
            update_all_clipsams = evaluation_segmentation_train_with_SAM(feature_train_dataset, tensorf, renderer, f'{logfolder}/segmentations_train_refine/', f'{logfolder}/segmentations_train_refine_vis', N_vis=-1, N_samples=-1, classes=classes, ndc_ray=ndc_ray, device=device, iteration=iteration)
            
            train_nvs_c2ws = feature_train_dataset.render_path
            nvs_all_clipsams = evaluation_segmentation_path_with_SAM(feature_train_dataset, tensorf, train_nvs_c2ws, renderer, f'{logfolder}/segmentations_train_nvs_refine_rgb/', f'{logfolder}/segmentations_train_nvs_refine_vis', N_vis=-1, N_samples=-1, classes=classes, ndc_ray=ndc_ray, device=device)   
            
            
        
        # import pdb; pdb.set_trace()
        if iteration < 10000:
            clipsams_train = allclipsams[ray_idx]     
            relevancy_map_train = clipsams_train
        else:
            update_clipsams_train = update_all_clipsams[ray_idx]
            relevancy_map_train = update_clipsams_train
        relevancy_loss = crossentropy_loss(relevancy_map, relevancy_map_train) 
        
        
        nvs_relevancy_loss = 0.        
        if iteration >= 10000:
            nvs_ray_idx = nvs_trainingSampler.nextids()
            nvs_rays_train = nvs_allrays[nvs_ray_idx]
            nvs_feature_map, nvs_select_map, nvs_rgb_map = renderer(nvs_rays_train, tensorf, chunk=args.chunk_size, N_samples=nSamples, 
                    ndc_ray=ndc_ray, is_train=True, render_feature=True, out_rgb=out_rgb, device=device)
            
            nvs_relevancy_map = torch.mm(nvs_feature_map, text_features.T)
            nvs_relevancy_map = F.log_softmax(nvs_relevancy_map, dim=-1)  
            nvs_relevancy_loss = crossentropy_loss(nvs_relevancy_map, nvs_all_clipsams[nvs_ray_idx]) 
            
            
        rgb_loss = 0.
        if out_rgb:
            rgbs_train = allrgbs[ray_idx].to(device)
            rgb_loss = F.mse_loss(rgb_map, rgbs_train)


        ####################
        ##correlation_loss##
        ####################
        batch = next(patch_train_loader)
        rays_train, rgbs_train = batch['rays'], batch['rgbs'].to(device) #[B, H//8, W//8, 6], [B, H, W, 3]
        feature_shape_wo_dim = rays_train.shape[:3]  
        rays_train = rays_train.reshape(-1, 6)  # rays = torch.cat([rays_o, rays_d], 1)  # (h*w, 6)  
        feature_map, _, _ = renderer(rays_train, tensorf, chunk=args.chunk_size, N_samples=nSamples, 
                            ndc_ray=ndc_ray, is_train=True, render_feature=True, device=device)     
        feature_map_normalized = F.normalize(feature_map, dim=1) # [N1,D]  
        relevancy_map = torch.mm(feature_map_normalized, text_features.T) # [N1,N2=num_classes]
        log_p_class = F.log_softmax(relevancy_map / args.temperature, dim=1) # [N1,N2] 
        log_p_class = log_p_class.reshape(*feature_shape_wo_dim, -1)  
        
        
        # import pdb; pdb.set_trace()
        with torch.no_grad():
            # extract dino feature
            dino_ret = dino.get_vit_feature(rgbs_train.permute(0,3,1,2))  
            dino_feature_map = dino_ret[:, 1:, :]   
            dino_feature_map = dino_feature_map.reshape(dino_ret.size(0), feature_shape_wo_dim[1], feature_shape_wo_dim[2], dino_ret.size(-1)) 
        dino_pos_loss, dino_neg_loss = correlation_loss(dino_feature_map, log_p_class)
        dino_pos_loss = dino_pos_loss * args.dino_pos_weight if args.dino_pos_weight != 0 else 0
        dino_neg_loss = dino_neg_loss * args.dino_neg_weight if args.dino_neg_weight != 0 else 0
        relevancy_loss = relevancy_loss * args.relevancy_weight if args.relevancy_weight != 0 else 0
        
        ## 
        loss = rgb_loss + dino_pos_loss + dino_neg_loss + relevancy_loss + feature_loss + nvs_relevancy_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * lr_factor

        #############
        ###logging###
        #############
        if iteration % args.progress_refresh_rate == 0:
            pbar.set_description(
                f'loss={loss.detach().item():.4f}, '   
                + f'relevancy_loss={relevancy_loss.detach().item():.4f}, ' 
                + f'nvs_relevancy_loss={nvs_relevancy_loss.detach().item():.4f}, '
                + f'feature_loss={feature_loss.detach().item():.4f}, '
            )

            feature_loss = feature_loss.detach().item() 
            rgb_loss = rgb_loss.detach().item() if out_rgb else 0.
            dino_pos_loss = dino_pos_loss.detach().item() if args.dino_pos_weight != 0 else 0
            dino_neg_loss = dino_neg_loss.detach().item() if args.dino_neg_weight != 0 else 0
            relevancy_loss = relevancy_loss.detach().item() if args.relevancy_weight != 0 else 0
            nvs_relevancy_loss = nvs_relevancy_loss.detach().item() if iteration >= 10000 else 0
            
            summary_writer.add_scalar('feature_loss', feature_loss, global_step=iteration)
            summary_writer.add_scalar('rgb_loss', rgb_loss, global_step=iteration)
            summary_writer.add_scalar('dino_pos_loss', dino_pos_loss, global_step=iteration)
            summary_writer.add_scalar('dino_neg_loss', dino_neg_loss, global_step=iteration)
            summary_writer.add_scalar('relevancy_loss', relevancy_loss, global_step=iteration)
            summary_writer.add_scalar('nvs_relevancy_loss', relevancy_loss, global_step=iteration)
                    
                    
        #############
        ###testing###
        #############
        if (iteration % (5000) == 0 or iteration == (args.n_iters-1)):
            with torch.no_grad():
                savePath = f'{logfolder}/imgs_vis'
                os.makedirs(savePath, exist_ok=True)
                IoUs, accuracies = [], []
                for i, frame_idx in tqdm(enumerate(feature_train_dataset.idxes)):
                    gt_seg = feature_train_dataset.seg_maps[i] # [H*W=N1, n_classes]

                    W, H = feature_train_dataset.img_wh
                    
                    # import pdb; pdb.set_trace()
                    if frame_idx in feature_train_dataset.img_list:
                        cur_idx = feature_train_dataset.img_list.index(frame_idx)
                        rays = feature_train_dataset.all_rays_stack[cur_idx].reshape(-1, 6)
                        rgb = feature_train_dataset.all_rgbs_stack[cur_idx].reshape(-1, 3)
                        
                        feature_map, _, _ = renderer(rays, tensorf, chunk=1024, N_samples=nSamples, 
                                            ndc_ray=ndc_ray, is_train=False, render_feature=True, device=device)

                        feature_map_normalized = F.normalize(feature_map, dim=1) # [N1,D]
                        relevancy_map = torch.mm(feature_map_normalized, text_features.T) # [N1,N2]

                        p_class = F.softmax(relevancy_map, dim=1) # [N1,N2]
                        class_index = torch.argmax(p_class, dim=-1).cpu() # [N1]
                        segmentation_map = vis_seg(dc, class_index, H, W, rgb=rgb, alpha=0.8)
                        
                        one_hot = F.one_hot(class_index.long(), num_classes=gt_seg.shape[-1]) # [N1, n_classes]
                        one_hot = one_hot.detach().cpu().numpy().astype(np.int8)
                        IoUs.append(jaccard_score(gt_seg, one_hot, average=None))
                        accuracies.append(accuracy_score(gt_seg, one_hot))

                        if savePath is not None:
                            imageio.imwrite(f'{savePath}/{iteration:05d}_{frame_idx:02d}.png', segmentation_map)

                # write IoUs to log file
                logger.info(f'\n\niteration: {iteration}')
                logger.info(f'overall: mIoU={np.mean(IoUs)}, accuracy={np.mean(accuracies)}\n')
                for i, iou in enumerate(IoUs):
                    logger.info(f'test image {i}: mIoU={np.mean(iou)}, accuracy={accuracies[i]}')
                    logger.info(f'classes iou: {iou}')
                    

    tensorf.save(f'{logfolder}/{args.expname}.th')

    
    
if __name__ == '__main__':

    torch.set_default_dtype(torch.float32)
    torch.manual_seed(20230417)
    np.random.seed(20230417)

    args = config_parser()

    if args.render_only:
        render_test(args)
    else:
        reconstruction(args)

