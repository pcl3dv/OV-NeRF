import torch,os,imageio,sys
from tqdm.auto import tqdm
from dataLoader.ray_utils import get_rays
from models.tensoRF import raw2alpha, TensorVMSplit, AlphaGridMask
from funcs import *
from dataLoader.ray_utils import ndc_rays_blender
from sklearn.decomposition import PCA
from sklearn.metrics import jaccard_score, accuracy_score
from torchvision.utils import save_image
import clip
import open_clip
import time


def OctreeRender_trilinear_fast(rays, tensorf, chunk=4096, N_samples=-1, ndc_ray=False, white_bg=True, is_train=False, 
                                render_feature=False, out_rgb=False, device='cuda'):

    rgbs, alphas, depth_maps, weights, uncertainties = [], [], [], [], []
    features, selects = [], []

    N_rays_all = rays.shape[0]
    for chunk_idx in range(N_rays_all // chunk + int(N_rays_all % chunk > 0)):
        rays_chunk = rays[chunk_idx * chunk:(chunk_idx + 1) * chunk].to(device)

        if render_feature:
            feature_map, select_map, rgb_map = tensorf.render_feature_map(rays_chunk, 
                                                                          out_rgb=out_rgb, 
                                                                          is_train=is_train, 
                                                                          ndc_ray=ndc_ray, 
                                                                          N_samples=N_samples)
            features.append(feature_map)
            selects.append(select_map)
            rgbs.append(rgb_map)

        else:
            rgb_map, depth_map = tensorf(rays_chunk, is_train=is_train, white_bg=white_bg, ndc_ray=ndc_ray, N_samples=N_samples)
            rgbs.append(rgb_map)
            depth_maps.append(depth_map)
    
    if render_feature: 
        ret_rgbs = None if rgbs[0] is None else torch.cat(rgbs)
        return torch.cat(features), torch.cat(selects), ret_rgbs

    return torch.cat(rgbs), None, torch.cat(depth_maps), None, None


def OctreeRender_trilinear_fast_depth(rays, tensorf, chunk=4096, N_samples=-1, ndc_ray=False, is_train=False, device='cuda'):

    depth_maps = []
    N_rays_all = rays.shape[0]
    for chunk_idx in range(N_rays_all // chunk + int(N_rays_all % chunk > 0)):
        rays_chunk = rays[chunk_idx * chunk:(chunk_idx + 1) * chunk].to(device)
        depth_map = tensorf.render_depth_map(rays_chunk, is_train=is_train, ndc_ray=ndc_ray, N_samples=N_samples)
        depth_maps.append(depth_map)
    
    return torch.cat(depth_maps)


prompt_templates = ['a bad photo of a {}.', 'a photo of many {}.', 'a sculpture of a {}.', 'a photo of the hard to see {}.', 'a low resolution photo of the {}.', 'a rendering of a {}.', 'graffiti of a {}.', 'a bad photo of the {}.', 'a cropped photo of the {}.', 'a tattoo of a {}.', 'the embroidered {}.', 'a photo of a hard to see {}.', 'a bright photo of a {}.', 'a photo of a clean {}.', 'a photo of a dirty {}.', 'a dark photo of the {}.', 'a drawing of a {}.', 'a photo of my {}.', 'the plastic {}.', 'a photo of the cool {}.', 'a close-up photo of a {}.', 'a black and white photo of the {}.', 'a painting of the {}.', 'a painting of a {}.', 'a pixelated photo of the {}.', 'a sculpture of the {}.', 'a bright photo of the {}.', 'a cropped photo of a {}.', 'a plastic {}.', 'a photo of the dirty {}.', 'a jpeg corrupted photo of a {}.', 'a blurry photo of the {}.', 'a photo of the {}.', 'a good photo of the {}.', 'a rendering of the {}.', 'a {} in a video game.', 'a photo of one {}.', 'a doodle of a {}.', 'a close-up photo of the {}.', 'a photo of a {}.', 'the origami {}.', 'the {} in a video game.', 'a sketch of a {}.', 'a doodle of the {}.', 'a origami {}.', 'a low resolution photo of a {}.', 'the toy {}.', 'a rendition of the {}.', 'a photo of the clean {}.', 'a photo of a large {}.', 'a rendition of a {}.', 'a photo of a nice {}.', 'a photo of a weird {}.', 'a blurry photo of a {}.', 'a cartoon {}.', 'art of a {}.', 'a sketch of the {}.', 'a embroidered {}.', 'a pixelated photo of a {}.', 'itap of the {}.', 'a jpeg corrupted photo of the {}.', 'a good photo of a {}.', 'a plushie {}.', 'a photo of the nice {}.', 'a photo of the small {}.', 'a photo of the weird {}.', 'the cartoon {}.', 'art of the {}.', 'a drawing of the {}.', 'a photo of the large {}.', 'a black and white photo of a {}.', 'the plushie {}.', 'a dark photo of a {}.', 'itap of a {}.', 'graffiti of the {}.', 'a toy {}.', 'itap of my {}.', 'a photo of a cool {}.', 'a photo of a small {}.', 'a tattoo of the {}.', 'there is a {} in the scene.', 'there is the {} in the scene.', 'this is a {} in the scene.', 'this is the {} in the scene.', 'this is one {} in the scene.',]

prompt_templates_3dovs = ['a photo of a {}.']

    
def zeroshot_CLIP_classifier(model_name, classnames, templates):
    clip_model, preprocess = clip.load(model_name)

    with torch.no_grad():
        zeroshot_weights = []
        for classname in classnames:
            texts = [template.format(classname) for template in templates] # format with class
            texts = clip.tokenize(texts).cuda()                         # tokenize
            class_embeddings = clip_model.encode_text(texts).float()    # embed with text encoder
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)              # average multi templates
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=0).cuda()

    del clip_model
    return zeroshot_weights

def zeroshot_OpenCLIP_classifier(classnames, templates):
    clip_model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-16', pretrained='laion2b_s34b_b88k')  
    clip_model = clip_model.to("cuda")

    with torch.no_grad():
        zeroshot_weights = []
        for classname in classnames:
            texts = [template.format(classname) for template in templates] # format with class
            texts = clip.tokenize(texts).cuda()                         # tokenize
            class_embeddings = clip_model.encode_text(texts).float()    # embed with text encoder
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)              # average multi templates
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=0).cuda()

    del clip_model
    return zeroshot_weights


@torch.no_grad()
def evaluation(test_dataset,tensorf, args, renderer, savePath=None, N_vis=5, prtx='', N_samples=-1,
               white_bg=False, ndc_ray=False, compute_extra_metrics=True, device='cuda'):
    PSNRs, rgb_maps, depth_maps = [], [], []
    ssims, l_alex, l_vgg=[],[],[]
    os.makedirs(savePath, exist_ok=True)
    os.makedirs(savePath+"/rgbd", exist_ok=True)

    try:
        tqdm._instances.clear()
    except Exception:
        pass

    near_far = test_dataset.near_far
    img_eval_interval = 1 if N_vis < 0 else max(test_dataset.all_rays_stack.shape[0] // N_vis,1)
    idxs = list(range(0, test_dataset.all_rays_stack.shape[0], img_eval_interval))
    
    for idx, samples in tqdm(enumerate(test_dataset.all_rays_stack[0::img_eval_interval]), file=sys.stdout):
        cur_image_name = test_dataset.img_path_l[idx].split("/")[-1][:-4]
        
        W, H = test_dataset.img_wh
        rays = samples.view(-1, samples.shape[-1])

        rgb_map, _, depth_map, _, _ = renderer(rays, tensorf, chunk=4096, N_samples=N_samples,
                                        ndc_ray=ndc_ray, white_bg = white_bg, device=device)
        rgb_map = rgb_map.clamp(0.0, 1.0)

        rgb_map, depth_map = rgb_map.reshape(H, W, 3).cpu(), depth_map.reshape(H, W).cpu()

        depth_map, _ = visualize_depth_numpy(depth_map.numpy(),near_far)
        if len(test_dataset.all_rgbs_stack):
            gt_rgb = test_dataset.all_rgbs_stack[idxs[idx]].view(H, W, 3)
            loss = torch.mean((rgb_map - gt_rgb) ** 2)
            PSNRs.append(-10.0 * np.log(loss.item()) / np.log(10.0))

            if compute_extra_metrics:
                ssim = rgb_ssim(rgb_map, gt_rgb, 1)
                l_a = rgb_lpips(gt_rgb.numpy(), rgb_map.numpy(), 'alex', tensorf.device)
                l_v = rgb_lpips(gt_rgb.numpy(), rgb_map.numpy(), 'vgg', tensorf.device)
                ssims.append(ssim)
                l_alex.append(l_a)
                l_vgg.append(l_v)

        rgb_map = (rgb_map.numpy() * 255).astype('uint8')
        rgb_maps.append(rgb_map)
        depth_maps.append(depth_map)
        if savePath is not None:
            imageio.imwrite(f'{savePath}/{prtx}{cur_image_name}.png', rgb_map)
            imageio.imwrite(f'{savePath}/rgbd/{prtx}{cur_image_name}.png', depth_map)

    imageio.mimwrite(f'{savePath}/{prtx}video.mp4', np.stack(rgb_maps), fps=30, quality=10)
    imageio.mimwrite(f'{savePath}/{prtx}depthvideo.mp4', np.stack(depth_maps), fps=30, quality=10)

    if PSNRs:
        psnr = np.mean(np.asarray(PSNRs))
        if compute_extra_metrics:
            ssim = np.mean(np.asarray(ssims))
            l_a = np.mean(np.asarray(l_alex))
            l_v = np.mean(np.asarray(l_vgg))
            np.savetxt(f'{savePath}/{prtx}mean.txt', np.asarray([psnr, ssim, l_a, l_v]), fmt='%.04f')
            
            with open(f'{savePath}/{prtx}PerView.txt', 'a+') as f:
                for cur_id in range(len(PSNRs)):
                    f.write(f'Idx={cur_image_name}, PSNR={PSNRs[cur_id]}, SSIM={ssims[cur_id]}, LPIPS_Alex={l_alex[cur_id]}, LPIPS_Vgg={l_vgg[cur_id]},\n\n')
        else:
            np.savetxt(f'{savePath}/{prtx}mean.txt', np.asarray([psnr]), fmt='%.04f')
            with open(f'{savePath}/{prtx}PerView.txt', 'a+') as f:
                for cur_id in range(len(PSNRs)):
                    f.write(f'PSNR={PSNRs[cur_id]}\n\n') 


    return PSNRs


@torch.no_grad()
def evaluation_path(test_dataset, tensorf, c2ws, renderer, savePath=None, N_vis=5, prtx='', N_samples=-1,
                    white_bg=False, ndc_ray=False, compute_extra_metrics=True, device='cuda'):
    PSNRs, rgb_maps, depth_maps = [], [], []
    ssims,l_alex,l_vgg=[],[],[]
    os.makedirs(savePath, exist_ok=True)
    os.makedirs(savePath+"/rgbd", exist_ok=True)

    try:
        tqdm._instances.clear()
    except Exception:
        pass

    near_far = test_dataset.near_far
    for idx, c2w in tqdm(enumerate(c2ws)):
        
        W, H = test_dataset.img_wh

        c2w = torch.FloatTensor(c2w)
        rays_o, rays_d = get_rays(test_dataset.directions, c2w)  # both (h*w, 3)
        
        if ndc_ray:
            rays_o, rays_d = ndc_rays_blender(H, W, test_dataset.focal[0], 1.0, rays_o, rays_d)

        rays = torch.cat([rays_o, rays_d], 1)  # (h*w, 6)

        rgb_map, _, depth_map, _, _ = renderer(rays, tensorf, chunk=8192, N_samples=N_samples,
                                        ndc_ray=ndc_ray, white_bg = white_bg, device=device)
        rgb_map = rgb_map.clamp(0.0, 1.0)

        rgb_map, depth_map = rgb_map.reshape(H, W, 3).cpu(), depth_map.reshape(H, W).cpu()

        depth_map, _ = visualize_depth_numpy(depth_map.numpy(),near_far)

        rgb_map = (rgb_map.numpy() * 255).astype('uint8')
        rgb_maps.append(rgb_map)
        depth_maps.append(depth_map)
        if savePath is not None:
            imageio.imwrite(f'{savePath}/{prtx}{idx:04d}.png', rgb_map)
            rgb_map = np.concatenate((rgb_map, depth_map), axis=1)
            imageio.imwrite(f'{savePath}/rgbd/{prtx}{idx:04d}.png', rgb_map)

    imageio.mimwrite(f'{savePath}/{prtx}video.mp4', np.stack(rgb_maps), fps=30, quality=8)
    imageio.mimwrite(f'{savePath}/{prtx}depthvideo.mp4', np.stack(depth_maps), fps=30, quality=8)

    if PSNRs:
        psnr = np.mean(np.asarray(PSNRs))
        if compute_extra_metrics:
            ssim = np.mean(np.asarray(ssims))
            l_a = np.mean(np.asarray(l_alex))
            l_v = np.mean(np.asarray(l_vgg))
            np.savetxt(f'{savePath}/{prtx}mean.txt', np.asarray([psnr, ssim, l_a, l_v]), fmt='%.04f')
        else:
            np.savetxt(f'{savePath}/{prtx}mean.txt', np.asarray([psnr]), fmt='%.04f')


    return PSNRs


@torch.no_grad()
def evaluation_feature_pca_path(test_dataset,tensorf, c2ws, renderer, savePath=None, N_vis=5, prtx='', N_samples=-1,
                        ndc_ray=False,  device='cuda'):
    feature_maps = []
    os.makedirs(savePath, exist_ok=True)

    try:
        tqdm._instances.clear()
    except Exception:
        pass

    for idx, c2w in tqdm(enumerate(c2ws)):

        W, H = test_dataset.img_wh

        c2w = torch.FloatTensor(c2w)
        rays_o, rays_d = get_rays(test_dataset.directions, c2w)  # both (h*w, 3)
        if ndc_ray:
            rays_o, rays_d = ndc_rays_blender(H, W, test_dataset.focal[0], 1.0, rays_o, rays_d)
        rays = torch.cat([rays_o, rays_d], 1)  # (h*w, 6)

        feature_map, _, _ = renderer(rays, tensorf, chunk=4096, N_samples=N_samples, 
                            ndc_ray=ndc_ray, is_train=False, render_feature=True, device=device)
        
        feature_map = feature_map.squeeze().detach().cpu().numpy() 

        pca = PCA(n_components=3)

        component = pca.fit_transform(feature_map)
        component = component.reshape(H, W, 3)
        component = ((component - component.min()) / (component.max() - component.min())).astype(np.float32)
        component *= 255.
        component = component.astype(np.uint8)
        
        feature_maps.append(component)
        if savePath is not None:
            imageio.imwrite(f'{savePath}/{prtx}{idx:03d}.png', component)

    imageio.mimwrite(f'{savePath}/{prtx}video.mp4', np.stack(feature_maps), fps=30, quality=8)

    
@torch.no_grad()
def evaluation_feature_text_activation_path(test_dataset,tensorf, c2ws, renderer, savePath=None, N_vis=5, prtx='', N_samples=-1,
                        ndc_ray=False, text='', device='cuda'):
    activation_maps = []
    os.makedirs(savePath, exist_ok=True)

    model, _ = clip.load("ViT-B/16", device=device)
    text = clip.tokenize([text]).to(device)
    text_feature = model.encode_text(text)
    del model

    try:
        tqdm._instances.clear()
    except Exception:
        pass

    for idx, c2w in tqdm(enumerate(c2ws)):

        W, H = test_dataset.img_wh

        c2w = torch.FloatTensor(c2w)
        rays_o, rays_d = get_rays(test_dataset.directions, c2w)  # both (h*w, 3)
        if ndc_ray:
            rays_o, rays_d = ndc_rays_blender(H, W, test_dataset.focal[0], 1.0, rays_o, rays_d)
        rays = torch.cat([rays_o, rays_d], 1)  # (h*w, 6)

        feature_map, _, _ = renderer(rays, tensorf, chunk=4096, N_samples=N_samples, 
                            ndc_ray=ndc_ray, is_train=False, render_feature=True, device=device)
        
        feature_map = feature_map.reshape(H, W, -1).permute(2,0,1)[None,...]

        activation_map = F.cosine_similarity(feature_map, text_feature[:,:,None,None], dim=1)
        activation_map = (activation_map - activation_map.min()) / (activation_map.max() - activation_map.min()) # normalize [-1,1] to [0,1]
        activation_map = activation_map.permute(1,2,0).squeeze().detach().cpu().numpy().astype(np.float32)
        activation_map *= 255.
        activation_map = activation_map.astype(np.uint8)
 
        activation_maps.append(activation_map)
        if savePath is not None:
            imageio.imwrite(f'{savePath}/{prtx}{idx:03d}.png', activation_map)

    imageio.mimwrite(f'{savePath}/{prtx}video.mp4', np.stack(activation_maps), fps=30, quality=8)

    
@torch.no_grad()
def evaluation_select_path(test_dataset,tensorf, c2ws, renderer, savePath=None, N_vis=5, prtx='', N_samples=-1,
                        ndc_ray=False,  device='cuda'):
    select_maps = []
    os.makedirs(savePath, exist_ok=True)

    try:
        tqdm._instances.clear()
    except Exception:
        pass

    for idx, c2w in tqdm(enumerate(c2ws)):

        W, H = test_dataset.img_wh

        c2w = torch.FloatTensor(c2w)
        rays_o, rays_d = get_rays(test_dataset.directions, c2w)  # both (h*w, 3)
        if ndc_ray:
            rays_o, rays_d = ndc_rays_blender(H, W, test_dataset.focal[0], 1.0, rays_o, rays_d)
        rays = torch.cat([rays_o, rays_d], 1)  # (h*w, 6)

        _, select_map, _ = renderer(rays, tensorf, chunk=4096, N_samples=N_samples, 
                            ndc_ray=ndc_ray, is_train=False, render_feature=True, device=device)
        
        select_map = select_map.reshape(H, W, 3).cpu().numpy().astype(np.float32)
        select_map *= 255.
        select_map = select_map.astype(np.uint8)
        
        select_maps.append(select_map)
        if savePath is not None:
            imageio.imwrite(f'{savePath}/{prtx}{idx:03d}.png', select_map)

    imageio.mimwrite(f'{savePath}/{prtx}video.mp4', np.stack(select_maps), fps=30, quality=8)

    
@torch.no_grad()
def evaluation_segmentation_path(test_dataset, tensorf, c2ws, renderer, savePath=None, N_vis=5, prtx='', N_samples=-1,
                        ndc_ray=False, classes=None, device='cuda'):
    segmentation_maps = []
    os.makedirs(savePath, exist_ok=True)
    
    model, _ = clip.load("ViT-B/16", device=device)
    classes.sort()
    text_features = model.encode_text(clip.tokenize(classes).to(device)).float()
    del model
    
    # init color
    dc = DistinctColors()

    # save color of every class
    logger.info(classes)
    labels = np.ones((1, 64, 64)).astype(int)
    all_labels = []
    for i in range(len(classes)):
        all_labels.append(labels * i)
    all_labels = np.concatenate(all_labels, 0)
    shape = all_labels.shape
    labels_colored = dc.get_color_fast_numpy(all_labels.reshape(-1))
    labels_colored = (labels_colored.reshape(shape[0] * shape[1], shape[2], 3) * 255).astype(np.uint8)
    Image.fromarray(labels_colored).save(f"{savePath}/classes_color.png")


    try:
        tqdm._instances.clear()
    except Exception:
        pass
    
    # import pdb; pdb.set_trace()
    for idx, c2w in tqdm(enumerate(c2ws)):

        W, H = test_dataset.img_wh

        c2w = torch.FloatTensor(c2w)
        rays_o, rays_d = get_rays(test_dataset.directions, c2w)  # both (h*w, 3)
        if ndc_ray:
            rays_o, rays_d = ndc_rays_blender(H, W, test_dataset.focal[0], 1.0, rays_o, rays_d)
        rays = torch.cat([rays_o, rays_d], 1)  # (h*w, 6)

        feature_map, _, rgb = renderer(rays, tensorf, chunk=4096, N_samples=N_samples, 
                            ndc_ray=ndc_ray, is_train=False, render_feature=True, out_rgb=True, device=device)
        
        feature_map_normalized = F.normalize(feature_map, dim=1) # [N1, D] 
        text_features_normalized = F.normalize(text_features, dim=1) # [N2, D]
        relevancy_map = torch.mm(feature_map_normalized, text_features_normalized.T) # [N1,N2] 

        p_class = F.softmax(relevancy_map, dim=1) # [N1, N2]  
        class_index = torch.argmax(p_class, dim=-1).cpu() # [N1] 
        segmentation_map = vis_seg(dc, class_index, H, W, rgb=rgb.cpu())        
        
        # import pdb; pdb.set_trace()
        segmentation_maps.append(segmentation_map)
        if savePath is not None:
            imageio.imwrite(f'{savePath}/{prtx}{idx:03d}.png', segmentation_map)

    imageio.mimwrite(f'{savePath}/{prtx}video.mp4', np.stack(segmentation_maps), fps=30, quality=8)

    
    
    
  
def sam_refine_maskclip(masks, output_images_pred):
    
    sam_pred = []
    device = output_images_pred.device    
    sam_refined_pred = torch.zeros((output_images_pred.shape[1], output_images_pred.shape[2]), dtype=torch.long).to(device)

    for ann in masks:
        cur_mask = ann['segmentation']                      
        sub_mask = output_images_pred.squeeze().clone()     
        sub_mask[~cur_mask] = 0                             

        flat_sub_mask = sub_mask.clone().view(-1)           
        flat_sub_mask = flat_sub_mask[flat_sub_mask!=0]     

        if len(flat_sub_mask) != 0:                         
            unique_elements, counts = torch.unique(flat_sub_mask, return_counts=True)  
            second_most_common_element = unique_elements[int(counts.argmax().item())]  
        else:                                               
            continue 

        sam_refined_pred[cur_mask] = second_most_common_element  
        
    return sam_refined_pred



def compute_slic(cur_img_path, pred_tensor, model_sam):
    
    img_name = cur_img_path.split('/')[-1][:-4]   
    
    # sam input image 
    img_PIL = Image.open(cur_img_path).convert("RGB")
    img_array = np.array(img_PIL)                               
    
    # prediction 
    pred_tensor = pred_tensor.unsqueeze(0)

    # SAM
    output_masks = model_sam.generate(img_array)               # H*W*3 & numpy
    # print("==============>", img_name, " has been processed<==============")
    
    #import pdb; pdb.set_trace() 
    
    ## refinement
    refine_pred = sam_refine_maskclip(output_masks, pred_tensor)
    # print(refine_pred.type(), refine_pred, refine_pred.device)
    
    save_class_index = refine_pred.reshape(-1)   # tensor & cuda 
    save_clipsam_img = refine_pred.detach().cpu().numpy().astype(np.int8)  # numpy array & cpu 

    return save_class_index, save_clipsam_img



def compute_slic_with_loaded(output_masks, pred_tensor):
    
    ## refinement
    refine_pred = sam_refine_maskclip(output_masks, pred_tensor.unsqueeze(0))
    # print(refine_pred.type(), refine_pred, refine_pred.device)
    
    save_class_index = refine_pred.reshape(-1)   # tensor & cuda 
    save_clipsam_img = refine_pred.detach().cpu().numpy().astype(np.int8)  # numpy array & cpu 

    return save_class_index, save_clipsam_img



@torch.no_grad()
def evaluation_segmentation_train_with_SAM(test_dataset, tensorf, renderer, savePath=None, visPath=None, N_vis=5, prtx='', N_samples=-1, ndc_ray=False, classes=None, device='cuda', ScanNet_Convert=False):
    os.makedirs(savePath, exist_ok=True)
    os.makedirs(visPath, exist_ok=True)
    dc = DistinctColors()
    
    text_features = zeroshot_CLIP_classifier("ViT-B/16", classes, prompt_templates)
    print("Using text features from vanilla CLIP !")  
    # text_features = zeroshot_OpenCLIP_classifier(classes, prompt_templates)
    # print("Using text features from OpenCLIP !")   
        
        
    # init color for every class
    dc = DistinctColors()

    # save color of every class
    logger.info(classes)
    labels = np.ones((1, 64, 64)).astype(int)
    all_labels = []
    for i in range(len(classes)):
        all_labels.append(labels * i)
    all_labels = np.concatenate(all_labels, 0)
    shape = all_labels.shape
    labels_colored = dc.get_color_fast_numpy(all_labels.reshape(-1))
    labels_colored = (labels_colored.reshape(shape[0] * shape[1], shape[2], 3) * 255).astype(np.uint8)
    Image.fromarray(labels_colored).save(f"{savePath}/classes_color.png")
    
    # ### init SAM ### 
    # from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
    # sam_checkpoint = "Third_party/sam_space_checkpoint/sam_vit_h_4b8939.pth"
    # model_type = "vit_h" 
    # sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    # sam.to(device=device)                           
    # model_sam = SamAutomaticMaskGenerator(sam)
    
    try:
        tqdm._instances.clear()
    except Exception:
        pass
    
    update_all_clipsams = []
    
    IoUs, accuracies = [], []
    for frame_idx, img_seg in tqdm(enumerate(test_dataset.seg_path_l)):
        img_name = img_seg.split('/')[-1][:-4]   
        
        gt_seg = test_dataset.seg_maps_train[frame_idx] # [H*W=N1, n_classes] 
        
        gt_seg = gt_seg.astype(np.int8)
        W, H = test_dataset.img_wh
        rays = test_dataset.all_rays_stack[frame_idx].reshape(-1, 6)
        rgb = test_dataset.all_rgbs_stack[frame_idx].reshape(-1, 3)
        feature_map, _, _ = renderer(rays, tensorf, chunk=4096, N_samples=N_samples, 
                            ndc_ray=ndc_ray, is_train=False, render_feature=True, device=device)
        
        feature_map_normalized = F.normalize(feature_map, dim=1) # [N1,D]
        text_features_normalized = F.normalize(text_features, dim=1) # [N2,D]
        relevancy_map = torch.mm(feature_map_normalized, text_features_normalized.T) # [N1,N2]        
        p_class = F.softmax(relevancy_map, dim=1) # [N1,N2]
        class_index = torch.argmax(p_class, dim=-1) # [N1]
        pred_img = class_index.reshape(H, W)         # [H, W], cuda tensor 
        
        # import pdb; pdb.set_trace()
        
        ## Calculate the SAM mask online
        # sam_image_path = test_dataset.img_path_l[frame_idx]
        # update_class_index, update_pred_img = compute_slic(sam_image_path, pred_img, model_sam)      
        # update_all_clipsams.append(update_class_index.long())  # H*W, tensor, cuda 
        
        ## loaded the SAM mask offline
        cur_sam_mask = test_dataset.all_sammasks[frame_idx]   
        update_class_index, update_pred_img = compute_slic_with_loaded(cur_sam_mask, pred_img)      
        update_all_clipsams.append(update_class_index.long())  # H*W, tensor, cuda 
        segmentation_map = vis_seg(dc, update_class_index.cpu(), H, W, rgb=rgb, alpha=1)  
        
        acc, mIoU = measure_pa_miou(gt_seg.shape[-1], test_dataset.seg_maps_HW_train[frame_idx], update_pred_img)
        print("mIoU: ", '{:.3f}'.format(mIoU), "Pixel Accuracy:", '{:.3f}'.format(acc))
        IoUs.append(mIoU)
        accuracies.append(acc)
        
        if savePath is not None:
            cv2.imwrite(f'{savePath}/{prtx}{img_name}.png', update_pred_img)
        if visPath is not None:
            imageio.imwrite(f'{visPath}/{prtx}{img_name}.png', segmentation_map)

    # write IoUs and accuracies to file
    with open(f'{savePath}/{prtx}results.txt', 'a+') as f:
        f.write(f'classes: {classes}\n')
        f.write(f'overall: mIoU={np.mean(IoUs)}, accuracy={np.mean(accuracies)}\n\n')
        #for i, iou in enumerate(IoUs):
        #    f.write(f'test image {i}: mIoU={np.mean(iou)}, accuracy={accuracies[i]}\n')
            # f.write(f'classes iou: {iou}\n')  # for each class 
    
    print("Overall mIoU: ", '{:.3f}'.format(np.mean(IoUs)), "Overall Pixel Accuracy:", '{:.3f}'.format(np.mean(accuracies)))
    
    update_all_clipsams = torch.cat(update_all_clipsams, 0)  # (len(self.meta['frames'])*h*w,1)
    
    return update_all_clipsams 



def compute_slic_img(cur_img, pred_tensor, model_sam):
    # sam input image 
    img_array = np.array(cur_img)                               
    
    # prediction 
    pred_tensor = pred_tensor.unsqueeze(0)

    # SAM
    output_masks = model_sam.generate(img_array)               # H*W*3 & numpy
    
    ## refinement
    refine_pred = sam_refine_maskclip(output_masks, pred_tensor)
    
    ## save the image 
    save_class_index = refine_pred.reshape(-1)   # tensor & cuda 
    save_clipsam_img = refine_pred.detach().cpu().numpy().astype(np.int8)  # numpy array & cpu 
    
    # print("==============> The cur image has been processed<==============")
    return save_class_index, save_clipsam_img


@torch.no_grad()
def evaluation_segmentation_path_with_SAM(test_dataset, tensorf, c2ws, renderer, savePath=None, visPath=None, N_vis=5, prtx='', N_samples=-1,
                        ndc_ray=False, classes=None, device='cuda'):
    os.makedirs(savePath, exist_ok=True)
    os.makedirs(visPath, exist_ok=True)
    
    text_features = zeroshot_CLIP_classifier("ViT-B/16", classes, prompt_templates)
    print("Using text features from Vanilla CLIP !")     
    # text_features = zeroshot_OpenCLIP_classifier(classes, prompt_templates)
    # print("Using text features from OpenCLIP !")     
    
    # init color
    dc = DistinctColors()
    # save color of every class
    logger.info(classes)
    labels = np.ones((1, 64, 64)).astype(int)
    all_labels = []
    for i in range(len(classes)):
        all_labels.append(labels * i)
    all_labels = np.concatenate(all_labels, 0)
    shape = all_labels.shape
    labels_colored = dc.get_color_fast_numpy(all_labels.reshape(-1))
    labels_colored = (labels_colored.reshape(shape[0] * shape[1], shape[2], 3) * 255).astype(np.uint8)
    Image.fromarray(labels_colored).save(f"{savePath}/classes_color.png")
    try:
        tqdm._instances.clear()
    except Exception:
        pass
    
    ### init SAM ### 
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
    sam_checkpoint = "Third_party/sam_space_checkpoint/sam_vit_h_4b8939.pth"
    model_type = "vit_h" 
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)                                 
    model_sam = SamAutomaticMaskGenerator(sam)
    update_all_clipsams = []
    
    # import pdb; pdb.set_trace()
    for frame_idx, c2w in tqdm(enumerate(c2ws)):
        W, H = test_dataset.img_wh
        rays = test_dataset.all_nvs_all_rays[frame_idx].reshape(-1, 6)  
        feature_map, _, rgb = renderer(rays, tensorf, chunk=4096, N_samples=N_samples, 
                            ndc_ray=ndc_ray, is_train=False, render_feature=True, out_rgb=True, device=device)
        
        feature_map_normalized = F.normalize(feature_map, dim=1) # [N1,D]
        text_features_normalized = F.normalize(text_features, dim=1) # [N2,D]
        relevancy_map = torch.mm(feature_map_normalized, text_features_normalized.T) # [N1,N2]        
        p_class = F.softmax(relevancy_map, dim=1) # [N1,N2]
        class_index = torch.argmax(p_class, dim=-1) # [N1]
        pred_img = class_index.reshape(H, W)         # [H, W], cuda tensor 
        
        # import pdb; pdb.set_trace()
        # sam_image_input: (H, W, 3) || rgb: torch.Size([307200, 3]) tensor => 
        rgb_map = rgb.clamp(0.0, 1.0).reshape(H, W, 3).cpu()
        rgb_map = (rgb_map.numpy() * 255).astype('uint8')
        update_class_index, update_pred_img = compute_slic_img(rgb_map, pred_img, model_sam)      
        update_all_clipsams.append(update_class_index.long())  # H*W, tensor, cuda 
        segmentation_map = vis_seg(dc, update_class_index.cpu(), H, W, rgb=None, alpha=1)  # larger for more seg weight      
        
        if savePath is not None:
            imageio.imwrite(f'{savePath}/{prtx}{frame_idx:04d}.png', rgb_map)
        if visPath is not None:
            imageio.imwrite(f'{visPath}/{prtx}{frame_idx:04d}.png', segmentation_map)

    update_all_clipsams = torch.cat(update_all_clipsams, 0)  # (len(self.meta['frames'])*h*w,1)
    return update_all_clipsams         



@torch.no_grad()
def evaluation_segmentation_train(test_dataset, tensorf, renderer, savePath=None, visPath=None, N_vis=5, prtx='', N_samples=-1,
                        ndc_ray=False, classes=None, device='cuda'):
    os.makedirs(savePath, exist_ok=True)
    os.makedirs(visPath, exist_ok=True)

    # model, _ = clip.load("ViT-B/16", device=device)
    # classes_token = clip.tokenize(classes).to(device)
    # text_features = model.encode_text(classes_token).float()
    # del model
    
    text_features = zeroshot_CLIP_classifier("ViT-B/16", classes, prompt_templates)
    # text_features = zeroshot_OpenCLIP_classifier(classes, prompt_templates)
    

    # init color for every class
    dc = DistinctColors()
    
    
    # save color of every class
    dc = DistinctColors()
    labels = np.ones((1, 256, 256)).astype(int)
    all_labels = []
    for i in range(len(classes)):
        all_labels.append(labels * i)
    all_labels = np.concatenate(all_labels, 0)
    shape = all_labels.shape
    labels_colored = dc.get_color_fast_numpy(all_labels.reshape(-1))
    labels_colored = (labels_colored.reshape(shape[0] * shape[1], shape[2], 3) * 255).astype(np.uint8)
    Image.fromarray(labels_colored).save(f"{savePath}/classes_color.png")


    try:
        tqdm._instances.clear()
    except Exception:
        pass
    
    IoUs, accuracies = [], []
    for frame_idx, img_seg in tqdm(enumerate(test_dataset.seg_path_l)):
        img_name = img_seg.split('/')[-1][:-4]   
        
        # gt_seg = test_dataset.seg_maps[frame_idx] # [H*W=N1, n_classes] 
        # gt_seg = gt_seg.astype(np.int8)
        W, H = test_dataset.img_wh
        rays = test_dataset.all_rays_stack[frame_idx].reshape(-1, 6)
        rgb = test_dataset.all_rgbs_stack[frame_idx].reshape(-1, 3)
        feature_map, _, _ = renderer(rays, tensorf, chunk=4096, N_samples=N_samples, 
                            ndc_ray=ndc_ray, is_train=False, render_feature=True, device=device)
        
        feature_map_normalized = F.normalize(feature_map, dim=1) # [N1,D]
        text_features_normalized = F.normalize(text_features, dim=1) # [N2,D]
        relevancy_map = torch.mm(feature_map_normalized, text_features_normalized.T) # [N1,N2]        
        p_class = F.softmax(relevancy_map, dim=1) # [N1,N2]
        class_index = torch.argmax(p_class, dim=-1).cpu() # [N1]
        pred_img = class_index.reshape(H, W).detach().cpu().numpy().astype(np.int8)
        segmentation_map = vis_seg(dc, class_index, H, W, rgb=rgb, alpha=0.8)  # larger for more seg weight 
        
        # acc, mIoU = measure_pa_miou(gt_seg.shape[-1], test_dataset.seg_maps_HW[frame_idx], pred_img)
        # print("mIoU: ", '{:.3f}'.format(mIoU), "Pixel Accuracy:", '{:.3f}'.format(acc))
        # IoUs.append(mIoU)
        # accuracies.append(acc)
        
        if savePath is not None:
            cv2.imwrite(f'{savePath}/{prtx}{img_name}.png', pred_img)
        if visPath is not None:
            imageio.imwrite(f'{visPath}/{prtx}{img_name}.png', segmentation_map)

    # # write IoUs and accuracies to file
    # with open(f'{savePath}/{prtx}results.txt', 'w') as f:
    #     f.write(f'classes: {classes}\n')
    #     f.write(f'overall: mIoU={np.mean(IoUs)}, accuracy={np.mean(accuracies)}\n\n')
    #     for i, iou in enumerate(IoUs):
    #         f.write(f'test image {i}: mIoU={np.mean(iou)}, accuracy={accuracies[i]}\n')
    #         # f.write(f'classes iou: {iou}\n')  # for each class 
    
    # print("Overall mIoU: ", '{:.3f}'.format(np.mean(IoUs)), "Overall Pixel Accuracy:", '{:.3f}'.format(np.mean(accuracies)))


    
@torch.no_grad()
def evaluation_segmentation_test(test_dataset, tensorf, renderer, savePath=None, visPath=None, N_vis=5, prtx='', N_samples=-1,
                        ndc_ray=False, classes=None, device='cuda', ScanNet_Convert=False):
    os.makedirs(savePath, exist_ok=True)
    os.makedirs(visPath, exist_ok=True)
    dc = DistinctColors()
    
    # model, _ = clip.load("ViT-B/16", device=device)
    # classes_token = clip.tokenize(classes).to(device)
    # text_features = model.encode_text(classes_token).float()
    # del model
    
    text_features = zeroshot_CLIP_classifier("ViT-B/16", classes, prompt_templates)
    print("Using text features from vanilla CLIP !")
    # text_features = zeroshot_OpenCLIP_classifier(classes, prompt_templates)
    # print("Using text features from OpenCLIP !")     

    
    # init color for every class
    dc = DistinctColors()

    # save color of every class
    logger.info(classes)
    labels = np.ones((1, 64, 64)).astype(int)
    all_labels = []
    for i in range(len(classes)):
        all_labels.append(labels * i)
    all_labels = np.concatenate(all_labels, 0)
    shape = all_labels.shape
    labels_colored = dc.get_color_fast_numpy(all_labels.reshape(-1))
    labels_colored = (labels_colored.reshape(shape[0] * shape[1], shape[2], 3) * 255).astype(np.uint8)
    Image.fromarray(labels_colored).save(f"{visPath}/classes_color.png")

    try:
        tqdm._instances.clear()
    except Exception:
        pass
    
    IoUs, accuracies = [], []
    for frame_idx, img_seg in tqdm(enumerate(test_dataset.seg_path_l)):
        img_name = img_seg.split('/')[-1][:-4]   
        
        gt_seg = test_dataset.seg_maps[frame_idx] # [H*W=N1, n_classes] 
        
        gt_seg = gt_seg.astype(np.int8)
        W, H = test_dataset.img_wh
        rays = test_dataset.all_rays_stack[frame_idx].reshape(-1, 6)
        rgb = test_dataset.all_rgbs_stack[frame_idx].reshape(-1, 3)
        feature_map, _, _ = renderer(rays, tensorf, chunk=4096, N_samples=N_samples, 
                            ndc_ray=ndc_ray, is_train=False, render_feature=True, device=device)
        
        feature_map_normalized = F.normalize(feature_map, dim=1) # [N1,D]
        text_features_normalized = F.normalize(text_features, dim=1) # [N2,D]
        relevancy_map = torch.mm(feature_map_normalized, text_features_normalized.T) # [N1,N2]        
        p_class = F.softmax(relevancy_map, dim=1) # [N1,N2]
        class_index = torch.argmax(p_class, dim=-1).cpu() # [N1]
        pred_img = class_index.reshape(H, W).detach().cpu().numpy().astype(np.int8)
        segmentation_map = vis_seg(dc, class_index, H, W, rgb=rgb, alpha=1) 
        
        acc, mIoU = measure_pa_miou(gt_seg.shape[-1], test_dataset.seg_maps_HW[frame_idx], pred_img)
        print("mIoU: ", '{:.3f}'.format(mIoU), "Pixel Accuracy:", '{:.3f}'.format(acc))
        IoUs.append(mIoU)
        accuracies.append(acc)
        
        if savePath is not None:
            cv2.imwrite(f'{savePath}/{prtx}{img_name}.png', pred_img)
        if visPath is not None:
            imageio.imwrite(f'{visPath}/{prtx}{img_name}.png', segmentation_map)

    # write IoUs and accuracies to file
    with open(f'{savePath}/{prtx}results.txt', 'w') as f:
        f.write(f'classes: {classes}\n')
        f.write(f'overall: mIoU={np.mean(IoUs)}, accuracy={np.mean(accuracies)}\n\n')
        for i, iou in enumerate(IoUs):
            f.write(f'test image {i}: mIoU={np.mean(iou)}, accuracy={accuracies[i]}\n')
            # f.write(f'classes iou: {iou}\n')  # for each class 
    
    print("Overall mIoU: ", '{:.3f}'.format(np.mean(IoUs)), "Overall Pixel Accuracy:", '{:.3f}'.format(np.mean(accuracies)))

            
            
@torch.no_grad()
def evaluation_segmentation_depth(test_dataset, tensorf, renderer, savePath=None, N_vis=5, prtx='', N_samples=-1,
                        ndc_ray=False, classes=None, device='cuda'):
    os.makedirs(savePath, exist_ok=True)
    dc = DistinctColors()
    
    model, _ = clip.load("ViT-B/16", device=device)
    classes_token = clip.tokenize(classes).to(device)
    text_features = model.encode_text(classes_token).float()
    del model

    # init color for every class
    dc = DistinctColors()

    # save color of every class
    logger.info(classes)
    labels = np.ones((1, 64, 64)).astype(int)
    all_labels = []
    for i in range(len(classes)):
        all_labels.append(labels * i)
    all_labels = np.concatenate(all_labels, 0)
    shape = all_labels.shape
    labels_colored = dc.get_color_fast_numpy(all_labels.reshape(-1))
    labels_colored = (labels_colored.reshape(shape[0] * shape[1], shape[2], 3) * 255).astype(np.uint8)
    Image.fromarray(labels_colored).save(f"{savePath}/classes_color.png")

    try:
        tqdm._instances.clear()
    except Exception:
        pass

    W, H = test_dataset.img_wh

    i = 10
        
    rays = test_dataset.all_rays_stack[i].reshape(-1, 6)
    rgb = test_dataset.all_rgbs_stack[i].reshape(-1, 3)

    feature_map, _, _ = renderer(rays, tensorf, chunk=4096, N_samples=N_samples, 
                        ndc_ray=ndc_ray, is_train=False, render_feature=True, device=device)
    depth_map = OctreeRender_trilinear_fast_depth(rays, tensorf, chunk=4096, N_samples=N_samples,
                                                    ndc_ray=ndc_ray, is_train=False, device=device)
    
    # extract coordinates of points
    points_coordinates = construct_points_coordinates(rays.to(device), depth_map) # [N,3]
    points_coordinates = points_coordinates

    feature_map_normalized = F.normalize(feature_map, dim=1) # [N1,D]
    text_features_normalized = F.normalize(text_features, dim=1) # [N2,D]
    relevancy_map = torch.mm(feature_map_normalized, text_features_normalized.T) # [N1,N2]

    p_class = F.softmax(relevancy_map, dim=1) # [N1,N2]
    class_index = torch.argmax(p_class, dim=-1).cpu() # [N1]
    segmentation_map = dc.apply_colors_fast_torch(class_index)
    # segmentation_map = segmentation_map * alpha + rgb * (1 - alpha)
    segmentation_map = segmentation_map.reshape(-1, 3)
    
    save_points_to_ply(points_coordinates.cuda(), segmentation_map.cuda(), f'{savePath}/{prtx}seg{i}_points.ply')

    