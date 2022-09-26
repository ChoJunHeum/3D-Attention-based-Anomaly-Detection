# Trainer and inference funcs
import time
import logging

import torch
import numpy as np

import data_utils
import crop_utils
import calc_scores

from tqdm import tqdm

logger = logging.getLogger()


def detect_nob(last_frame, yolo_model, cfg, confidence=.4, coi=[0]):
    '''
    No bactch allowed.
    Return:
        areas_filters: list of np.arrays of [xmin, ymin, xmax, ymax, conf, class]
    '''
    results = yolo_model(last_frame)
    areas = results.xyxy[0]
    if cfg.device != 'cpu':
        areas = areas.cpu().numpy()
    else:
        areas = areas.numpy()
    areas_filtered = [area for area in areas if area[4] > confidence and area[-1] in coi]        
    return areas_filtered


def prep_frame(input_frames, target_frame, areas, transform, cfg, no_patch=False):
    '''
    No bactch allowed. Prepare (Patches + Frame) sets.
    i.e., [ [t0_p0, t1_p0, ..., ] ]
    Prepare inputs and targets to run generator forward.
    If no_path, do not crop and resize the entire frames.
    Else, do crop and stack patches.
    
    Should apply this function to one target_frame,
    since it generates several patches for detected areas (if exists).

    Inputs:
        areas: detected areas for one frame (array)
        target_frame:
    Outputs:
        gen_inputs: stacked cropped patches in order to run in batch
                    (or) resized frames (if areas is None)
                        [B, C*nframe, resize_h, resize_w]
        gen_target: stacked cropped targets
                    (or) resized target (if areas is None)
                        [B, C, resize_h, resize_w]
    '''
    frames_all = input_frames + [target_frame]
    # frames_all_resized: [C, h, w]
    frames_all_resized = [data_utils.img2tensor(input_frame,
                            transform=transform,
                            symmetric=cfg.symmetric,
                            ) for input_frame in frames_all]
    # frame_inputs: [1, C, nframe, h, w], frame_target: [1, C, h, w]
    frame_inputs = torch.stack(frames_all_resized[:-1], dim=1).to(cfg.device).unsqueeze(0)
    frame_target = torch.as_tensor(frames_all_resized[-1], device=cfg.device).unsqueeze(0)
    if no_patch:
        gen_inputs = frame_inputs; gen_target = frame_target
    else:
        new_areas = crop_utils.get_resized_area(areas, max_w=cfg.max_w, max_h=cfg.max_h,
                                                factor_x=cfg.factor_x, factor_y=cfg.factor_y)
        # cropped_all = [ [t0_p0, t1_p0, ..., t4_p0], [t0_p1, ..., t4_p1], ... ]
        # t for timestamp and p for patch (area)
        cropped_all = crop_utils.crop_save(frames_all, new_areas, None, None, save=False)

        # Generate several patches in one frame at one batch
        gen_inputs = []; gen_target = []
        for one_area in cropped_all:
            # one_area_list[0]: [C, resize_h, resize_w]
            # len(one_area_list) = nframe + 1
            one_area_list = [data_utils.img2tensor(one_patch,
                                transform=transform,
                                symmetric=cfg.symmetric,
                                ) for one_patch in one_area]

            x_tmp = torch.stack(one_area_list[:-1], dim=1).to(cfg.device) # [C * nframe, h, w]
            y_tmp = torch.as_tensor(one_area_list[-1], device=cfg.device) # [C, h, w]
            gen_inputs.append(x_tmp); gen_target.append(y_tmp)
        # B: num of patches + 1 (1 for the whole frame)
        gen_inputs = torch.stack(gen_inputs, dim=0).contiguous().to(cfg.device) # [B + 1, C * nframe, h, w]
        gen_target = torch.stack(gen_target, dim=0).contiguous().to(cfg.device) # [B + 1, C, h, w]
        gen_inputs = torch.cat([gen_inputs, frame_inputs], dim=0)
        gen_target = torch.cat([gen_target, frame_target], dim=0)
        assert gen_inputs.size()[0] == len(cropped_all) + 1

    return gen_inputs, gen_target


def prep_patch(input_frames, target_frame, areas, transform, cfg, no_patch=False):
    '''
    Prepare inputs and targets to run generator forward.
    If no_path, do not crop and resize the entire frames.
    Else, do crop and stack patches.
    
    Should apply this function to one target_frame,
    since it generates several patches for detected areas (if exists).

    Inputs:
        areas: detected areas for one frame (array)
        target_frame:
    Outputs:
        gen_inputs: stacked cropped patches in order to run in batch
                    (or) resized frames (if areas is None)
                        [B, C*nframe, resize_h, resize_w]
        gen_target: stacked cropped targets
                    (or) resized target (if areas is None)
                        [B, C, resize_h, resize_w]
    '''
    if no_patch:
        all_frames = [data_utils.img2tensor(input_frame,
                        transform=transform,
                        symmetric=cfg.symmetric,
                        ) for input_frame in input_frames+[target_frame]]
        gen_inputs = torch.stack(all_frames[:-1], dim=1).to(cfg.device).unsqueeze(0)
        gen_target = torch.as_tensor(all_frames[-1], device=cfg.device).unsqueeze(0)
    else:
        new_areas = crop_utils.get_resized_area(areas, max_w=cfg.max_w, max_h=cfg.max_h,
                                                factor_x=cfg.factor_x, factor_y=cfg.factor_y)
        frames_all = input_frames + [target_frame]
        # cropped_all = [ [t0_p0, t1_p0, ..., t4_p0], [t0_p1, ..., t4_p1], ... ]
        # t for timestamp and p for patch (area)
        cropped_all = crop_utils.crop_save(frames_all, new_areas, None, None, save=False)

        # Generate several patches in one frame at one batch
        gen_inputs = []; gen_target = []
        for one_area in cropped_all:
            # one_area_list[0]: [3, resize_h, resize_w]
            # len(one_area_list) = nframe + 1
            one_area_list = [data_utils.img2tensor(one_patch,
                                transform=transform,
                                symmetric=cfg.symmetric,
                                ) for one_patch in one_area]

            x_tmp = torch.stack(one_area_list[:-1], dim=1).to(cfg.device) # [C, nframe, h, w]
            y_tmp = torch.as_tensor(one_area_list[-1], device=cfg.device) # [C, h, w]
            gen_inputs.append(x_tmp); gen_target.append(y_tmp)
        gen_inputs = torch.stack(gen_inputs, dim=0).contiguous().to(cfg.device) # [B, C, nframe, h, w]
        gen_target = torch.stack(gen_target, dim=0).contiguous().to(cfg.device) # [B, C, h, w]
    return gen_inputs, gen_target


def _calc_area(area):
    dw = area[2] - area[0]
    dh = area[3] - area[1]
    s = dw * dh
    return s


def calc_weight(areas, squash_func, baseline=None, gamma=.3):
    sizes = [_calc_area(area) for area in areas]
    sizes = squash_func(sizes)
    if baseline is None:
        baseline = np.max(sizes)
    weights = (1-sizes/squash_func(baseline)) * gamma + 1
    return weights, sizes


def step_train(inputs, targets,
                models, losses,
                opts, schedulers,
                cfg, epoch, global_step=None):
    '''
    inputs: [B, C * nframe, H, W]
    '''
    nframe = cfg.nframe
    nc = 3 # channels for image = 3
    generator, discriminator, flownet = models
    opt_g, opt_d = opts
    sch_g, sch_d = schedulers
        
    generator.train(); discriminator.train()
    assert not flownet.training

    # forward
    inputs_forward = inputs
    target_forward = targets
    loss_g_forward, loss_d_forward, pred_forward = one_direction(inputs_forward,
                                                                target_forward,
                                                                models, losses,
                                                                cfg, 'forward',
                                                                global_step)
    # backward
    splited = torch.split(inputs, 1, dim=2)
    assert len(splited) == nframe
    t0, t1, t2, t3 = splited
    inputs_backward = torch.cat([pred_forward.detach().unsqueeze(2), t3, t2, t1], dim=2).contiguous()
    target_backward = t0.squeeze(2)
    assert inputs_backward.size() == inputs_forward.size()
    loss_g_backward, loss_d_backward, pred_backward = one_direction(inputs_backward,
                                                                    target_backward,
                                                                    models, losses,
                                                                    cfg, 'backward',
                                                                    global_step)
    loss_g_tot = loss_g_forward + loss_g_backward
    loss_d_tot = loss_d_forward + loss_d_backward
    
    opt_g.zero_grad()
    loss_g_tot.backward()
    opt_g.step()

    opt_d.zero_grad()
    loss_d_tot.backward()
    opt_d.step()

    if sch_g is not None:
        sch_g.step()
        sch_d.step()

    if global_step % cfg.verbose == 0:
        logger.info(f'[Train-{epoch}-{global_step}] Total forward: {loss_g_tot.item():.2f} |')
        logger.info(f'[Train-{epoch}-{global_step}] Total backward: {loss_d_tot.item():.2f} |')

    return pred_forward


def one_direction(inputs, targets,
                    models, losses,
                    cfg, dicrection,
                    global_step=None):
    '''
    Return generator_loss, discriminator_loss, generated_frame
    '''
    coefs = cfg.lambdas
    nframe = cfg.nframe
    generator, discriminator, flownet = models
    adversarial_loss, discriminate_loss, \
        gradient_loss, intensity_loss, \
        flow_loss = losses

    # one direction
    # generator
    preds = generator(inputs)
    inte_l = intensity_loss(preds, targets)
    grad_l = gradient_loss(preds, targets)
    adv_l = adversarial_loss(discriminator(preds))

    # flownet
    input_lasts = inputs[:, :, -1]
    input_lasts_ = set_range(input_lasts, cfg)
    targets_ = set_range(targets, cfg)
    preds_ = set_range(preds, cfg)
    if cfg.flownet == 'lite':
        gt_flow_input = torch.cat([input_lasts_, targets_], 1)
        pred_flow_input = torch.cat([input_lasts_, preds_], 1)
        
        flow_gt = flownet.batch_estimate(gt_flow_input, flownet).detach()
        flow_pred = flownet.batch_estimate(pred_flow_input, flownet).detach()
    elif cfg.flownet == '2sd':
        gt_flow_input = torch.cat([input_lasts_.unsqueeze(2), targets_.unsqueeze(2)], 2)
        pred_flow_input = torch.cat([input_lasts_.unsqueeze(2), preds_.unsqueeze(2)], 2)
        
        flow_gt = (flownet(gt_flow_input) / 255.).detach()
        flow_pred = (flownet(pred_flow_input) / 255.).detach()
    else:
        NotImplementedError

    flow_l = flow_loss(flow_pred, flow_gt)

    loss_gen = coefs[0] * inte_l + \
                coefs[1] * grad_l + \
                coefs[2] * adv_l + \
                coefs[3] * flow_l
    # discriminator
    loss_dis = discriminate_loss(discriminator(targets),
                                discriminator(preds.detach()))
    if global_step % cfg.verbose == 0:
        logger.info(f'[Train-{global_step}] {dicrection} | gen: {loss_gen.item():.2f} |')
        logger.info(f' | inte_l: {inte_l.item():.2f}, grad_l: {grad_l.item():.2f}, adv_l: {adv_l.item():.2f}, flow_l: {flow_l.item():.2f}')
        logger.info(f'[Train-{global_step}] {dicrection} | dis: {loss_dis.item():.2f} |')

    return loss_gen, loss_dis, preds


def set_range(img_t, cfg):
    # input, target so far: (0 ~ 1) w/ normalization
    # 1. unnorm, 2. reset range
    # img_t: [B, C, h, w] or [B, C, D, h, w] 
    if len(img_t.size()) == 4:
        unnormed = img_t * data_utils.STD4 + data_utils.MEAN4
    elif len(img_t.size()) == 5:
        unnormed = img_t * data_utils.STD5 + data_utils.MEAN5
    else:
        raise NotImplementedError

    if cfg.flownet == 'lite':
        unnormed = unnormed * 2 - 1 # input for lite: [-1, 1]
    else:
        unnormed *= 255             # input for flownet2sd: [0, 255]
    return unnormed


def infer(generator, yolo_model,
            test_dirs, labels,
            transform, cfg, smooth=True):
    '''
    return dict(
        'final_auc': auc,
        'total_auc': total_auc
        'elapsed': elapsed)
    '''
    assert not yolo_model.training
    generator.eval()
    st = time.time()
    total_scores = []; total_labels = []; total_auc = []
    with torch.no_grad():
        for folder_idx, folder in enumerate(tqdm(test_dirs)):
            folder_dataset = data_utils.TestDataset(cfg, folder)
            psnrs_min_folder = []; sizes_min_folder = []
            
            for frame_idx, frames in enumerate(folder_dataset):
                input_frames, target_frame = frames
                last_frame = input_frames[-1]

                areas = detect_nob(last_frame=last_frame,
                                        yolo_model=yolo_model,
                                        cfg=cfg,
                                        confidence=cfg.confidence,
                                        coi=[0])
                
                no_patch = False if len(areas) != 0 else True
                gen_inputs, gen_target = prep_frame(input_frames=input_frames,
                                                        target_frame=target_frame,
                                                        areas=areas,
                                                        transform=transform,
                                                        cfg=cfg,
                                                        no_patch=no_patch)                                  
                gen_pred = generator(gen_inputs).detach()
                
                gen_pred = gen_pred * data_utils.STD4 + data_utils.MEAN4
                gen_target = gen_target * data_utils.STD4 + data_utils.MEAN4
                psnr_frame = calc_scores.psnr_error(gen_pred, gen_target, reduce_mean=False).cpu().numpy()
                if no_patch:
                    assert psnr_frame.shape[0] == 1 # one element for the entire frame
                    size_norm = np.array(1.).reshape(-1)
                    size_unnorm = np.sqrt(cfg.max_h*cfg.max_w).reshape(-1)

                else:
                    size_norm, size_unnorm = calc_weight(areas, np.sqrt,
                                                            baseline=cfg.max_h*cfg.max_w,
                                                            gamma=cfg.gamma)
                    size_norm = np.concatenate([size_norm, np.array(1.).reshape(-1)])
                    size_unnorm = np.concatenate([size_unnorm, np.sqrt(cfg.max_h*cfg.max_w).reshape(-1)])
                psnr_weighted = psnr_frame * size_norm
                # psnr_weighted = psnr_frame

                psnr_min = np.min(psnr_weighted)
                size_min = size_unnorm[np.argmin(psnr_weighted)]

                sizes_min_folder.append(size_min)
                psnrs_min_folder.append(psnr_min)
            labels_folder = labels[folder_idx][4:]
            total_labels.append(labels_folder)
            scores_folder = calc_scores.norm_scores(psnrs_min_folder, sizes=None)
            if smooth:
                scores_folder = calc_scores.gaussian_smooth(scores_folder)
            total_scores.append(scores_folder)
            auc_folder = calc_scores.calc_auc(scores_folder, labels_folder, pad=True)
            total_auc.append(auc_folder)
    
    auc = calc_scores.calc_auc(total_scores, total_labels, pad=True)
    elapsed = time.time() - st
    return dict(auc=auc, total_auc=total_auc, elapsed=elapsed)



if __name__=="__main__":
    
    areas = [[0, 0, 10, 10],[0, 0, 10, 20],[0, 0, 15, 20],[0, 0, 20, 20],[0, 0, 20, 30],]

    w, s = calc_weight(areas, np.sqrt, 20*30, 0.3)

    print(w, s)
