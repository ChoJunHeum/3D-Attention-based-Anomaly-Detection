import os
from os.path import join

import glob
from stat import FILE_ATTRIBUTE_TEMPORARY
from PIL import Image


def set_dirs(outpath, nframe, prefix='t'):
    '''
    if nframe=4, mkdirs outpath/PREFIX0, outpath/PREFIX1, ..., outpath/PREFIX4.
    0 to nframe is generated in the chronological order.
    The last directory is for the target frame.
    '''
    dirs_to_gen = [join(outpath, prefix+str(postfix)) for postfix in list(range(nframe+1))]
    _ = [os.makedirs(dir_, exist_ok=True) for dir_ in dirs_to_gen]
    return dirs_to_gen


def batch_detect(model, frame_d, confidence, device, coi=[0]):
    '''
    Detect batch inputs and filter the results
    Filter condition:
        1. Over the given confidence
        2. In the classes of interest (COI)
            coi should passes to args in the form of list
    '''

    results = model(frame_d)
    areas = results.xyxy # list of tensors (on cuda if cuda:0)

    inp_size = len(frame_d)
    if len(areas) != inp_size:
        raise Exception('Got not equal area length')

    if device != 'cpu':
        areas = [area.cpu().numpy() for area in areas]
    else:
        areas = [area.numpy() for area in areas]

    areas_filtered = []
    for one_frame in areas:
        conf_mask = one_frame[:, 4] > confidence        
        coi_mask = [cls_ in coi for cls_ in one_frame[:, -1]]
        mask = conf_mask  * coi_mask
        
        if len(mask) > 0:
            areas_filtered.append(one_frame[mask])

    return areas_filtered


def crop_save(frames, areas, outdirs, num_detect, save=True):
    '''
    res: [[cropped for area1], [cropped for area2], ...]
    for area in areas:
        for frame in frames:
            save(crop(frame, area))
    '''
    results = []
    for num_area, area in enumerate(areas):
        one_area = []
        for i, frame in enumerate(frames):
            cropped = frame.crop(area[:4])
            if save:
                fname = f'{num_detect}-{num_area:03}.jpg'
                cropped.save(join(outdirs[i], fname))
            else:
                one_area.append(cropped)
        results.append(one_area)
    if not save:
        return results


def get_resized_area(areas, max_w, max_h, factor_x=1.5, factor_y=1.2):
    '''
    Should run it for each batch (for one area set from one frame_t)
    >>> batch_areas = detect(model, batch_frame_t)
    >>> for i in range(batch_size):
    >>>     new_areas = get_resized_area(batch_areas[i])
    '''
    new_areas = []
    for area in areas:
        xmin_, ymin_, xmax_, ymax_, conf, cls_token = area

        xmin = xmin_ - (factor_x-1)*(xmax_-xmin_)
        ymin = ymin_ - (factor_y-1)*(ymax_-ymin_)
        xmax = xmax_ + (factor_x-1)*(xmax_-xmin_)
        ymax = ymax_ + (factor_y-1)*(ymax_-ymin_)

        x = xmax - xmin
        y = ymax - ymin

        if y > x:
            dif = (y-x)/2
            xmax += dif
            xmin -= dif
        elif y < x:
            dif = (x-y)/2
            ymax += dif
            ymin -= dif
            
        if xmin < 0:
            xmin = 0

        if ymin < 0:
            ymin = 0

        if(xmax > max_w):
            xmax = max_w

        if(ymax > max_h):
            ymax = max_h

        new_areas.append([xmin, ymin, xmax, ymax, conf, cls_token])

    return new_areas

def get_cropped_img(yolo, clips):

    croppeds = []

    for clip in clips:
        print(clip)



    return croppeds

class VideoDataset(object):
    '''
    Cannot perform yielding batches for all folders,
    since imgs in different folders are not continuous.
    '''
    def __init__(self, dpath, nframe, num_folder):
        self.num_folder = num_folder
        self.nframe = nframe
        self.all_fpath = glob.glob(f'{join(dpath, self.num_folder)}/*.jpg')
        self.all_fpath.sort()
        self.all_seq = list(range(len(self.all_fpath) - self.nframe))
        self.len = len(self.all_seq)
        self.idx = 0
    
    def get_batches(self, batch_size):
        while (self.idx < self.len):
            batch_indices = self.all_seq[self.idx : self.idx + batch_size]
            self.idx += batch_size
            
            batch_frame_t=[]
            batch_frames_oth=[]
            batch_frames_d=[]
            batch_num_detect=[]
            for start in batch_indices:
                frames_tmp = []
                for i in range(start, start + self.nframe + 1):
                    frames_tmp.append(Image.open(self.all_fpath[i]))
                    # frames_tmp.append(self.all_fpath[i])

                batch_frames_oth.append(frames_tmp[:-2])
                batch_frames_d.append(frames_tmp[-2])
                batch_frame_t.append(frames_tmp[-1])
                # detect img number = last (target) - 1
                batch_num_detect.append(f'{self.num_folder}-{i-1:04}')

            yield batch_num_detect, batch_frames_oth, batch_frames_d, batch_frame_t