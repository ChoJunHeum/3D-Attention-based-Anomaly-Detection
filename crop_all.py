import os
import math
import argparse

import torch
from tqdm import tqdm

import crop_utils
from crop_utils import VideoDataset, batch_detect

from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', '-bs', default=16, type=int)
parser.add_argument('--confidence', '-c', default=.4, type=float, required=True)
parser.add_argument('--ngpu', '-ng', default=0, type=int, help='gpu number')
args = parser.parse_args()


if __name__ == "__main__":
    '''
    Crop all frames based on BBox in the last input (every fourth) frame.
    Note that sliding window concept is used to get every fourth frame.
    i.e.,
        raw frames:     img0, img1, img2, ..., img4, img5, ...
        detect frames:  img3, img4, img5, ...
        target frames:  img4, img5, img6, ...
    We can use large batch size for generator using these cropped patches.
    Note that the cropped patches are not resized (diverse size).
    
    Pseudo codes for cropping process
        Input: 
            nframe <- 4
            windows <- [ [frame0, frame1, frame2, frame3, frame_4],
                         [frame1, frame2, ..., frame5],
                         ... ]
            frame_oth <- [ frames[:-2] for frames in windows ]
                (abbrev. for other input frames)
                (e.g., [ [frame0, frame1, frame2], ... ])
            frame_ds <- [ frames[-2] for frames in windows ]
                (abbrev. for detect frames)
                (e.g., [ frame3, ... ])
            frame_ts <- [ frames[-1] for frames in windows ]
                (abbrev. for target frames)
                (e.g., [ frame4, ... ])
        
        // not batch below, but use batch in practice.
        for i, frame_d in enumerate(frame_ds):
            areas = detect_yolo(frame_d) // the last frame of input (fourth frame)
            for area in areas:
                new_area = resize_area(area)
                frame_all = frame_oth[i] + [frame_d] + [frame_t]
                for idx, frame in enumerate(frame_all):
                    // idx <- num_of_frame (e.g., t0, t1, t2, t3, t4)
                    cropped = frame.crop(new_area)
                    save(cropped) to each idx-linked-directory
        
    Index of cropped patches
        >>> e.g., 01-0234-005.jpg
            - 01: number of folder in the original 'avenue/training/'
            - 0234: number of detect frame
            - 005: index of detected area in detect frame

    Directory root of cropped patches
        >>> root/ <- about 3.5GB
                t0/ <- (t-4) th frame
                    01-0003-000.jpg, 01-0003-001.jpg, ...
                    02-0003-000.jpg, ...
                    ...
                    16-0003-000.jpg, ...

                t1/ <- (t-3) th frame
                    01-0003-000.jpg, ...

                t2/ <- (t-2) th frame
                t3/ <- (t-1) th frame
                t4/ <- (t) th frame (target frame, gt)
    '''
    outpath = '/home/chojh21c/ADGW/m8/datasets/shanghai_cropped_08/'
    dpath = '/home/chojh21c/ADGW/Future_Frame_Prediction/datasets/shanghai/training/'
    dirs = os.listdir(dpath)
    dirs.sort()
    
    nframe = 4
    batch_size = args.batch_size
    max_h = 480
    max_w = 856
    factor_x = 1.5
    factor_y = 1.2
    device = f'cuda:{args.ngpu}'
    model = torch.hub.load('ultralytics/yolov5', 'yolov5l', pretrained=True).cuda()
    outdirs = crop_utils.set_dirs(outpath, nframe)

    for num_folder in dirs:
        vdatasets = VideoDataset(dpath, nframe=nframe, num_folder=num_folder)
        print(f'start folder number: {num_folder} / {len(dirs)}')

        total = math.ceil(vdatasets.len / batch_size)
        for batches in tqdm(vdatasets.get_batches(batch_size=batch_size), total=total):
            batch_num_detect, batch_frames_oth, batch_frames_d, batch_frame_t = batches

            batch_frames_d_ = batch_frames_d
            batch_areas = crop_utils.batch_detect(model, batch_frames_d_,
                                                confidence=args.confidence,
                                                device=device)

            for i, d in enumerate(batch_frames_d):

                batch_frames_d[i] = Image.fromarray(batch_frames_d[i])

            for i in range(len(batch_areas)):
                new_areas = crop_utils.get_resized_area(batch_areas[i],
                                                        max_w, max_h,
                                                        factor_x, factor_y)
                frames = batch_frames_oth[i] + [batch_frames_d[i]] + [batch_frame_t[i]]

                crop_utils.crop_save(frames, new_areas, outdirs, batch_num_detect[i], save=True)

