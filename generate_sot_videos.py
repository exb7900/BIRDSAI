import os
import os.path as osp
import argparse
from tqdm import tqdm
import pandas as pd
from sequence import Sequence, SotSequence
from tracker import Tracker

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', "--sequence")
    parser.add_argument('--img_dir', default='../Real/images')
    parser.add_argument('--annot_dir', default='../Real/sot/data_split_perfect/test')
    parser.add_argument('--output_dir', default='./vids')
    parser.add_argument('--start_frame', type=int, default=0)
    parser.add_argument('--end_frame', type=int)
    parser.add_argument('--trackers', nargs='+')
    parser.add_argument('--trackers_resdir', default='../Real/sot/results')
    args = parser.parse_args()
    
    tracker_names = args.trackers
    if tracker_names is None:
        tracker_names = []
    if args.sequence:
        seqs = [args.sequence]
    else:
        seqs = [i for i in os.listdir(args.annot_dir)]# if i[-4:] == '.csv']
        seqs.sort()
    
    trackers = []
    for t in tracker_names:
        trackers.append(Tracker(t, osp.join(args.trackers_resdir,t)))

    for seq in tqdm(seqs):

        d = osp.join(args.img_dir,seq[:21])
        g = osp.join(args.annot_dir,seq,'groundtruth_rect.txt')
        m = osp.join(args.annot_dir,seq,'img_list.txt')
        sequence = SotSequence(d, g, m, seqname=seq, trackers=trackers)
        sequence.generate_video_w_trackers(osp.join(args.output_dir, sequence.obj_size), start_frame=args.start_frame, end_frame=args.end_frame)
