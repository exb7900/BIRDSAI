"""
Class that represents the annotations for a Sequence
"""

import pandas as pd
import numpy as np
import os
import scipy.io as sio

"""
This class loads and represents a video sequence's tracking annotations

Contains functions to access the annotations and some properties"""
class Annotations:

    # <frame-id>, <object-id>, <min-x>, <min-y>, <width>, <height>, <object_class>, <species>, <occluded>, <noisy-frame>

    CSV_SCHEMA = {
        'fr': 0,
        'id': 1,
        'x': 2,
        'y': 3,
        'w': 4,
        'h': 5,
        'obj_class': 6,
        'species': 7,
        'occluded': 8,
        'noisy': 9
    }

    """
    Read and load annotations from filepath.
    
    mode: 'xywh' or 'xyxy', the mode in which the tracks array is loaded"""
    def __init__(self, filepath, seqname=None, split_in_frames=False, mode='xywh', ismat=False):
        if (seqname == None):
            # Assumed data format is /path/to/data/seqname.{csv|txt}
            self.seqname = filepath.split('/')[-1][:-4]
        else:
            self.seqname = seqname

        self.filepath = filepath

        if not ismat:
            tracks = pd.read_csv(filepath, header=None).dropna().values.astype(np.int)
        else:
            tracks = sio.loadmat(filepath)['results']
        if (tracks.shape[1] == 4):
            # SOT mode
            tracks = np.insert(tracks, 0, 1, axis=1)
            tracks = np.insert(tracks, 0, np.arange(tracks.shape[0]), axis=1)
            # print(tracks)

        tracks = tracks[np.lexsort((tracks[:,1], tracks[:,0]))]
        self.num_frames = np.max(tracks[:,0])+1
        
        self.split_in_frames = split_in_frames
        if (split_in_frames):
            self.frames = [[] for i in range(self.num_frames)]
            for i in tracks:
                f = i[0]
                self.frames[f].append(i)

            #self.frames = [i for i in self.frames if i != []]

            self.stitch_frames()
            
            # Sanity check
            assert np.array_equal(tracks, self.tracks)
        
        if mode=='xyxy':
            tracks[:,4]+=tracks[:,2]
            tracks[:,5]+=tracks[:,3]

        self.mode = mode
        self.tracks = tracks
        self.max_neg_id = np.min(tracks[:,1])
        self.obj_size = self.get_avg_size()

    def get_tracks(self, mode='xywh'):
        if self.mode == mode:
            return self.tracks
        if mode == 'xywh':
            print('switching modes')
            self.mode = mode
            self.tracks[:,4]-=self.tracks[:,2]
            self.tracks[:,5]-=self.tracks[:,3]
            return self.tracks
        if mode == 'xyxy':
            self.mode = mode
            self.tracks[:,4]+=self.tracks[:,2]
            self.tracks[:,5]+=self.tracks[:,3]
            return t

    def get_ids(self):
        return np.unique(self.tracks[:,1])

    def get_frame(self, frame_num, copy=False):
        boxes = self.tracks[self.tracks[:,0] == frame_num]
        if copy:
            return boxes.copy()
        else:
            return boxes

    def set_frame(self, frame_num, tracks):
        if self.split_in_frames:
            for j in tracks:
                if j[1] == -1:
                    self.max_neg_id-=1
                    j[1] = self.max_neg_id
            self.frames[frame_num] = tracks.tolist()

        if self.max_neg_id > np.min(tracks[:,1]):   
            self.max_neg_id = np.min(tracks[:,1])

    def stitch_frames(self):
        frames = [i for i in self.frames if i != []]
        self.tracks = np.concatenate(frames, axis=0)

    def write_csv(self, output_dir=None):
        if output_dir is None:
            path = self.filepath
        else:
            path = os.path.join(output_dir, self.seqname)
            path = path + '.csv'
        if (self.split_in_frames):
            self.stitch_frames()
        self.tracks = self.tracks[np.lexsort((self.tracks[:,1], self.tracks[:,0]))]
        pd.DataFrame(self.tracks).to_csv(path, header=False, index=False)

    def get_avg_size(self): 
        areas = self.tracks[:,4] * self.tracks[:,5]
        avg_area = np.mean(areas)
        if avg_area < 200:
            return 'small'
        elif avg_area < 2000:
            return 'medium'
        else:
            return 'large'


