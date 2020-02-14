This readme explains the organisation of the Real Object Tracking dataset, 
and explains the usage of the code provided to read the data.
# Data Description
## MOT
The image files are zipped into the Real/zip folder. To use the MOT data, extract the zip files in Real/zip in a new folder Real/images, in appropriate subfolders. 
For example, contents of "Real/zip/0000000371_0000000000.zip" will be extracted in "Real/images/0000000371_0000000000/"

After that. the MOT data is arranged as follows:
```
Real/
    |---- annotations/
           |----seq1.csv
           |----seq2.csv
           ..
           ..
           |----seqn.csv
    |
    |---- Images/
           |---- seq1/
                   |----seq1_00000000001.jpg
                   |----seq1_00000000002.jpg
                   ..
                   ..
                   |----seq1_n.jpg
           |---- seq2/
           ..
           ..
           |---- seqn/
```

Here, the .csv files are the ground truth bounding boxes of that sequence. Each row in the groundtruth has the following format:

\<frame-id\>, \<object-id\>, \<min-x\>, \<min-y\>, \<width\>, \<height\>, \<object_class\>, \<species\>, \<occluded\>, \<noisy-frame\>

The image frames are in the corresponding sub folders in Rea/images. Note that the frames in the anotations as well as the files are 0-indexed.

## SOT
The SOT data was created by extracting trajectories of single objects from the MOT data. Naturally, there are multiple subsequences from the same set of frames. To avoid duplicating the images, the SOT subsequences have mapping to the original image directory Real/images instead of actual images.

There are 2 types of SOT dataset. data_split_perfect contains only contiguous trajectories for each object.
data_split_full contains longer trajectories, which may or may not be contiguous. 

The data is organised in the following way:

```
Real/
    |---- sot/
            |---- data_split/
                        |----train/
                            |----train_seq1/
                                    |----groundtruth_rect.txt
                                    |----img_list.txt
                            |----train_seq2/
                                    |----groundtruth_rect.txt
                                    |----img_list.txt
                            ..
                            ..
                            |----train_seqn/
                                    |----groundtruth_rect.txt
                                    |----img_list.txt
                        |                    
                        |----test/
                            |----test_seq1/
                                    |----groundtruth_rect.txt
                                    |----img_list.txt
                            |----test_seq2/
                                    |----groundtruth_rect.txt
                                    |----img_list.txt
                            ..
                            ..
                            |----test_seqn/
                                    |----groundtruth_rect.txt
                                    |----img_list.txt
```

Here, for each subsequence, groundtruth_rect.txt contains the bounding boxes in SOT format : \<xmin\>, \<xmax\>, \<width\>, \<height\>
and img_list.txt is the mapping of images to the original video's frames. Note that this indexing is 1-indexed, while
frames in the MOT format are 0-indexed. Similarly for data_split_full


# Code files
Helper code is provided in the Code folder to read both MOT and SOT sequences:
- annotations.py 
    * class Annotations: Read ground truth bounding box annotations of a sequence
- sequence.py
    * class Sequence: Read annotations and images of an MOT video sequence
    To read the sequence:
    ```
    # MOT Sequence
    from sequence import Sequence
    d = '../Real/images/0000000058_0000000000'
    g = '../Real/annotations/0000000067_0000000055.csv'
    sequence = SotSequence(d,g, seqname = '0000000067_0000000055')
    gt = seqeunce.gt_annotations
    frames = sequence.images
    ```
    
    * class SotSequence: Read annoations and images from the image mappings for an SOT sequence
    ```
    # SOT Sequence
    from sequence import SotSequence
    d = '../Real/images/0000000058_0000000000'
    g = '../Real/sot/data_split_full/train/0000000067_0000000055_14_357-595/groundtruth_rect.txt'
    m = '../Real/sot/data_split_full/train/0000000067_0000000055_14_357-595/img_list.txt
    sequence = SotSequence(d,g,m, seqname = '0000000067_0000000055_14_357-595')
    gt = seqeunce.gt_annotations
    frames = sequence.images
    ```

- tracker.py : Class to represent results from a tracker
    * class Tracker: Read all results of a tracker, return results of a particular sequence
- generate_sot_videos.py
    An example script that generates videos of SOT sequences, demonstrating the usage of above classes SotSequence and Tracker

    Run `python generate_sot_videos.py` to generate the sot videos with default options.
    To generate videos along with tracker results, put the results of that tracker in path/to/results/tracker_name/ in .csv or .mat format (For example Real/sot/results/ADNet/0000000058_000000000.mat). The format of the results should be same as the SOT GT annotations. Pass the path/to/results in --tracker_resdir argument. And pass the names of all the trackers to the --tracker argument. For example, `python generate_sot_videos.py --trackers_resdir ../Real/sot/results --trackers ADNet SiamRPN ECO`
    
    Use `python generate_sot_videos.py -h` for more information
- generate_mot_videos.py
    An example script that generates videos of MOT sequences, demonstrating the usage of above classes Sequence and Tracker

    Run `python generate_sot_videos.py` to generate the sot videos with default options.
    To generate videos along with tracker results, put the results of that tracker in path/to/results/tracker_name/ in .csv or .mat format (For example Real/mot-results/MDP/0000000058_000000000.csv). The format of the results should be same as the MOT GT annotations. Pass the path/to/results in --tracker_resdir argument. And pass the names of all the trackers to the --tracker argument. For example, `python generate_mot_videos.py --trackers_resdir ../Real/mot-results --trackers iou MDP`

    Use `python generate_sot_videos.py -h` for more information
