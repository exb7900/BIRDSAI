"""
Class to represent a Video Seqeunce
"""
from annotations import Annotations
import numpy as np
from PIL import Image
from PIL import ImageDraw
import os
import cv2
from tqdm import tqdm

COLOR = [(255, 0, 0),
         (0  , 162, 232),
         (136, 0  , 2255),
         (128, 128, 128),
         (0, 255, 0),
         (255, 0, 255),
         (255, 255, 0),
         (0, 0, 255),
         (0, 255, 255),
         (255, 127, 39),
         (0.2,0.7,0.255),
         (0.7,0.2,0.255),
         (65, 51, 178),
         (0, 0, 0)]

class Sequence:
    def __init__(self, image_dir, gt_path, seqname=None, trackers=[]): 
        if (seqname == None):
            # Assumed data format is /path/to/data/seqname.{csv|txt}
            self.seqname = gt_path.split('/')[-1][:-4]
        else:
            self.seqname = seqname

        self.gt_annotations = Annotations(gt_path, seqname=seqname)
        self.tracker_res = {}
        for i in trackers:
            try:
                self.tracker_res[i.name] = i.get_res_of(seqname)
            except:
                print(self.seqname, 'not available for', i.name)
        self.img_dir = image_dir
        self.images = [i for i in os.listdir(image_dir) if i[-4:] == '.png' or i[-4:] == '.jpg']
        self.images.sort()
        height, width, layers = cv2.imread(os.path.join(image_dir, self.images[0])).shape
        self.height = height
        self.width = width
        self.size = (width, height)
        self.obj_size = self.gt_annotations.obj_size


    def get_frame(self, frame_num, scale_bb=1, sort_order='x',
                  off_x=0, off_y=0, show_boxes=True, boxColor=0, 
                  show_text=True, textColor=0, save_path=None):
        img_path = os.path.join(self.img_dir, self.images[frame_num])
        try:
            with Image.open(img_path) as image:
                #print("Mode", image.mode, type(image.mode))
                if (image.mode != 'L'):
                    image = image.convert(mode='L')
                draw = ImageDraw.Draw(image, 'L')

                dets = self.gt_annotations.get_frame(frame_num+1)[:,1:].astype(np.float)
                if len(dets) == 0:
                    if (save_path is not None):
                        image.save(save_path)
                    return image

                # Format: [frame, ID, x1, y1, width, height, obj_class, species, occluded, noisy_frame]
                labels = None
                #print('sort_order', sort_order)
                if (sort_order == 'x+y'):
                    d = dets[np.argsort(dets[:,2] + dets[:,1])]
                else:
                    d = dets[np.lexsort((dets[:,2],dets[:,1]))]
                ids = d[:,0]
                #print(ids)
                h,w = image.size
                #frame_text = "Frame: " + frame_num + "ids on screen: \n" + str(ids) + "\n count = " + str(len(ids))
                frame_text = "Frame: {0}\nids on screen: \n{1}\n count = {2}".format(frame_num, ids, len(ids))
                if d.shape[1]>8 and d.shape[0]>0 and d[0,8] == 1:
                    # Noisy frame
                    frame_text = "Noisy Frame\n" + frame_text
                draw.text((0, 0), frame_text, (textColor))#(0))
                # Custom Offset
                dets[:,1] += off_x
                dets[:,2] += off_y


                # Scale the boxes:
                dets[:, 1] -= dets[:, 3] * ((scale_bb - 1)/2)
                dets[:, 2] -= dets[:, 4] * ((scale_bb - 1)/2)
                dets[:, 3:5] = dets[:, 3:5] * scale_bb
                # Convert from [x1, y1, width, height] to [x1, y1, x2, y2]
                dets[:, 3:5] += dets[:,1:3]
                ids = []
                species = {
                '-1':"Unknown",
                '0':"Human",
                '1':"Elephant",
                '2':"Lion",
                '3':"Giraffe",
                '4':"Dog"
                }
                for i, d in enumerate(dets):
                    if (show_boxes):
                        boxcolor = boxColor
                        if d.shape[0]>7 and d[7] == 1:
                            boxcolor = 255 - boxColor
                        draw.rectangle([d[3], d[4], d[1], d[2]], outline=(boxcolor))#c)
                    #ids.append(d[0])
                    d = d.astype(np.int32)

                    if (show_text):
                        boxTag = str(d[0])
                        if len(d)>6 and d[5] == 0:
                            # animal
                            boxTag += '-' + species[str(d[6])]
                        draw.text((d[1],d[2] -10),boxTag, (textColor))

                if (save_path is not None):
                    image.save(save_path)

                return image

        except Exception as e:
            #print(str(e))
            print("error in redrawing image")
            raise e


    def get_frame_w_trackers(self, frame_num, scale_bb=1, sort_order='x',
                  off_x=0, off_y=0, show_boxes=True, boxColor=0, 
                  show_text=True, textColor=0, save_path=None):
        img_path = os.path.join(self.img_dir, self.images[frame_num])
        det_matrix = {
            'GT': self.gt_annotations.get_frame(frame_num)[:,1:].astype(np.float)
        }
        colors = {
            'GT': [textColor, boxColor]
        }
        index=1
        for i in self.tracker_res:
            det_matrix[i] = self.tracker_res[i].get_frame(frame_num)[:,1:].astype(np.float)
            colors[i] = [COLOR[index], COLOR[index]]
            index+=1

        try:
            with Image.open(img_path) as image:
                #print("Mode", image.mode, type(image.mode))
                if (image.mode != 'RGB'):
                    image = image.convert(mode='RGB')
                draw = ImageDraw.Draw(image)

                frame_text = "Frame: {0}".format(frame_num)
                d = det_matrix['GT']
                if d.shape[1]>8 and d.shape[0]>0 and d[0,8] == 1:
                    # Noisy frame
                    frame_text += " (Noisy Frame)"
                draw.text((0, 0), frame_text, (textColor))#(0))
                text_start = draw.textsize(frame_text)[1] +1

                for e in det_matrix:
                    dets = det_matrix[e]
                    textColor = colors[e][0]
                    boxColor = colors[e][1]

                    #dets = self.gt_annotations.get_frame(frame_num)[:,1:].astype(np.float)
                    if len(dets) == 0:
                        if (save_path is not None):
                            image.save(save_path)
                        continue#return image

                    # Format: [frame, ID, x1, y1, width, height, obj_class, species, occluded, noisy_frame]
                    labels = None
                    #print('sort_order', sort_order)
                    if (sort_order == 'x+y'):
                        d = dets[np.argsort(dets[:,2] + dets[:,1])]
                    else:
                        d = dets[np.lexsort((dets[:,2],dets[:,1]))]
                    ids = d[:,0]
                    #print(ids)
                    h,w = image.size
                    #frame_text = "Frame: " + frame_num + "ids on screen: \n" + str(ids) + "\n count = " + str(len(ids))
                    frame_text = "{0}: {1}  count = {2}".format(e, ids, len(ids))
                    draw.text((0, text_start), frame_text, (textColor))#(0))
                    text_start += draw.textsize(frame_text)[1] +1
                    # Custom Offset
                    dets[:,1] += off_x
                    dets[:,2] += off_y


                    # Scale the boxes:
                    dets[:, 1] -= dets[:, 3] * ((scale_bb - 1)/2)
                    dets[:, 2] -= dets[:, 4] * ((scale_bb - 1)/2)
                    dets[:, 3:5] = dets[:, 3:5] * scale_bb
                    # Convert from [x1, y1, width, height] to [x1, y1, x2, y2]
                    dets[:, 3:5] += dets[:,1:3]
                    ids = []
                    species = {
                    '-1':"Unknown",
                    '0':"Human",
                    '1':"Elephant",
                    '2':"Lion",
                    '3':"Giraffe",
                    '4':"Dog"
                    }
                    for i, d in enumerate(dets):
                        if (show_boxes):
                            boxcolor = boxColor
                            if d.shape[0]>7 and d[7] == 1:
                                boxcolor = 255 - boxColor
                            draw.rectangle([d[3], d[4], d[1], d[2]], outline=(boxcolor))#c)
                        #ids.append(d[0])
                        d = d.astype(np.int32)

                        if (show_text):
                            boxTag = str(d[0])
                            if len(d)>6 and d[5] == 0:
                                # animal
                                boxTag += '-' + species[str(d[6])]
                            draw.text((d[1],d[2] -10),boxTag, (textColor))

                    if (save_path is not None):
                        image.save(save_path)

                return image

        except Exception as e:
            #print(str(e))
            print("error in redrawing image")
            raise e

    def generate_video(self, output_dir, fps=15, start_frame=0, end_frame=None):
        if end_frame is None:
            end_frame = len(self.images)
        if start_frame > len(self.images) or end_frame < start_frame or end_frame > len(self.images):
            print("Invalid input", start_frame, end_frame, "number of images is", len(self.images))
            return
        outpath = os.path.join(output_dir, self.seqname + '.avi')
        out = cv2.VideoWriter(outpath, cv2.VideoWriter_fourcc(*"XVID"), float(fps), self.size)
        #for i in tqdm(range(start_frame,end_frame)):
        for i in range(start_frame,end_frame):
            img_path = './output-imgs/' + self.images[i]
            #out.write(np.asarray(self.get_frame(i)))
            self.get_frame(i, save_path = img_path)
            out.write(cv2.imread(img_path))


        out.release()
        #print('Written video to', outpath)


    def generate_video_w_trackers(self, output_dir, fps=15, start_frame=0, end_frame=None):
        if end_frame is None:
            end_frame = len(self.images)
        if start_frame > len(self.images) or end_frame < start_frame or end_frame > len(self.images):
            print("Invalid input", start_frame, end_frame, "number of images is", len(self.images))
            return
        outpath = os.path.join(output_dir, self.seqname + '.avi')
        out = cv2.VideoWriter(outpath, cv2.VideoWriter_fourcc(*"XVID"), float(fps), self.size)
        #for i in tqdm(range(start_frame,end_frame)):
        for i in range(start_frame,end_frame):
            img_path = './output-imgs/' + self.images[i]
            #out.write(np.asarray(self.get_frame(i)))
            self.get_frame_w_trackers(i, save_path = img_path)
            out.write(cv2.imread(img_path))


        out.release()
        print('Written video to', outpath)


class SotSequence(Sequence):
    def __init__(self, orig_image_dir, gt_path, img_list_file, seqname=None, trackers=[]): 
        if (seqname == None):
            # Assumed data format is /path/to/data/seqname.{csv|txt}
            self.seqname = gt_path.split('/')[-2]
        else:
            self.seqname = seqname

        super().__init__(orig_image_dir, gt_path, seqname=seqname, trackers=trackers)
        with open(img_list_file, 'r') as f:
            self.img_map = [i.split(':') for i in f.read().splitlines()]
        self.images = [i[1].split('/')[-1] for i in self.img_map]
