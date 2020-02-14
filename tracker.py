import os
import os.path as osp
import scipy.io as sio
from annotations import Annotations

class Tracker():
    def __init__(self, name, res_dir):
        self.name = name
        self.res_dir = res_dir
        res = os.listdir(res_dir)
        if res[0][:-4] == '.mat':
            self.ismat = True

        self.res = {}

    def get_res_of(self, seqname):
        if seqname not in self.res:
            path = osp.join(self.res_dir, seqname + '.csv')
            if self.ismat:
                path = osp.join(self.res_dir, seqname + '.mat')

            self.res[seqname] = Annotations(path, ismat=ismat)

        return self.res[seqname]
