'''
Author: Lu ShaoAn
Brief: 
Version: 0.1
Date: 2024-11-01 17:18:06
Copyright: Copyright (c) 2022
'''
import json
import os
from typing import List
import cv2
import os
from mmengine.config import Config
import numpy as np
from mmengine.dataset import BaseDataset
from mmpose.registry import DATASETS

@DATASETS.register_module()
class BarcodeDataset(BaseDataset):
    def __init__(self, *args, train_file_infos:List[dict], **kwargs):
        self.train_file_infos = train_file_infos  
        super().__init__(*args, **kwargs)

    def load_data_list(self) -> List[dict]:
        data_list = []
        img_id = 0
        for file_info in self.train_file_infos:
            root_path = file_info["root"]
            ann_file_path = file_info["path"]
            with open(ann_file_path, "r") as train_file:
                lines = train_file.readlines()
                for line in lines:
                    data_info = {}
                    img_path = os.path.join(root_path, line).split("\n")[0]
                    file_name, _ = os.path.splitext(img_path)
                    json_path = file_name + ".json"
                    if not os.path.exists(img_path) or not os.path.exists(json_path):
                        continue
                    
                    with open(json_path, "r") as json_file:
                        json_data = json.load(json_file)

                        if len(json_data['shapes']) != 2:
                            continue
                        data_info['img_path'] = img_path

                        for shape in json_data['shapes']:
                            if 'coarse' not in shape['label']:
                                keypoints = np.array(shape['points'], dtype=np.float32)
                                keypoints = np.expand_dims(keypoints, 0)
                                data_info['keypoints'] = keypoints
                                data_info['keypoints_visible'] = np.ones((1,4))
                            else:
                                coarse = np.array(shape['points'], dtype=np.float32)
                                x,y,w,h = cv2.boundingRect(coarse)
                                bbox = np.array([x,y,x+w,y+h],dtype=np.float32).reshape(1, 4)
                                data_info['bbox'] = bbox
                                data_info['bbox_score'] = np.ones(1, dtype=np.float32)
                        
                        if data_info.get("keypoints") is None or data_info.get("bbox") is None:
                            continue

                        data_info['img_id'] = img_id
                    img_id += 1
                    data_list.append(data_info)
        
        return data_list
