import torch
from PIL import Image, ImageDraw, ImageColor, ImageFont
import torchvision.transforms.functional as F
import numpy as np

class SSCBoxDataToMaskNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "bboxes": ("BBOX,JSON", {"forceInput":True}),
                "width": ("INT", {"forceInput":True}),
                "height": ("INT", {"forceInput":True}),
                "box_max_width": ("INT", {"default": 10000}),
                "box_max_height": ("INT", {"default": 10000}),     
            },
        }
 
    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)

    DESCRIPTION = "Creates a mask from a bbox or from a data JSON list of bboxes given by the Florence2Run caption_to_phrase_grounding node. It can also filter out bboxes above a size."
    FUNCTION = "test"
    CATEGORY = "mask"
    #OUTPUT_NODE = False
 


    @classmethod
    def VALIDATE_INPUTS(cls, input_types):
        if input_types["bboxes"] not in ("BBOX","JSON"):
            return "bboxes must be an BBOX or data must be JSON type"
        else:
            return True

    def test(self, bboxes, width, height, box_max_width, box_max_height):
        # print("Hello world",data,"data[0]",data[0])
        # print("BBox to Mask & Size Filter SSC",bboxes,data)
        print("BBox to Mask & Size Filter SSC")

        masks=[]
        if check_nesting_level(bboxes)==2:
            masks.append(bboxes)
        if check_nesting_level(bboxes)==3:
            masks.extend(bboxes)

        maskList=[]
        for frame in masks:
            mask = torch.zeros((height, width), dtype=torch.float32)
            for bbox in frame:
                # Unpack bounding box (x1, y1, x2, y2)
                x1, y1, x2, y2 = map(int, bbox[:4])
                if x2-x1 > box_max_width:
                    continue
                if y2-y1 > box_max_height:
                    continue

                # Ensure coordinates are within image bounds
                x1 = max(0, min(x1, width-1))
                y1 = max(0, min(y1, height-1))
                x2 = max(0, min(x2, width-1))
                y2 = max(0, min(y2, height-1))
                
                # Ensure x1 < x2 and y1 < y2
                x1, x2 = min(x1, x2), max(x1, x2)
                y1, y2 = min(y1, y2), max(y1, y2)
                
                # Draw white rectangle
                mask[y1:y2+1, x1:x2+1] = 1.0
            maskList.append(mask)

        # out_mask_tensor = torch.cat(maskList, dim=0)

        out_mask_tensor = torch.stack(maskList)
        return (out_mask_tensor,)

 
def check_nesting_level(lst):
    if not isinstance(lst, list):
        return 0  # Not a list

    max_level = 0
    for item in lst:
        level = check_nesting_level(item) + 1
        max_level = max(max_level, level)

    return max_level

# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "SSCBoxDataToMask": SSCBoxDataToMaskNode
}
 
# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "SSCBoxDataToMask": "BBox to Mask & Size Filter SSC"
}