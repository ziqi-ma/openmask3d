import os
import numpy as np
import clip
import torch
import pdb
from eval_semantic_instance import evaluate
import tqdm
import argparse
import json
import time

class InstSegEvaluator():
    def __init__(self, clip_model_type):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_model_type = clip_model_type
        self.clip_model = self.get_clip_model(clip_model_type)
        self.feature_size = self.get_feature_size(clip_model_type)

    def get_query_sentences(self, label_list, sentence_structure="{}"):
        new_label_list = ["other"] + label_list
        return [sentence_structure.format(label) for label in new_label_list]

    def get_clip_model(self, clip_model_type):
        clip_model, _ = clip.load(clip_model_type, self.device)
        return clip_model

    def get_feature_size(self, clip_model_type):
        if clip_model_type == 'ViT-L/14' or clip_model_type == 'ViT-L/14@336px':
            return 768
        elif clip_model_type == 'ViT-B/32':
            return 512
        else:
            raise NotImplementedError

    def get_text_query_embeddings(self):
        # ViT_L14_336px for OpenSeg, clip_model_vit_B32 for LSeg
        text_query_embeddings = torch.zeros((len(self.query_sentences), self.feature_size))

        for label_idx, sentence in enumerate(self.query_sentences):
            #print(label_idx, sentence) #CLASS_LABELS_20[label_idx],
            text_input_processed = clip.tokenize(sentence).to(self.device)
            with torch.no_grad():
                sentence_embedding = self.clip_model.encode_text(text_input_processed)

            sentence_embedding_normalized =  (sentence_embedding/sentence_embedding.norm(dim=-1, keepdim=True)).float().cpu()
            text_query_embeddings[label_idx, :] = sentence_embedding_normalized

        return text_query_embeddings

    def compute_classes_per_pt(self, masks_path, mask_features_path, keep_first=None):
        masks = torch.load(masks_path)
        mask_features = np.load(mask_features_path)

        if keep_first is not None:
            masks = masks[:, 0:keep_first]
            mask_features = mask_features[0:keep_first, :]

        # normalize mask features
        mask_features_normalized = mask_features/(np.linalg.norm(mask_features, axis=1)[..., None]+1e-12)

        similarity_scores = mask_features_normalized@self.text_query_embeddings.T #(n_masks, n_classes)l1873
        n_masks_per_point = masks.sum(axis=1)
        pt_score_sum = masks @ similarity_scores
        pt_score = pt_score_sum/(n_masks_per_point.reshape(-1,1)+1e-6)
        pred = pt_score.argmax(axis=1)
        return pred

    def evaluate_full(self, ordered_label_list, data_dir):
        self.query_sentences = self.get_query_sentences(ordered_label_list)
        self.text_query_embeddings = self.get_text_query_embeddings().numpy()
        pt_pred = self.compute_classes_per_pt(f"{data_dir}/pc_zup_masks.pt", f"{data_dir}/pc_zup_openmask3d_features.npy", keep_first=None)
        gt = np.load(f"{data_dir}/label.npy",allow_pickle=True).item()['semantic_seg']+1 # this starts with -1, add 1 to start with 0 (other)
        acc = (pt_pred==gt).sum()/gt.shape[0]
        # get iou
        part_ious = []
        for part in range(len(ordered_label_list)):
            I = np.sum(np.logical_and(pt_pred == part+1, gt == part+1))
            U = np.sum(np.logical_or(pt_pred == part+1, gt == part+1))
            if U == 0:
                iou = 1  # If the union of groundtruth and prediction points is empty, then count part IoU as 1
            else:
                iou = I / float(U)
                part_ious.append(iou)
        mean_iou = np.mean(part_ious)
        print(f"acc: {acc}, iou: {mean_iou}")
        return acc, mean_iou


if __name__ == '__main__':
    my_dict = {}
    my_dict["Bottle"]="4403 6771 3558 3520 4233"
    my_dict["Box"]="100664 100658 100141 100243 100189"
    my_dict["Bucket"]="100477 100482 102359 100486 100470"
    my_dict["Camera"]="102873 102442 102831 102417 102876"
    my_dict["Cart"]="101053 100852 101091 100504 100075"
    my_dict["Chair"]="37099 38325 40982 44729 41653"
    my_dict["Clock"]="6613 6934 6917 7037 7007"
    my_dict["CoffeeMachine"]="103041 103140 103084 103129 103143"
    my_dict["Dishwasher"]="12606 12592 12559 12092 12259"
    my_dict["Dispenser"]="101542 101528 103378 101561 101490"
    my_dict["Display"]="4608 4589 3392 5088 4541"
    my_dict["Door"]="9107 9388 8897 8961 8983"
    my_dict["Eyeglasses"]="102590 101843 102612 102586 102617"
    my_dict["Faucet"]="1667 156 2017 1785 857"
    my_dict["FoldingChair"]="100557 100609 100532 100598 100568"
    my_dict["Globe"] = "100803 100745 100786 100758 100756"
    my_dict["Kettle"]="102739 102768 101305 102761 102753"
    my_dict["Dispenser"]="101542 101528 103378 101561 101490"
    my_dict["Keyboard"]="13082 12727 13136 12917 12965"
    my_dict["KitchenPot"]="100017 100028 100025 100047 100051"
    my_dict["Knife"]="103583 103713 103700 103739 101660"
    my_dict["Lamp"]="14306 14703 16012 14567 14402"
    my_dict["Laptop"]="11156 10238 10885 11876 9918"
    my_dict["Lighter"]="100289 100285 100334 100335 100320"
    my_dict["Microwave"]="7273 7221 7167 7292 7263"
    my_dict["Mouse"]="102273 103022 101408 102276 102272"
    my_dict["Oven"]="101909 102019 7347 7130 101773"
    my_dict["Pen"]="102965 101713 102939 102963 102960"
    my_dict["Phone"]="103892 103917 103251 103350 103941"
    my_dict["Pliers"]="102258 102260 100142 102221 102242"
    my_dict["Printer"]="104009 104004 103811 104006 103866"
    my_dict["Refrigerator"]="11846 10612 12038 12055 10627"
    my_dict["Microwave"]="7273 7221 7167 7292 7263"
    my_dict["Remote"]="104044 100395 104039 101139 101014"
    my_dict["Safe"]="101612 102381 101611 101584 101579"
    my_dict["Scissors"]="10502 11029 10895 10968 11077"
    my_dict["Stapler"]="103099 103271 103111 103789 103301"
    my_dict["StorageFurniture"]="46456 45384 46879 45247 46556"
    my_dict["Suitcase"]="101668 100550 101049 100248 100842"
    my_dict["Switch"]="100911 100952 102872 100971 100845"
    my_dict["Table"]="20745 27619 25913 26899 26545"
    my_dict["Toaster"]="103485 103560 103558 103473 103469"
    my_dict["Toilet"]="102648 102631 102675 102636 102688"
    my_dict["TrashCan"]="102187 102165 102227 102202 102229"
    my_dict["USB"]="100061 100065 102052 100128 101999"
    my_dict["WashingMachine"]="103452 103776 100283 103369 103518"
    my_dict["Window"]="103332 103239 102985 103315 103015"
    
    category = "Window"
    ids = my_dict[category].split(" ")
    print(ids)

    #parser = argparse.ArgumentParser()
    #opt = parser.parse_args()
    # ScanNet200, "a {} in a scene", all masks are assigned 1.0 as the confidence score
    stime = time.time()
    evaluator = InstSegEvaluator('ViT-L/14@336px')
    with open(f"/data/partnet-mobility/PartNetE_meta.json") as f:
        all_mapping = json.load(f)
    labels = all_mapping[category]
    print(labels)
    acc_all = []
    iou_all = []
    for id in ids:
        data_dir = f"/data/partnet-mobility/test/{category}/{id}"
        acc, iou = evaluator.evaluate_full(labels, data_dir)
        acc_all.append(acc)
        iou_all.append(iou)
    print(np.mean(acc_all))
    print(np.mean(iou_all))
    etime = time.time()
    print((etime-stime)/5)
       