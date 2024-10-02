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

    def evaluate_full(self, data_dir):
        with open(f"{data_dir}/label_map.json") as f:
            label_dict = json.load(f)
        ordered_label_list = []
        for i in range(len(label_dict)):
            ordered_label_list.append(label_dict[str(i+1)])
        self.query_sentences = self.get_query_sentences(ordered_label_list)
        self.text_query_embeddings = self.get_text_query_embeddings().numpy()
        pt_pred = self.compute_classes_per_pt(f"{data_dir}/pc_zup_masks.pt", f"{data_dir}/pc_zup_openmask3d_features.npy", keep_first=None)
        gt = np.load(f"{data_dir}/labels.npy") # 0 is unlabeled
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

    #parser = argparse.ArgumentParser()
    #opt = parser.parse_args()
    # ScanNet200, "a {} in a scene", all masks are assigned 1.0 as the confidence score
    stime = time.time()
    evaluator = InstSegEvaluator('ViT-L/14@336px')
    split = "shapenetpart"
    class_uids = sorted(os.listdir(f"/data/objaverse/holdout/{split}")) # inside docker, data is data/ziqi
    stime = time.time()
    all_accs = []
    all_ious = []
    cat_iou = {}
    cat_acc = {}
    for class_uid in class_uids:
        obj_path = f"/data/objaverse/holdout/{split}/{class_uid}"
        print(class_uid)
        cat = class_uid.split("_")[0]
        acc, iou = evaluator.evaluate_full(obj_path)
        all_accs.append(acc)
        all_ious.append(iou)
        
    inst_mean_acc = np.mean(all_accs)
    inst_mean_iou = np.mean(all_ious)
    print(f"mean acc {inst_mean_acc}, iou {inst_mean_iou}")
    etime = time.time()
    print((etime-stime))
       