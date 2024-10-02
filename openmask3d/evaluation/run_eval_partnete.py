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
    classes = ["Bottle","Box","Bucket","Camera","Cart","Chair","Clock","CoffeeMachine",
                "Dishwasher","Dispenser","Display","Door","Eyeglasses","Faucet","FoldingChair",
                "Globe","Kettle","Keyboard","KitchenPot","Knife","Lamp","Laptop","Lighter",
                "Microwave","Mouse","Oven","Pen","Phone","Pliers","Printer","Refrigerator",
                "Remote","Safe","Scissors","Stapler","StorageFurniture","Suitcase","Switch",
                "Table","Toaster","Toilet","TrashCan","USB","WashingMachine","Window"]
    print(len(classes))
    evaluator = InstSegEvaluator('ViT-L/14@336px')
    with open(f"/data/partnet-mobility/PartNetE_meta.json") as f:
        all_mapping = json.load(f)
    print(all_mapping)
    stime = time.time()
    acc_allcats = []
    iou_allcats = []
    for category in classes:
        cat_stime = time.time()
        labels = all_mapping[category]
        print(labels)
        decorated_labels = [f"a {label} of a {category}" for label in labels]
        print(labels)
        with open(f"/data/partnet-mobility/test/{category}/subsampled_ids.txt", 'r') as f:
            data_paths = f.read().splitlines()
        acc_all = []
        iou_all = []
        for data_dir in data_paths:
            id = data_dir.split("/")[-1]
            data_dir = f"/data/partnet-mobility/test/{category}/{id}"
            acc, iou = evaluator.evaluate_full(decorated_labels, data_dir)
            acc_all.append(acc)
            iou_all.append(iou)
        acc_allcats.append(np.mean(acc_all))
        iou_allcats.append(np.mean(iou_all))
        print(f"category {category}, acc mean {np.mean(acc_all)}, iou mean {np.mean(iou_all)}")
        cat_etime = time.time()
        print(f"time {cat_etime-cat_stime}")
    all_macc = np.mean(acc_allcats)
    all_miou = np.mean(iou_allcats)
    print(f"all category, mean acc {all_macc}, mean iou {all_miou}")
    etime = time.time()
    print(etime-stime)
       