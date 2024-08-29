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
        gt = np.load(f"{data_dir}/gt.npy") # this starts with 0
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

    ids = ["0","1","2","3","4"]
    cat2part = {'airplane': ['body','wing','tail','engine or frame'], 'bag': ['handle','body'], 'cap': ['panels or crown','visor or peak'], 
            'car': ['roof','hood','wheel or tire','body'],
            'chair': ['back','seat pad','leg','armrest'], 'earphone': ['earcup','headband','data wire'], 
            'guitar': ['head or tuners','neck','body'], 
            'knife': ['blade', 'handle'], 'lamp': ['leg or wire','lampshade'], 
            'laptop': ['keyboard','screen or monitor'], 
            'motorbike': ['gas tank','seat','wheel','handles or handlebars','light','engine or frame'], 'mug': ['handle', 'cup'], 
            'pistol': ['barrel', 'handle', 'trigger and guard'], 
            'rocket': ['body','fin','nose cone'], 'skateboard': ['wheel','deck','belt for foot'], 'table': ['desktop','leg or support','drawer']}
    times = []
    accs = []
    ious = []
    for category in cat2part.keys():
        stime = time.time()
        evaluator = InstSegEvaluator('ViT-L/14@336px')
        labels = cat2part[category]
        print(labels)
        acc_all = []
        iou_all = []
        for id in ids:
            data_dir = f"/data/shapenetpart_rendered/{category}/{id}"
            acc, iou = evaluator.evaluate_full(labels, data_dir)
            acc_all.append(acc)
            iou_all.append(iou)
        mean_acc = np.mean(acc_all)
        mean_iou = np.mean(iou_all)
        print(mean_acc)
        print(mean_iou)
        etime = time.time()
        timep = (etime-stime)/5
        print((etime-stime)/5)
        f = open("shapenetp_inf.txt", "a")
        f.write(f"category {category}, acc {mean_acc}, iou {mean_iou}, time {timep}")
        f.close()
        accs.append(mean_acc)
        ious.append(mean_iou)
        times.append(timep)
    class_mean_acc = np.mean(accs)
    class_mean_iou = np.mean(ious)
    class_mean_time = np.mean(times)
    print(class_mean_acc)
    print(class_mean_iou)
    print(class_mean_time)
    f = open("shapenetp_inf.txt", "a")
    f.write(f"class mean acc {class_mean_acc}, class mean iou {class_mean_iou}, class mean time {class_mean_time}")
    f.close()

    
       