#!/bin/bash
export OMP_NUM_THREADS=3  # speeds up MinkowskiEngine
export CUDA_VISIBLE_DEVICES=2
set -e

# RUN OPENMASK3D FOR A SINGLE SCENE
# This script performs the following:
# 1. Compute class agnostic masks and save them
# 2. Compute mask features for each mask and save them

# --------
# NOTE: SET THESE PARAMETERS BASED ON YOUR SCENE!
# data paths

declare -A my_dict

my_dict['Bottle']="4500 4403 5688 6209 3868"
my_dict['Box']="100174 100685 100658 100426 48492"
my_dict['Bucket']="102352 100462 100438 100442 100431"
my_dict['Camera']="102520 102532 102432 101352 102417"
my_dict['Cart']="102556 101075 102551 101072 100501"
my_dict['Chair']="762 38803 43142 42452 40982"
my_dict['Clock']="7074 7037 7111 6839 6613"
my_dict["CoffeeMachine"]="103084 103069 103129 103038 103140"
my_dict['Dishwasher']="12552 12617 12484 11661 12536"
my_dict['Dispenser']="101566 103356 103419 101528 101546"
my_dict["Display"]="4633 5050 4681 4563 4562"
my_dict["Door"]="8919 9065 9107 9164 8983"
my_dict['Eyeglasses']="102587 102603 102567 101284 102588"
my_dict['Faucet']="168 1832 991 1931 960"
my_dict["FoldingChair"]="100568 100587 100598 100579 102333"
my_dict["Globe"]="100798 100768 100793 100779 100763"
my_dict["Kettle"]="103222 102714 102761 102768 102753"
my_dict["Keyboard"]="12738 12917 13024 12999 13082"
my_dict["KitchenPot"]="100623 102085 100033 100056 100017"
my_dict["Knife"]="102401 101236 101659 101108 103728"
my_dict["Lamp"]="16794 15425 13525 16012 14567"
my_dict["Laptop"]="9748 11395 11888 10707 10213"
my_dict["Lighter"]="100317 100335 100340 100343 100348"
my_dict["Microwave"]="7263 7273 7296 7167 7292"
my_dict["Mouse"]="103025 102276 103022 102272 102273"
my_dict["Oven"]="101921 101773 101971 7187 101946"
my_dict["Pen"]="102944 102917 102960 102911 101736"
my_dict["Phone"]="103251 103917 103593 103813 103347"
my_dict["Pliers"]="100142 102243 100150 102285 102253"
my_dict["Printer"]="103863 104020 104000 103972 103894"
my_dict["Refrigerator"]="10347 10627 10612 11260 10849"
my_dict["Remote"]="100816 101139 100408 104041 101028"
my_dict["Safe"]="102318 101611 101584 102381 101593"
my_dict["Scissors"]="10960 10902 11113 10895 10495"
my_dict["Stapler"]="103095 103113 103792 103789 103283"
my_dict["StorageFurniture"]="45948 45524 47238 48513 47585"
my_dict["Suitcase"]="100839 101668 101673 100249 103757"
my_dict["Switch"]="100883 100882 100849 100937 100871"
my_dict["Table"]="32566 22692 32259 26652 26545"
my_dict["Toaster"]="103556 103482 103560 103558 103502"
my_dict["Toilet"]="102703 102645 102648 101319 102710"
my_dict["TrashCan"]="102182 100731 12231 102259 103013"
my_dict["USB"]="100061 100123 100078 102009 100064"
my_dict["WashingMachine"]="100283 103518 103776 103781 100282"
my_dict["Window"]="100982 103050 103315 103323 102984"

cd openmask3d
cats=('Bottle' 'Box' 'Bucket' 'Camera' 'Cart' 'Chair' 'Clock' "CoffeeMachine" 'Dishwasher' 'Dispenser' "Display" 'Eyeglasses' 'Faucet' "FoldingChair" "Globe" "Kettle" "Keyboard" "KitchenPot" "Knife" \
"Lamp" "Laptop" "Lighter" "Microwave" "Mouse" "Oven" "Pen" "Phone" "Pliers" "Printer" "Refrigerator" "Remote" "Safe" "Scissors" "Stapler" "StorageFurniture" "Suitcase" "Switch" "Table" "Toaster" \
"Toilet" "TrashCan" "USB" "WashingMachine" "Window" "Door")
all_start=`date +%s`
for cat in ${cats[@]};
do
    start=`date +%s`
    list="${my_dict[$cat]}"
    IFS=' ' read -r -a array <<< "$list"
    echo "$cat"
    echo "$cat" >> partnete_feat_time.txt

    # Print all values in the array
    for id in "${array[@]}";
    do
        SCENE_DIR="/data/partnet-mobility/test/${cat}/${id}"
        SCENE_POSE_DIR="${SCENE_DIR}/extrinsics"
        SCENE_INTRINSIC_PATH="${SCENE_DIR}/intrinsics/intrinsic.txt"
        SCENE_INTRINSIC_RESOLUTION="[640,640]" # change if your intrinsics are based on another resolution
        SCENE_PLY_PATH="${SCENE_DIR}/pc_zup.ply"
        SCENE_COLOR_IMG_DIR="${SCENE_DIR}/rendered_img"
        SCENE_DEPTH_IMG_DIR="${SCENE_DIR}/depth"
        IMG_EXTENSION=".jpeg"
        DEPTH_EXTENSION=".png"
        DEPTH_SCALE=1
        # model ckpt paths
        MASK_MODULE_CKPT_PATH="/data/checkpoints/semseg3d/openmask3d_scannet200_model.ckpt"
        SAM_CKPT_PATH="/data/checkpoints/semseg3d/sam_vit_h_4b8939.pth"
        # output directories to save masks and mask features
        EXPERIMENT_NAME="experiment"
        OUTPUT_DIRECTORY="${SCENE_DIR}/openmask3drotated"
        TIMESTAMP=$(date +"%Y-%m-%d-%H-%M-%S")
        OUTPUT_FOLDER_DIRECTORY="${OUTPUT_DIRECTORY}"
        SAVE_VISUALIZATIONS=false #if set to true, saves pyviz3d visualizations
        SAVE_CROPS=false 
        # gpu optimization
        OPTIMIZE_GPU_USAGE=false

        # 1. Compute class agnostic masks and save them
        #echo "[INFO] Extracting class agnostic masks..."
        python class_agnostic_mask_computation/get_masks_single_scene.py \
        general.experiment_name=${EXPERIMENT_NAME} \
        general.checkpoint=${MASK_MODULE_CKPT_PATH} \
        general.train_mode=false \
        data.test_mode=test \
        model.num_queries=120 \
        general.use_dbscan=true \
        general.dbscan_eps=0.95 \
        general.save_visualizations=${SAVE_VISUALIZATIONS} \
        general.scene_path=${SCENE_PLY_PATH} \
        general.mask_save_dir="${OUTPUT_FOLDER_DIRECTORY}" \
        hydra.run.dir="${OUTPUT_FOLDER_DIRECTORY}/hydra_outputs/class_agnostic_mask_computation" 
        #echo "[INFO] Mask computation done!"

        # get the path of the saved masks
        MASK_FILE_BASE=$(echo $SCENE_PLY_PATH | sed 's:.*/::')
        MASK_FILE_NAME=${MASK_FILE_BASE/.ply/_masks.pt}
        SCENE_MASK_PATH="${OUTPUT_FOLDER_DIRECTORY}/${MASK_FILE_NAME}"
        #echo "[INFO] Masks saved to ${SCENE_MASK_PATH}."

        # 2. Compute mask features for each mask and save them
        #echo "[INFO] Computing mask features..."

        python compute_features_single_scene.py \
        data.masks.masks_path=${SCENE_MASK_PATH} \
        data.camera.poses_path=${SCENE_POSE_DIR} \
        data.camera.intrinsic_path=${SCENE_INTRINSIC_PATH} \
        data.camera.intrinsic_resolution=${SCENE_INTRINSIC_RESOLUTION} \
        data.depths.depths_path=${SCENE_DEPTH_IMG_DIR} \
        data.depths.depth_scale=${DEPTH_SCALE} \
        data.depths.depths_ext=${DEPTH_EXTENSION} \
        data.images.images_path=${SCENE_COLOR_IMG_DIR} \
        data.images.images_ext=${IMG_EXTENSION} \
        data.point_cloud_path=${SCENE_PLY_PATH} \
        output.output_directory=${OUTPUT_FOLDER_DIRECTORY} \
        output.save_crops=${SAVE_CROPS} \
        hydra.run.dir="${OUTPUT_FOLDER_DIRECTORY}/hydra_outputs/mask_features_computation" \
        external.sam_checkpoint=${SAM_CKPT_PATH} \
        gpu.optimize_gpu_usage=${OPTIMIZE_GPU_USAGE}
        #echo "[INFO] Feature computation done!"
        
    done
    end=`date +%s`
    runtime=$((end-start))
    echo $runtime >> partnete_feat_time.txt
done
all_end=`date +%s`
fulltime=$((all_end-all_start))
echo $fulltime >> partnete_feat_time.txt


