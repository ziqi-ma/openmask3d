#!/bin/bash
export OMP_NUM_THREADS=3  # speeds up MinkowskiEngine
set -e

# RUN OPENMASK3D FOR A SINGLE SCENE
# This script performs the following:
# 1. Compute class agnostic masks and save them
# 2. Compute mask features for each mask and save them

# --------
# NOTE: SET THESE PARAMETERS BASED ON YOUR SCENE!
# data paths

declare -A my_dict

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

cd openmask3d
cats=("WashingMachine" "Window")
for cat in ${cats[@]};
do
    start=`date +%s`
    list="${my_dict[$cat]}"
    IFS=' ' read -r -a array <<< "$list"
    echo "$cat"
    echo "$cat" >> feat_time.txt

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
        OUTPUT_DIRECTORY="${SCENE_DIR}"
        TIMESTAMP=$(date +"%Y-%m-%d-%H-%M-%S")
        OUTPUT_FOLDER_DIRECTORY="${OUTPUT_DIRECTORY}"
        SAVE_VISUALIZATIONS=false #if set to true, saves pyviz3d visualizations
        SAVE_CROPS=false 
        # gpu optimization
        OPTIMIZE_GPU_USAGE=false

        # 1. Compute class agnostic masks and save them
        echo "[INFO] Extracting class agnostic masks..."
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
        echo "[INFO] Mask computation done!"

        # get the path of the saved masks
        MASK_FILE_BASE=$(echo $SCENE_PLY_PATH | sed 's:.*/::')
        MASK_FILE_NAME=${MASK_FILE_BASE/.ply/_masks.pt}
        SCENE_MASK_PATH="${OUTPUT_FOLDER_DIRECTORY}/${MASK_FILE_NAME}"
        echo "[INFO] Masks saved to ${SCENE_MASK_PATH}."

        # 2. Compute mask features for each mask and save them
        echo "[INFO] Computing mask features..."

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
    echo $runtime/5 >> feat_time.txt
done


