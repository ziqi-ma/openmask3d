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

start=`date +%s`
cd openmask3d
#catids=('monitor_computer_equipment_computer_monitor_b26a53419075442ca284cdf1d5541765' 'motor_scooter_69ad00f6688749de9934481f3e7ea669' 'motor_scooter_c1511756307a4c63af12c67fef603dec' 'motor_vehicle_62c046fe80614cd09d66b792b7947569' 'pepper_ea9f3e4b09d048988864d6d373567b9b' 'pickup_truck_5e77a057585d47309d2605934197669f' 'saucepan_fda8f9c3326745ac840f7115e50a8ce8' 'sherbert_9ab7bd74b12d44d1928c6b0b08a92b85' 'shopping_cart_945c394dd4a24602af03d75d5b762932' 'shopping_cart_c22286ade5ea493b890b3269e57b191f' 'silo_a2e4ec6c85aa4908a365c3ec909bf9e3' 'slide_fa56c3d537e940bf98dfcbb43801d8f9' 'sunglasses_d0827a414b0540bbb9fa2d6ff3f38e16' 'teddy_bear_ad89f59278e5408a8dd8112dc07929e6' 'telephone_fba528bb5a1541a8961c6e805c473108' 'timer_be07b36086df441291df751c6d528d2f' 'umbrella_798ffbb44f334de79b6f7185d762cce0' 'wagon_dc0bb314a28d40bda1af9de1b012db8f' 'wineglass_53b9bc12a01c4b1688a4b560c9f5548d' 'parakeet_ee88784602ce42daba6824b19e87987a' 'peach_0289474b7e6c4f0882275bc1ccde6b5c' 'power_shovel_673a0ce082ad42728f90f3649a6fc727' 'rhinoceros_19541e167655499b9e15dca3798b8b68' 'shirt_a7440e534ef6444895a6cc5feaf15a33' 'taco_705cd9de52814dd4b79e7dad46516b37')
catids=('apple_0176be079c2449e7aaebfb652910a854' 'bookcase_7a545709aa98429a9b30f812f70193f7' 'bookcase_97c1c0c9ec6e49b993b37efca14fa830' 'chair_79fbc9501aff4538bf077197d2502016' 'chair_e83dc29e69e94ca7954c0b25d52ea3b2' 'fireplug_1a10ecd4e72746109ee29a8631ea4d70' 'gun_19f1c00170c14dfa900f563bd05c5dac' 'gun_8a193bd9d7944b5baf2031f5fb80a7a3' 'lampshade_ed82ecb06ef9438fa83a2584b24d9f14' 'mug_c25119c5ac6e4654be3b75d78e34a912' 'mushroom_33172c4e091e4af8bfa0071363ac6019' 'ring_65586b5f86b5481abb692b94ee709112' 'shoe_c5c2133684f646e9940ddb2ff7f81334' 'skateboard_b73497d937774c96a92f05947c738b9b' 'snowman_c55eff0309a14cf09423d238900cc7c2' 'snowman_f769c1ea1ce84566a3a9c0b5f4388c0f' 'spectacles_a4c0eaf66bf748c5a2a1d6bb595672c7' 'armor_cf2e364e14884a2c982a2964ac735b76' 'fire_extinguisher_dc429020e35e4fd88937d9e12614ea5c' 'car_automobile_33657ac0e044493daf495d45424ead7a' 'car_automobile_ec042ced12384aa6924f44c7876a8faf')
for catid in ${catids[@]};
do
    SCENE_DIR="/data/objaverse/holdout/seenclass/${catid}"
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
    CUDA_VISIBLE_DEVICES=3 python class_agnostic_mask_computation/get_masks_single_scene.py \
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
echo $runtime/21 >> feat_time.txt



