set -e
pip install pip==23.2
# OpenMask3D Installation
#   - If you encounter any problem with the Detectron2 or MinkowskiEngine installations, 
#     it might be because you don't have properly set up gcc, g++, pybind11, openblas installations.
#     First, make sure you have those are installed properly. 
#   - More details about installing on different platforms can be found in the GitHub repositories of 
#     Detectron2(https://github.com/facebookresearch/detectron2) and MinkowskiEngine (https://github.com/NVIDIA/MinkowskiEngine).
#   - If you encounter any other problems, take a look at the installation guidelines in https://github.com/JonasSchult/Mask3D, which might be helpful as our mask module relies on Mask3D.

# Note: The following commands were tested on Ubuntu 18.04 and 20.04, with CUDA 11.1 and 11.4.

pip install torch==1.12.1 torchvision==0.13.1 -f https://download.pytorch.org/whl/cu113/torch_stable.html
pip install ninja==1.10.2.3
pip install pytorch-lightning==1.7.2 fire==0.5.0 imageio==2.23.0 tqdm==4.64.1 wandb==0.13.2
pip install python-dotenv==0.21.0 pyviz3d==0.2.32 scipy==1.9.3 plyfile==0.7.4 scikit-learn==1.2.0 trimesh==3.17.1 loguru==0.6.0 albumentations==1.3.0 volumentations==0.1.8
pip install antlr4-python3-runtime==4.8 black==21.4b2 omegaconf==2.0.6 hydra-core==1.0.5 --no-deps