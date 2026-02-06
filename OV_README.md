# OpenVINO enable MaskDINO

## Step 1. Init Installation

- Linux with Python ≥ 3.10
- PyTorch ≥ 2.0 and [torchvision](https://github.com/pytorch/vision/) that matches the PyTorch installation.
  Install them together at [pytorch.org](https://pytorch.org) to make sure of this. Note, please check
  PyTorch version matches that is required by Detectron2.
- Detectron2: Detectron2 requires the [custom version](https://github.com/18582088138/openvino_detectron2/tree/ov_enable) that supports OpenVINO, and the install step pls follow [Detectron2 installation instructions](https://detectron2.readthedocs.io/tutorials/install.html). 
- OpenCV is optional but needed by demo and visualization
- `pip install -r requirements.txt`


### CUDA kernel for MSDeformAttn
After preparing the required environment, run the following command to compile CUDA kernel for MSDeformAttn:

`CUDA_HOME` must be defined and points to the directory of the installed CUDA toolkit.

```bash
cd maskdino/modeling/pixel_decoder/ops
sh make.sh
```

## Step 2. conda environment setup 
```bash
conda create --name ov_maskdino python=3.12 -y
conda activate ov_maskdino
pip install torch torchvision opencv-python

# under your working directory
cd <WORKING_DIR>
git clone -b ov_enable https://github.com/18582088138/openvino_detectron2.git
cd openvino_detectron2
pip install -e . --no-build-isolation
pip install git+https://github.com/cocodataset/panopticapi.git
pip install git+https://github.com/mcordts/cityscapesScripts.git

cd <WORKING_DIR>
git clone -b ov_maskdino https://github.com/18582088138/OpenVINO_MaskDINO.git
cd OpenVINO_MaskDINO
pip install -r requirements.txt
cd maskdino/modeling/pixel_decoder/ops
sh make.sh
cd <MASKDINO_ROOT_DIR>
```

## Step 3. OpenVINO MaskDINO Inference 
```bash
# prepare pytorch model file
cd <MASKDINO_ROOT_DIR>
wget https://github.com/IDEA-Research/detrex-storage/releases/download/maskdino-v0.1.0/maskdino_r50_50ep_300q_hid1024_3sd1_instance_maskenhanced_mask46.1ap_box51.5ap.pth

# MaskDINO model convert (pth -> onnx -> OV IR)
cd demo
# onnx model export 
python demo.py --config-file ../configs/coco/instance-segmentation/maskdino_R50_bs16_50ep_3s.yaml --input input1.jpg --output result.jpg --onnx-export  --opts MODEL.WEIGHTS ../maskdino_r50_50ep_300q_hid1024_3sd1_instance_maskenhanced_mask46.1ap_box51.5ap.pth MODEL.DEVICE cpu

# OV IR model convert
ovc maskdino.onnx --output_model maskdino.xml 

# OV Demo Inference (CPU)
python demo.py --config-file ../configs/coco/instance-segmentation/maskdino_R50_bs16_50ep_3s.yaml --input input1.jpg --output result.jpg --ov-infer --ov-device CPU  --opts MODEL.WEIGHTS ../maskdino_r50_50ep_300q_hid1024_3sd1_instance_maskenhanced_mask46.1ap_box51.5ap.pth MODEL.DEVICE cpu

# OV Demo Inference (GPU)
python demo.py --config-file ../configs/coco/instance-segmentation/maskdino_R50_bs16_50ep_3s.yaml --input input1.jpg --output result.jpg --ov-infer --ov-device GPU  --opts MODEL.WEIGHTS ../maskdino_r50_50ep_300q_hid1024_3sd1_instance_maskenhanced_mask46.1ap_box51.5ap.pth MODEL.DEVICE cpu
```