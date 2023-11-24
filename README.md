# migration_assay
Code for running migration assays on cellular microscope images

# Download sam checkpoints 
Located in this google drive.  Download to wherever this package is located. (same folder as this readme)
`https://drive.google.com/drive/folders/15gEc46zENDvvK1fjSG7OJDKVAuh3K9GP?usp=sharing`

# Install requirements
python needs to be installed.  I recommend creating a new conda environment with python 3.8, and then do
```
pip install torch==2.0.0+cpu torchvision==0.15.1+cpu torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cpu
pip install git+https://github.com/facebookresearch/segment-anything.git
pip install opencv-python pycocotools matplotlib onnxruntime onnx
```