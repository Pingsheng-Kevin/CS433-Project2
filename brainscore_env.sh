module load anaconda/3

conda create --name brainscore python=3.7
conda activate brainscore
conda install h5py=1.12.2
conda install netcdf4

# some tools
conda install ipykernel ipython

# brainscore
git clone https://github.com/brain-score/sample-model-submission.git
cd sample-model-submission
pip install .
cd ..

# conda install pytorch==1.11.0 torchvision==0.12.0 cudatoolkit=11.3 -c pytorch

# # if you want to work on jupyter notebook
# which ipython
# ipython kernel install --user --name=brainscore_kernel