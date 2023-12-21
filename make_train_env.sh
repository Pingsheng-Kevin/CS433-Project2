module load anaconda/3
conda create -n ffcv_ssl python=3.9 conda conda-libmamba-solver -c conda-forge
conda activate ffcv_ssl
export CONDA_EXE="$(hash -r; which conda)"
conda config --set solver libmamba
conda install cupy pkg-config compilers libjpeg-turbo opencv pytorch==1.11.0 torchvision==0.12.0 cudatoolkit=11.3 numba terminaltables matplotlib scikit-learn pandas assertpy pytz -c pytorch -c nvidia -c conda-forge
pip install -e FFCV-SSL/
# pip install ffcv==0.0.3
# pip install -r requirements.txt -e .
# pip install setuptools==59.5.0 --force 
