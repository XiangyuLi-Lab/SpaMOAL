# SpaMOAL: A spatial multi-omics graph contrastive learning method for spatial domains identification
## Overview
Recent advances in spatial multi-omics technologies have opened new avenues for characterizing tissue architecture and 
function in situ, by simultaneously providing multimodal and complementary information—such as spatially resolved 
transcriptomic, epigenomic, and proteomic features. Current computational approaches face substantial challenges such 
as effective integration of multi-omics molecular information with spatial information and corresponding high
resolution histology images. To address this challenge, we proposed **SpaMOAL (Spatially Multi-Omics graph 
contrAstive Learning)**, a graph-based contrastive learning approach for spatial domain identification. SpaMOAL learns 
clustering-friendly representations from spatial multi-omics data by integrating spatial coordinates, histological image 
features and molecular profiles, enabling accurate delineation of spatial tissue domains. Benchmarking across multiple 
recent paired spatial multi-omics datasets demonstrated that SpaMOAL consistently outperforms existing methods. By 
enabling accurate spatial domain delineation, SpaMOAL provides a powerful framework for interpreting tissue 
organization and cellular microenvironments.   
<p align="center">
  <img width="1356" height="582" alt="Fig1" src="https://github.com/user-attachments/assets/04d157aa-6b60-40bb-ab8e-a042effef9ec" />

</p>


## Tutorial
#### Start by grabbing this source codes:
```
git clone https://github.com/XiangyuLi-Lab/SpaMOAL.git
cd SpaMOAL
```

### 2. Virtual environment
#### (Recommended) Using python virtual environment with  [`conda`](https://anaconda.org/)
```
# Configuring the virtual environment
conda env create -f environment.yml
conda activate SpaMOAL
```
### 3. Usage
#### 3-1 Image cutting (optional)
```
cd SpaMOAL/denoising
conda create -n muse
conda activate muse
pip install -r muse_requirement.txt
python deal_cut.py --name mouse_embryos --size 299
```
##### ```--name```: Name of sample
##### ```--size```: Image resolution
#### 3-2 Extraction of morphological features (optional)
```
python deal_inception.py --name mouse_embryos --size 299 --model resnet50  (or inception_resnet_v2 or inception_resnet_v3)
```
##### ```--name```: Name of sample
##### ```--size```: Image resolution
##### ```--model```: The Convolutional Neural Network used

#### 3-3  training
```
cd SpaMOAL
python main_old.py --dataset mouse_embryos --omics1 RNA --omics2 ATAC --input_folder /input/mouse_embryos/  --n_clusters 14  --num_iters 10 --omics3 image --num_view 3
```
##### ```--dataset```: Name of sample
##### ```--omics1```: Name of omics
##### ```--omics2```: Name of omics
##### ```--input_folder```: Path of feature input files
##### ```--n_clusters```: Number of clusters 
##### ```--num_iters```: Number of training iterations
##### ```--omics2```: Name of omics
##### ```--num_view```: Number of modals

#### 3-4 Clustering and Visualization
We provided specific examples of clustering and visualization in the notebook.

### 4. Download data
The simulated datasets is available at [https://github.com/XiangyuLi-Lab/SpaMOAL](https://github.com/XiangyuLi-Lab/SpaMOAL). The MISAR-seq mouse brain dataset is accessible at the National Genomics Data Center with accession number OEP003285. The spatial ATAC-RNA-seq mouse brain dataset can be found at [atlasxomics](https://web.atlasxomics.com/visualization/Fan). Spatial ATAC-RNA-seq mouse embryonic day 13 (E13) data reported in https://cells.ucsc.edu/?ds=brain-spatial-omics. 10x Visium human breast cancer gene and protein expression data can be found at https://www.10xgenomics.com/resources/datasets/gene-and-protein-expression-library-of-human-breast-cancer-cytassist-ffpe-2-standard.
### 5. Contact
Feel free to submit an issue or contact us at wangjinxia0116@163.com for problems about the packages.
