# SpaDAC: SPAtially embedded Deep Attentional graph Clustering
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
  [Uploading Fig1.tif…]()
</p>

## Tutorial
#### Start by grabbing this source codes:
```
git clone https://github.com/huoyuying/SpaDAC.git
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

#### 3-4  training
```
cd SpaMOAL
python main_old.py --dataset mouse_embryos --omics1 RNA --omics2 ATAC --input_folder /input/mouse_embryos/  --n_clusters 14  --num_iters 10 --omics3 image_unnormalize --num_view 3

```
##### ```--dataset```: Name of sample
##### ```--omics1```: Name of sample
##### ```--omics2```: Name of sample
##### ```--dataset```: Name of sample
##### ```--exp```: The number of highly variable features(HVGs) selected
##### ```--adj```: The 01-Matrix of whether cells are neighbors or not, based on geographical similarity
##### ```--img```: The 01-Matrix of whether cells are neighbors or not, based on morphological similarity
##### ```--max_epoch```: The number of iterations of this training

#### 3-6 Clustering and optimization
```
cd SpaDAC
python clustering.py
```
#### 3-7 Denoising of gene expression profile
```
cd SpaDAC/denoising
python denoising.py
```
### 4. Download data
|      Platform      |       Tissue     |    SampleID   |
|:----------------:|:----------------:|:------------:|
| [10x Visium](https://support.10xgenomics.com) | Human dorsolateral pre-frontal cortex (DLPFC) | [151507,](https://spatial-dlpfc.s3.us-east-2.amazonaws.com/h5/151507_filtered_feature_bc_matrix.h5) [151508,](https://spatial-dlpfc.s3.us-east-2.amazonaws.com/h5/151508_filtered_feature_bc_matrix.h5) [151509,](https://spatial-dlpfc.s3.us-east-2.amazonaws.com/h5/151509_filtered_feature_bc_matrix.h5) [151510,](https://spatial-dlpfc.s3.us-east-2.amazonaws.com/h5/151510_filtered_feature_bc_matrix.h5) [151669,](https://spatial-dlpfc.s3.us-east-2.amazonaws.com/h5/151669_filtered_feature_bc_matrix.h5) [151670,](https://spatial-dlpfc.s3.us-east-2.amazonaws.com/h5/151570_filtered_feature_bc_matrix.h5) [151671,](https://spatial-dlpfc.s3.us-east-2.amazonaws.com/h5/151671_filtered_feature_bc_matrix.h5) [151672,](https://spatial-dlpfc.s3.us-east-2.amazonaws.com/h5/151672_filtered_feature_bc_matrix.h5) [151673,](https://spatial-dlpfc.s3.us-east-2.amazonaws.com/h5/151673_filtered_feature_bc_matrix.h5) [151674,](https://spatial-dlpfc.s3.us-east-2.amazonaws.com/h5/151674_filtered_feature_bc_matrix.h5) [151675,](https://spatial-dlpfc.s3.us-east-2.amazonaws.com/h5/151675_filtered_feature_bc_matrix.h5) [151676](https://spatial-dlpfc.s3.us-east-2.amazonaws.com/h5/151676_filtered_feature_bc_matrix.h5)
| [10x Visium](https://support.10xgenomics.com) | Mouse brain section| [Sagittal-Anterior,](https://www.10xgenomics.com/resources/datasets/mouse-brain-serial-section-1-sagittal-anterior-1-standard-1-1-0) [Sagittal-Posterior](https://www.10xgenomics.com/resources/datasets/mouse-brain-serial-section-1-sagittal-posterior-1-standard-1-1-0)
| [10x Visium](https://support.10xgenomics.com) | Human breast cancer| [Ductal Carcinoma In Situ & Invasive Carcinoma](https://www.10xgenomics.com/resources/datasets/human-breast-cancer-ductal-carcinoma-in-situ-invasive-carcinoma-ffpe-1-standard-1-3-0) 
| [Stereo-Seq](https://www.biorxiv.org/content/10.1101/2021.01.17.427004v2) | Mouse olfactory bulb| [Olfactory bulb](https://github.com/BGIResearch/stereopy) 
| [ST](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE111672) |  Pancreatic ductal adenocarcinoma tissue| [PDAC1,](https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE111672&format=file&file=GSE111672%5FPDAC%2DA%2Dindrop%2Dfiltered%2DexpMat%2Etxt%2Egz) [PDAC2](https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE111672&format=file&file=GSE111672%5FPDAC%2DB%2Dindrop%2Dfiltered%2DexpMat%2Etxt%2Egz) 

Spatial transcriptomics data of other platforms can be downloaded https://www.spatialomics.org/SpatialDB/

### 5. Contact
Feel free to submit an issue or contact us at 21121732@bjtu.edu.cn for problems about the packages.
