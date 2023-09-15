## This is a guide to run the K-mer pipeline and get models going

* **before running the pipeline you need to**

I. clone kmer pipeline in github

```
https://github.com/Thilanka-lt/pCRE_identification_mod.git
```

II. Activate environments

```
conda create -n pCR_identification python=3.6.4
conda activate pCR_identification
pip install scikit-learn==0.23.1
pip install imbalanced-learn==0.7.0
pip install matplotlib==3.3.2
pip install pandas==1.1.5
pip install numpy==1.18.5
pip install statsmodels==0.12.1
pip install joblib==0.17.0
pip install scipy==1.5.4
```

* Once you set up your environment you can activate environment in a shell script using : conda activate ***path to environment**/pCR_identification*

Ex:
```
conda activate /mnt/home/ranawee1/anaconda3/envs/pCR_identification
```

* You can install required packages for the virtual environment using the requirement.txt file in the repo

```
conda create -n pCR_identification python=3.6.4
conda activate sc_dim_red
pip install -r requirement.txt
```

### 1. Set Up Your Files:

#### 1.a. Inside directory where your model is going to run make sub directories for FASTA files and Motif Lists:
```
mkdir FastaFiles
mkdir MotifLists
``` 
#### 1.b. Copy/generate fasta file (file that includes sequence information) for all the genes including the region of interest to FastaFiles

```
python /mnt/gs21/scratch/ranawee1/05_Singel_cell/scripts/00_extract_regions_from_gff.py -g TAIR10_GFF3_genes.gff -c TAIR10_chr_all.fas -u 1000 -d 0 -r 3 -i 500
```
 * you can also copy from 
 ```
 /mnt/gs21/scratch/ranawee1/05_Singel_cell/data/TAIR10_chr_all_upstream_1000_ingene_500_from_TSS_downstream_0.fas
 ```
#### 1.c. copy your negative and positive examples files to FastaFiles directory

* These are text files with gene names

Ex:
```
AT1G01070
AT1G01140
AT1G01230
AT1G01440
AT1G01470
AT1G01550
AT1G01620
```

#### 1.d Make singleton and paired kmer list. Reverse complement sequences are separated by “.”, pairs separated by a space (k = length of kmer you want):

* First navigate to MotifLists directory (you can dedicate a directory to save all the kmer lists and call that directory using -path_m flag once you are running the pipeline)
* Then,

```
python ./Pairwise_kmers.py -f make_pairs2 –k 6
```
* change the -k flag to generate kmers of preferred length

### 2. Running the model:

* run this code in the base directory

```
python /mnt/gs21/scratch/ranawee1/Maddy_Rajneesh_prj/Motif_dicovary/pCRE_identification/000_pipeline_for_MotifDiscovery_stratified_for_test.py -pos_ID positive_set.txt -neg_ID negative_set.txt -fasta TAIR10_chr_all.fas_upstream_1000_ingene_500_downstream_0.fas -bal n -holdout 0.2 -kmers 5mers_withRC.txt,6mers_withRC.txt,7mers_withRC.txt -pval 0.05 -smote y -stratified_for_test n -path_p /mnt/gs21/scratch/ranawee1/Maddy_Rajneesh_prj/Motif_dicovary/pCRE_identification -path_k /mnt/gs21/scratch/ranawee1/Maddy_Rajneesh_prj/Motif_dicovary/models/MotifLists -short n
```



