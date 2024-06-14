# HSICLassoForMe
Feature Selection Tool to Detect DNA Methylation Sites Associated with Cell Phenotype by HSICLasso.

## Description
### all_path.txt
Example of input file path and output for teacher data (DNA methylation data file).

### make_sh.py
Source code written with Python3 which used for making following shell script files.  
When you run this script, several arguments have to be needed; details are written in below "Usage" section.

### hlfm.py
Source code written with Python3 which used for selecting highly relevant methylation variation sites for output.

## Requirement
* pyHSICLasso (https://github.com/riken-aip/pyHSICLasso)

## Usage
1. Make all_path.txt
| column number | description |
|:--------------|:------------|
| 1             |Text or numeric for classification or regression labels|
| 2             |Absolute path of the data used for training|
| 3             |Group number for K-fold cross validation|

2. Setting HSIC Lasso parameters and creating executables file.
```
cd /working/directory/
python make_sh.py
```

3. Run hsiclasso.sh
arguments is already set in make_sh.py
```
sh hsiclasso.sh
```
