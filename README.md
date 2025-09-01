

# *KCARD*
The PyTorch implementation for the KCARD: Knowledge Adaptation for Cross-Domain Moving Infrared Small Target Detection.
## 1. Pre-request
### 1.1. Environment
```bash
conda create -n KCARD python=3.10.11
conda activate KCARD
conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=11.3 -c pytorch

git clone --depth=1 https://github.com/xiaomingxige/KCARD
cd KCARD
pip install -r requirements.txt
```
### 1.2. DCNv2
#### Build DCNv2

```bash
cd nets/ops/dcn/
# You may need to modify the paths of cuda before compiling.
bash build.sh
```
#### Check if DCNv2 works (optional)

```bash
python simple_check.py
```
> The DCNv2 source files here is different from the [open-sourced version](https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch) due to incompatibility. [[issue]](https://github.com/open-mmlab/mmediting/issues/84#issuecomment-644974315)

### 1.3. Datasets
Our experiments are conducted on three datasets: **DAUB**, **IRDST**, and **ITSDT-15K**. Six domain adaptation experiments are performed:  **IRDST$\rightarrow$ DAUB**, **ITSDT-15K$\rightarrow$ DAUB**, **DAUB$\rightarrow$ IRDST**, **ITSDT-15K$\rightarrow$ IRDST**, **DAUB$\rightarrow$ ITSDT-15K**, and **IRDST$\rightarrow$ ITSDT-15K**. 
We would like to thank [SSTNet](https://github.com/UESTC-nnLab/SSTNet) and [Tridos](https://github.com/UESTC-nnLab/Tridos) for providing the datasets download links:
- **DAUB**: [Download Link](https://pan.baidu.com/s/1nNTvjgDaEAQU7tqQjPZGrw?pwd=saew) (Extraction Code: saew)
- **IRDST**: [Download Link](https://pan.baidu.com/s/1igjIT30uqfCKjLbmsMfoFw?pwd=rrnr) (Extraction Code: rrnr)
- **ITSDT-15K**: [Download Link](https://drive.google.com/file/d/1nnlXK0QCoFqToOL-7WdRQCZfbGJvHLh2/view?usp=sharing)

We also provide the 1% target training samples used in the paper within this repository.
## 2. Train
Taking **ITSDT-15K$\rightarrow$ IRDST** as an example, you can use the following command:
```bash
CUDA_VISIBLE_DEVICES=0 nohup python -u  ITSDT_to_IRDST.py >  ITSDT_to_IRDST.out &
```
> Please modify the corresponding file path in train.py before training.

For other transfer scenarios, you can proceed in a similar manner after modifying the corresponding file paths.
## 3. Test
We utilize 1 NVIDIA GeForce RTX 4090D GPU for testing. For the **ITSDT-15K$\rightarrow$ IRDST**：
```bash
python vid_ITSDT_to_IRDST.py
```
## Citation
If you find this project is useful for your research, please cite:

```bash
@ARTICLE{11145131,
  author={Luo, Dengyan and Xiang, Yanping and Wang, Hu and Ji, Luping and Ye, Mao},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={Knowledge Adaptation for Cross-Domain Moving Infrared Small Target Detection}, 
  year={2025},
  volume={},
  pages={1-1},
  doi={10.1109/TGRS.2025.3604069}
  }

```

## 4. Visualization
For the **ITSDT-15K$\rightarrow$ IRDST**：
```bash
python vid_predict_ITSDT_to_IRDST.py
```
## Acknowledgements
This work is based on [SSTNet](https://github.com/UESTC-nnLab/SSTNet), [STDF-Pytoch](https://github.com/ryanxingql/stdf-pytorch), [MGANet](https://github.com/mengab/MGANet-DCC2020), and [MobileSAM](https://github.com/ChaoningZhang/MobileSAM). Thank them for sharing the codes.
