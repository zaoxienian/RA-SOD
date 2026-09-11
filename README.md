# (ECCV 2026) RA-SOD: Reliability-Aware RGB-T Salient Object Detection under Modality Degradation

Hongbo Gao, Zhengyu Li, Xueru Nie, Dihao Zhu, Lijun Zhao, Yunke Wang, and Chang Xu.

Official PyTorch implementation of **RA-SOD: Reliability-Aware RGB-T Salient Object Detection under Modality Degradation**.

## Framework

RA-SOD estimates RGB and thermal modality reliability, uses uncertainty to guide dual-stream decoding, and adaptively fuses the two predictions under modality degradation.

![Examples of RGB and thermal modality degradation](figs/intro.png)

The complete network diagram is available [here](figs/method.pdf).

## Demo

A qualitative video demo is available [here](assets/RASOD.mp4).

## Datasets

- [VT5000](https://drive.google.com/drive/folders/1So0dHK5-aKj1t6OmFhRGLh_0nsXbldZE?usp=sharing)
- [VT1000](https://drive.google.com/drive/folders/1kEGOuljxKxIYwH54sNH_Wqmw7Sf7tTw5?usp=sharing)
- [VT821](https://drive.google.com/drive/folders/1gjTRVwvTNL0MJaJwS6vkpoi5rGyxIh41?usp=sharing)
- [VT-IMAG](https://drive.google.com/file/d/1xzvqoYLrmJ-6x33DygCP-LhFNYfhQL-u/view?usp=sharing)

Each dataset directory should contain aligned `RGB`, `T`, and `GT` folders. The model is trained on the VT5000 training set and tested on VT821, VT1000, VT5000, and VT-IMAG.

## How to run

The reproduced environment uses Python 3.7.13, PyTorch 1.13.1, torchvision 0.14.1, and CUDA 11.7.

```bash
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu117
```

The ImageNet-pretrained Res2Net-50 backbone is downloaded automatically when first constructing the model.

### Training

```bash
python train.py \
  --rgb_label_root ./datasets/VT5000/Train/RGB/ \
  --thermal_label_root ./datasets/VT5000/Train/T/ \
  --gt_label_root ./datasets/VT5000/Train/GT/ \
  --gpu_id 0 \
  --save_path ./Checkpoints/
```

Checkpoints are saved as `RASOD_epoch_*.pth` under `./Checkpoints/`.

### Testing

Download [`RASOD_best.pth`](https://pan.baidu.com/s/1pI3NyXngcYa-beolyq5fmw?pwd=sbjb) (extraction code: `sbjb`) and place it in `./Checkpoints/`.

```bash
python test.py \
  --test_path ./datasets \
  --model_path ./Checkpoints/RASOD_best.pth \
  --gpu_id 0
```

Prediction maps are saved to `./Predict_maps/<dataset>/`.

## Evaluation

We use the [Saliency-Evaluation-Toolbox](https://github.com/jiwei0921/Saliency-Evaluation-Toolbox) to evaluate the prediction maps.

## Citation

Please cite our paper if you find this work useful:

```bibtex
@inproceedings{gao2026rasod,
  title     = {RA-SOD: Reliability-Aware RGB-T Salient Object Detection under Modality Degradation},
  author    = {Gao, Hongbo and Li, Zhengyu and Nie, Xueru and Zhu, Dihao and Zhao, Lijun and Wang, Yunke and Xu, Chang},
  booktitle = {European Conference on Computer Vision},
  year      = {2026}
}
```

## Acknowledgement

This implementation is built on [ConTriNet](https://github.com/CSer-Tang-hao/ConTriNet_RGBT-SOD) and [Res2Net](https://github.com/Res2Net/Res2Net-PretrainedModels). We thank the authors for releasing their code.

## License

RA-SOD is released under the [MIT License](LICENSE). Res2Net-derived components retain their original noncommercial license terms; see [Third-Party Notices](THIRD_PARTY_NOTICES.md).
