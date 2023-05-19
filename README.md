# Meta Omnium: A Benchmark for General-Purpose Learning-to-Learn
Meta-learning and other approaches to few-shot learning are widely studied for image recognition, and are increasingly applied to other vision tasks such as pose estimation and dense prediction. This naturally raises the question of whether there is any few-shot meta-learning algorithm capable of generalizing across these diverse task types? To support the community in answering this question, we introduce Meta Omnium, a dataset-of-datasets spanning multiple vision tasks including recognition, keypoint localization, semantic segmentation and regression. We experiment with popular few-shot meta-learning baselines and analyze their ability to generalize across tasks and to transfer knowledge between them. Meta Omnium enables meta-learning researchers to evaluate model generalization to a much wider array of tasks than previously possible, and provides a single framework for evaluating meta-learners across a wide suite of vision applications in a consistent manner.

[[Paper]](https://arxiv.org/abs/2305.07625)

## Data
You can download data from [here](https://drive.google.com/drive/folders/1NKb0uLJqmAauE9FY18T-qQ-T6k5LA_yt?usp=sharing) and put it in `data` directory (each unzipped dataset should have a directory in `data`).
Data is in the format shown on `Example_dataset` in `data` directory. We also include scripts that we have used for processing the original datasets.

We use the following splits of datasets:

Classification:
- Meta-train: BCT_Mini_Trn, BRD_Mini_Trn, CRS_Mini_Trn.
- Meta-val: 1) In-domain validation: BCT_Mini_Val, BRD_Mini_Val, CRS_Mini_Val, 2) Cross-domain validation: FLW_Mini, MD_MIX_Mini, PLK_Mini. 
- Meta-test: 1) In-domain testing: BCT_Mini_Test, BRD_Mini_Test, CRS_Mini_Test, 2) Cross-domain testing: PLT_VIL_Mini, RESISC_Mini, SPT_Mini, TEX_Mini.

Segmentation:
- Meta-train: FSS_Trn.
- Meta-val: 1) In-domain validation: FSS_Val, 2) Cross-domain validation: Vizwiz. 
- Meta-test: 1) In-domain testing: FSS_Test, 2) Cross-domain testing: PASCAL, PH2. 

Keypoint/pose estimation:
- Meta-train: Animal_Pose_Trn.
- Meta-val: 1) In-domain validation: Animal_Pose_Val, 2) Cross-domain validation: Synthetic_Animal_Pose.
- Meta-test: 1) In-domain testing: Animal_Pose_Test, 2) Cross-domain testing: MPII. 

Regression:
- Out-of-task testing: ShapeNet2D_Test, Distractor_Test, ShapeNet1D_Test, Pascal1D_Test.

## Required Libraries
Libraries that are needed: `numpy`, `torch`, `opencv`, `sklearn`, `scipy`, `tqdm`, `optuna` (for HPO)

## Experiments

Example script (for quick testing):

```
python metaomnium/trainers/train_cross_task_fsl.py --experiment_name dbg --best_hp_file_name protonet_cls_hpo --model protonet --n_way_eval 5 --max_shots_eval 5 --train_datasets FSS_Trn,BCT_Mini_Trn,BRD_Mini_Trn,CRS_Mini_Trn,Animal_Pose_Trn --val_id_datasets FSS_Val,BCT_Mini_Val,BRD_Mini_Val,CRS_Mini_Val,Animal_Pose_Val --val_od_datasets FLW_Mini,MD_MIX_Mini,PLK_Mini,Vizwiz,Synthetic_Animal_Pose --test_id_datasets FSS_Test,BCT_Mini_Test,BRD_Mini_Test,CRS_Mini_Test,Animal_Pose_Test --test_od_datasets PLT_VIL_Mini,RESISC_Mini,SPT_Mini,TEX_Mini,PASCAL,PH2,MPII --runs 1 --train_iters 30 --eval_iters 6
```

Full set of scripts to run the experiments is in `scripts/run_experiments.sh`. Scripts to run HPO are in `scripts/run_hpo` (note that hyperparameters from our HPO runs are available in `hpo_summaries` directory).

## Results
The results will be stored in the `summaries` directory, while HPO results will be stored in `hpo_summaries` directory.

## Citation
If you use our benchmark or code, please cite:
```
@inproceedings{bohdal2023metaomnium,
  title={Meta Omnium: A Benchmark for General-Purpose Learning-to-Learn},
  author={Bohdal, Ondrej and Tian, Yinbing and Zong, Yongshuo and Chavhan, Ruchika and Li, Da and Gouk, Henry and Guo, Li and Hospedales, Timothy},
  booktitle={CVPR},
  year={2023}
}
```

