#!/bin/sh

META_OMNIUM_DIR=/disk/meta-omnium

# single task classification
ARGS=(
"--experiment_name maml_cls_hpo --model maml --n_way_eval 5 --max_shots_eval 5 --train_datasets BCT_Mini_Trn,BRD_Mini_Trn,CRS_Mini_Trn --val_id_datasets BCT_Mini_Val,BRD_Mini_Val,CRS_Mini_Val --val_od_datasets FLW_Mini,MD_MIX_Mini,PLK_Mini --T 5 --T_val 10 --T_test 10 --runs 1 --train_iters 5000 --num_samples 30 --eval_iters 100 --root_dir ${META_OMNIUM_DIR}"
"--experiment_name protomaml_cls_hpo --model protomaml --n_way_eval 5 --max_shots_eval 5 --train_datasets BCT_Mini_Trn,BRD_Mini_Trn,CRS_Mini_Trn --val_id_datasets BCT_Mini_Val,BRD_Mini_Val,CRS_Mini_Val --val_od_datasets FLW_Mini,MD_MIX_Mini,PLK_Mini --T 5 --T_val 10 --T_test 10 --runs 1 --train_iters 5000 --num_samples 30 --eval_iters 100 --root_dir ${META_OMNIUM_DIR}"
"--experiment_name metacurvature_cls_hpo --model metacurvature --n_way_eval 5 --max_shots_eval 5 --train_datasets BCT_Mini_Trn,BRD_Mini_Trn,CRS_Mini_Trn --val_id_datasets BCT_Mini_Val,BRD_Mini_Val,CRS_Mini_Val --val_od_datasets FLW_Mini,MD_MIX_Mini,PLK_Mini --T 5 --T_val 10 --T_test 10 --runs 1 --train_iters 5000 --num_samples 30 --eval_iters 100 --root_dir ${META_OMNIUM_DIR}"
"--experiment_name protonet_cls_hpo --model protonet --n_way_eval 5 --max_shots_eval 5 --train_datasets BCT_Mini_Trn,BRD_Mini_Trn,CRS_Mini_Trn --val_id_datasets BCT_Mini_Val,BRD_Mini_Val,CRS_Mini_Val --val_od_datasets FLW_Mini,MD_MIX_Mini,PLK_Mini --runs 1 --train_iters 5000 --num_samples 30 --eval_iters 100 --root_dir ${META_OMNIUM_DIR}"
"--experiment_name ddrr_cls_hpo --model ddrr --n_way_eval 5 --max_shots_eval 5 --train_datasets BCT_Mini_Trn,BRD_Mini_Trn,CRS_Mini_Trn --val_id_datasets BCT_Mini_Val,BRD_Mini_Val,CRS_Mini_Val --val_od_datasets FLW_Mini,MD_MIX_Mini,PLK_Mini --runs 1 --train_iters 5000 --num_samples 30 --eval_iters 100 --root_dir ${META_OMNIUM_DIR}"
"--experiment_name proto_finetuning_cls_hpo --model proto_finetuning --n_way_eval 5 --max_shots_eval 5 --train_datasets BCT_Mini_Trn,BRD_Mini_Trn,CRS_Mini_Trn --val_id_datasets BCT_Mini_Val,BRD_Mini_Val,CRS_Mini_Val --val_od_datasets FLW_Mini,MD_MIX_Mini,PLK_Mini --T 20 --T_val 20 --T_test 20 --runs 1 --train_iters 5000 --num_samples 30 --eval_iters 100 --root_dir ${META_OMNIUM_DIR}"
"--experiment_name finetuning_cls_hpo --model finetuning --n_way_eval 5 --max_shots_eval 5 --train_datasets BCT_Mini_Trn,BRD_Mini_Trn,CRS_Mini_Trn --val_id_datasets BCT_Mini_Val,BRD_Mini_Val,CRS_Mini_Val --val_od_datasets FLW_Mini,MD_MIX_Mini,PLK_Mini --T 20 --T_val 20 --T_test 20 --runs 1 --train_iters 5000 --num_samples 30 --eval_iters 100 --root_dir ${META_OMNIUM_DIR}"
"--experiment_name linear_readout_cls_hpo --model finetuning --n_way_eval 5 --max_shots_eval 5 --train_datasets BCT_Mini_Trn,BRD_Mini_Trn,CRS_Mini_Trn --val_id_datasets BCT_Mini_Val,BRD_Mini_Val,CRS_Mini_Val --val_od_datasets FLW_Mini,MD_MIX_Mini,PLK_Mini --T 20 --T_val 20 --T_test 20 --freeze --runs 1 --train_iters 5000 --num_samples 30 --eval_iters 100 --root_dir ${META_OMNIUM_DIR}"
"--experiment_name tfs_cls_hpo --model tfs --n_way_eval 5 --max_shots_eval 5 --train_datasets BCT_Mini_Trn,BRD_Mini_Trn,CRS_Mini_Trn --val_id_datasets BCT_Mini_Val,BRD_Mini_Val,CRS_Mini_Val --val_od_datasets FLW_Mini,MD_MIX_Mini,PLK_Mini --T 20 --T_val 20 --T_test 20 --runs 1 --train_iters 5000 --num_samples 30 --eval_iters 100 --root_dir ${META_OMNIUM_DIR}"
)

for ARG in ${ARGS[@]}
do
python metaomnium/trainers/cross_task_fsl_hpo.py ${ARG}
done

# single task segmentation
ARGS=(
"--experiment_name maml_seg_hpo --model maml --n_way_eval 5 --max_shots_eval 5 --train_datasets FSS_Trn --val_id_datasets FSS_Val --val_od_datasets Vizwiz --T 5 --T_val 10 --T_test 10 --runs 1 --train_iters 5000 --num_samples 30 --eval_iters 100 --root_dir ${META_OMNIUM_DIR}"
"--experiment_name protomaml_seg_hpo --model protomaml --n_way_eval 5 --max_shots_eval 5 --train_datasets FSS_Trn --val_id_datasets FSS_Val --val_od_datasets Vizwiz --T 5 --T_val 10 --T_test 10 --runs 1 --train_iters 5000 --num_samples 30 --eval_iters 100 --root_dir ${META_OMNIUM_DIR}"
"--experiment_name metacurvature_seg_hpo --model metacurvature --n_way_eval 5 --max_shots_eval 5 --train_datasets FSS_Trn --val_id_datasets FSS_Val --val_od_datasets Vizwiz --T 5 --T_val 10 --T_test 10 --runs 1 --train_iters 5000 --num_samples 30 --eval_iters 100 --root_dir ${META_OMNIUM_DIR}"
"--experiment_name protonet_seg_hpo --model protonet --n_way_eval 5 --max_shots_eval 5 --train_datasets FSS_Trn --val_id_datasets FSS_Val --val_od_datasets Vizwiz --runs 1 --train_iters 5000 --num_samples 30 --eval_iters 100 --root_dir ${META_OMNIUM_DIR}"
"--experiment_name ddrr_seg_hpo --model ddrr --n_way_eval 5 --max_shots_eval 5 --train_datasets FSS_Trn --val_id_datasets FSS_Val --val_od_datasets Vizwiz --runs 1 --train_iters 5000 --num_samples 30 --eval_iters 100 --root_dir ${META_OMNIUM_DIR}"
"--experiment_name proto_finetuning_seg_hpo --model proto_finetuning --n_way_eval 5 --max_shots_eval 5 --train_datasets FSS_Trn --val_id_datasets FSS_Val --val_od_datasets Vizwiz --T 20 --T_val 20 --T_test 20 --runs 1 --train_iters 5000 --num_samples 30 --eval_iters 100 --root_dir ${META_OMNIUM_DIR}"
"--experiment_name finetuning_seg_hpo --model finetuning --n_way_eval 5 --max_shots_eval 5 --train_datasets FSS_Trn --val_id_datasets FSS_Val --val_od_datasets Vizwiz --T 20 --T_val 20 --T_test 20 --runs 1 --train_iters 5000 --num_samples 30 --eval_iters 100 --root_dir ${META_OMNIUM_DIR}"
"--experiment_name linear_readout_seg_hpo --model finetuning --n_way_eval 5 --max_shots_eval 5 --train_datasets FSS_Trn --val_id_datasets FSS_Val --val_od_datasets Vizwiz --T 20 --T_val 20 --T_test 20 --freeze --runs 1 --train_iters 5000 --num_samples 30 --eval_iters 100 --root_dir ${META_OMNIUM_DIR}"
"--experiment_name tfs_seg_hpo --model tfs --n_way_eval 5 --max_shots_eval 5 --train_datasets FSS_Trn --val_id_datasets FSS_Val --val_od_datasets Vizwiz --T 20 --T_val 20 --T_test 20 --runs 1 --train_iters 5000 --num_samples 30 --eval_iters 100 --root_dir ${META_OMNIUM_DIR}"
)

for ARG in ${ARGS[@]}
do
python metaomnium/trainers/cross_task_fsl_hpo.py ${ARG}
done

# single task keypoint/pose estimation
ARGS=(
"--experiment_name maml_pose_hpo --model maml --n_way_eval 5 --max_shots_eval 5 --train_datasets Animal_Pose_Trn --val_id_datasets Animal_Pose_Val --val_od_datasets Synthetic_Animal_Pose --T 5 --T_val 10 --T_test 10 --runs 1 --train_iters 5000 --num_samples 30 --eval_iters 100 --root_dir ${META_OMNIUM_DIR}"
"--experiment_name protomaml_pose_hpo --model protomaml --n_way_eval 5 --max_shots_eval 5 --train_datasets Animal_Pose_Trn --val_id_datasets Animal_Pose_Val --val_od_datasets Synthetic_Animal_Pose --T 5 --T_val 10 --T_test 10 --runs 1 --train_iters 5000 --num_samples 30 --eval_iters 100 --root_dir ${META_OMNIUM_DIR}"
"--experiment_name metacurvature_pose_hpo --model metacurvature --n_way_eval 5 --max_shots_eval 5 --train_datasets Animal_Pose_Trn --val_id_datasets Animal_Pose_Val --val_od_datasets Synthetic_Animal_Pose --T 5 --T_val 10 --T_test 10 --runs 1 --train_iters 5000 --num_samples 30 --eval_iters 100 --root_dir ${META_OMNIUM_DIR}"
"--experiment_name protonet_pose_hpo --model protonet --n_way_eval 5 --max_shots_eval 5 --train_datasets Animal_Pose_Trn --val_id_datasets Animal_Pose_Val --val_od_datasets Synthetic_Animal_Pose --runs 1 --train_iters 5000 --num_samples 30 --eval_iters 100 --root_dir ${META_OMNIUM_DIR}"
"--experiment_name ddrr_pose_hpo --model ddrr --n_way_eval 5 --max_shots_eval 5 --train_datasets Animal_Pose_Trn --val_id_datasets Animal_Pose_Val --val_od_datasets Synthetic_Animal_Pose --runs 1 --train_iters 5000 --num_samples 30 --eval_iters 100 --root_dir ${META_OMNIUM_DIR}"
"--experiment_name proto_finetuning_pose_hpo --model proto_finetuning --n_way_eval 5 --max_shots_eval 5 --train_datasets Animal_Pose_Trn --val_id_datasets Animal_Pose_Val --val_od_datasets Synthetic_Animal_Pose --T 20 --T_val 20 --T_test 20 --runs 1 --train_iters 5000 --num_samples 30 --eval_iters 100 --root_dir ${META_OMNIUM_DIR}"
"--experiment_name finetuning_pose_hpo --model finetuning --n_way_eval 5 --max_shots_eval 5 --train_datasets Animal_Pose_Trn --val_id_datasets Animal_Pose_Val --val_od_datasets Synthetic_Animal_Pose --T 20 --T_val 20 --T_test 20 --runs 1 --train_iters 5000 --num_samples 30 --eval_iters 100 --root_dir ${META_OMNIUM_DIR}"
"--experiment_name linear_readout_pose_hpo --model finetuning --n_way_eval 5 --max_shots_eval 5 --train_datasets Animal_Pose_Trn --val_id_datasets Animal_Pose_Val --val_od_datasets Synthetic_Animal_Pose --T 20 --T_val 20 --T_test 20 --freeze --runs 1 --train_iters 5000 --num_samples 30 --eval_iters 100 --root_dir ${META_OMNIUM_DIR}"
"--experiment_name tfs_pose_hpo --model tfs --n_way_eval 5 --max_shots_eval 5 --train_datasets Animal_Pose_Trn --val_id_datasets Animal_Pose_Val --val_od_datasets Synthetic_Animal_Pose --T 20 --T_val 20 --T_test 20 --runs 1 --train_iters 5000 --num_samples 30 --eval_iters 100 --root_dir ${META_OMNIUM_DIR}"
)

for ARG in ${ARGS[@]}
do
python metaomnium/trainers/cross_task_fsl_hpo.py ${ARG}
done

# multi-task training
ARGS=(
"--experiment_name maml_multi_hpo --model maml --n_way_eval 5 --max_shots_eval 5 --train_datasets FSS_Trn,BCT_Mini_Trn,BRD_Mini_Trn,CRS_Mini_Trn,Animal_Pose_Trn --val_id_datasets FSS_Val,BCT_Mini_Val,BRD_Mini_Val,CRS_Mini_Val,Animal_Pose_Val --val_od_datasets FLW_Mini,MD_MIX_Mini,PLK_Mini,Vizwiz,Synthetic_Animal_Pose --T 5 --T_val 10 --T_test 10 --runs 1 --train_iters 5000 --num_samples 30 --eval_iters 100 --root_dir ${META_OMNIUM_DIR}"
"--experiment_name protomaml_multi_hpo --model protomaml --n_way_eval 5 --max_shots_eval 5 --train_datasets FSS_Trn,BCT_Mini_Trn,BRD_Mini_Trn,CRS_Mini_Trn,Animal_Pose_Trn --val_id_datasets FSS_Val,BCT_Mini_Val,BRD_Mini_Val,CRS_Mini_Val,Animal_Pose_Val --val_od_datasets FLW_Mini,MD_MIX_Mini,PLK_Mini,Vizwiz,Synthetic_Animal_Pose --T 5 --T_val 10 --T_test 10 --runs 1 --train_iters 5000 --num_samples 30 --eval_iters 100 --root_dir ${META_OMNIUM_DIR}"
"--experiment_name metacurvature_multi_hpo --model metacurvature --n_way_eval 5 --max_shots_eval 5 --train_datasets FSS_Trn,BCT_Mini_Trn,BRD_Mini_Trn,CRS_Mini_Trn,Animal_Pose_Trn --val_id_datasets FSS_Val,BCT_Mini_Val,BRD_Mini_Val,CRS_Mini_Val,Animal_Pose_Val --val_od_datasets FLW_Mini,MD_MIX_Mini,PLK_Mini,Vizwiz,Synthetic_Animal_Pose --T 5 --T_val 10 --T_test 10 --runs 1 --train_iters 5000 --num_samples 30 --eval_iters 100 --root_dir ${META_OMNIUM_DIR}"
"--experiment_name protonet_multi_hpo --model protonet --n_way_eval 5 --max_shots_eval 5 --train_datasets FSS_Trn,BCT_Mini_Trn,BRD_Mini_Trn,CRS_Mini_Trn,Animal_Pose_Trn --val_id_datasets FSS_Val,BCT_Mini_Val,BRD_Mini_Val,CRS_Mini_Val,Animal_Pose_Val --val_od_datasets FLW_Mini,MD_MIX_Mini,PLK_Mini,Vizwiz,Synthetic_Animal_Pose --runs 1 --train_iters 5000 --num_samples 30 --eval_iters 100 --root_dir ${META_OMNIUM_DIR}"
"--experiment_name ddrr_multi_hpo --model ddrr --n_way_eval 5 --max_shots_eval 5 --train_datasets FSS_Trn,BCT_Mini_Trn,BRD_Mini_Trn,CRS_Mini_Trn,Animal_Pose_Trn --val_id_datasets FSS_Val,BCT_Mini_Val,BRD_Mini_Val,CRS_Mini_Val,Animal_Pose_Val --val_od_datasets FLW_Mini,MD_MIX_Mini,PLK_Mini,Vizwiz,Synthetic_Animal_Pose --runs 1 --train_iters 5000 --num_samples 30 --eval_iters 100 --root_dir ${META_OMNIUM_DIR}"
"--experiment_name proto_finetuning_multi_hpo --model proto_finetuning --n_way_eval 5 --max_shots_eval 5 --train_datasets FSS_Trn,BCT_Mini_Trn,BRD_Mini_Trn,CRS_Mini_Trn,Animal_Pose_Trn --val_id_datasets FSS_Val,BCT_Mini_Val,BRD_Mini_Val,CRS_Mini_Val,Animal_Pose_Val --val_od_datasets FLW_Mini,MD_MIX_Mini,PLK_Mini,Vizwiz,Synthetic_Animal_Pose --T 20 --T_val 20 --T_test 20 --runs 1 --train_iters 5000 --num_samples 30 --eval_iters 100 --root_dir ${META_OMNIUM_DIR}"
"--experiment_name finetuning_multi_hpo --model finetuning --n_way_eval 5 --max_shots_eval 5 --train_datasets FSS_Trn,BCT_Mini_Trn,BRD_Mini_Trn,CRS_Mini_Trn,Animal_Pose_Trn --val_id_datasets FSS_Val,BCT_Mini_Val,BRD_Mini_Val,CRS_Mini_Val,Animal_Pose_Val --val_od_datasets FLW_Mini,MD_MIX_Mini,PLK_Mini,Vizwiz,Synthetic_Animal_Pose --T 20 --T_val 20 --T_test 20 --runs 1 --train_iters 5000 --num_samples 30 --eval_iters 100 --root_dir ${META_OMNIUM_DIR}"
"--experiment_name linear_readout_multi_hpo --model finetuning --n_way_eval 5 --max_shots_eval 5 --train_datasets FSS_Trn,BCT_Mini_Trn,BRD_Mini_Trn,CRS_Mini_Trn,Animal_Pose_Trn --val_id_datasets FSS_Val,BCT_Mini_Val,BRD_Mini_Val,CRS_Mini_Val,Animal_Pose_Val --val_od_datasets FLW_Mini,MD_MIX_Mini,PLK_Mini,Vizwiz,Synthetic_Animal_Pose --T 20 --T_val 20 --T_test 20 --freeze --runs 1 --train_iters 5000 --num_samples 30 --eval_iters 100 --root_dir ${META_OMNIUM_DIR}"
"--experiment_name tfs_multi_hpo --model tfs --n_way_eval 5 --max_shots_eval 5 --train_datasets FSS_Trn,BCT_Mini_Trn,BRD_Mini_Trn,CRS_Mini_Trn,Animal_Pose_Trn --val_id_datasets FSS_Val,BCT_Mini_Val,BRD_Mini_Val,CRS_Mini_Val,Animal_Pose_Val --val_od_datasets FLW_Mini,MD_MIX_Mini,PLK_Mini,Vizwiz,Synthetic_Animal_Pose --T 20 --T_val 20 --T_test 20 --runs 1 --train_iters 5000 --num_samples 30 --eval_iters 100 --root_dir ${META_OMNIUM_DIR}"
)

for ARG in ${ARGS[@]}
do
python metaomnium/trainers/cross_task_fsl_hpo.py ${ARG}
done