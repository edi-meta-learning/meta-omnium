#!/bin/sh

# you can run this script to copy the data to a server and unzip it there

src_path=/local_disk/meta-omnium
dest_path=/server_disk/meta-omnium/data
mkdir -p ${dest_path}

scp ${src_path}/Classification/BCT_Mini_Trn.zip ${dest_path}/
scp ${src_path}/Classification/BRD_Mini_Trn.zip ${dest_path}/
scp ${src_path}/Classification/CRS_Mini_Trn.zip ${dest_path}/
scp ${src_path}/Classification/BCT_Mini_Val.zip ${dest_path}/
scp ${src_path}/Classification/BRD_Mini_Val.zip ${dest_path}/
scp ${src_path}/Classification/CRS_Mini_Val.zip ${dest_path}/
scp ${src_path}/Classification/FLW_Mini.zip ${dest_path}/
scp ${src_path}/Classification/MD_MIX_Mini.zip ${dest_path}/
scp ${src_path}/Classification/PLK_Mini.zip ${dest_path}/
scp ${src_path}/Classification/BCT_Mini_Test.zip ${dest_path}/
scp ${src_path}/Classification/BRD_Mini_Test.zip ${dest_path}/
scp ${src_path}/Classification/CRS_Mini_Test.zip ${dest_path}/
scp ${src_path}/Classification/PLT_VIL_Mini.zip ${dest_path}/
scp ${src_path}/Classification/RESISC_Mini.zip ${dest_path}/
scp ${src_path}/Classification/SPT_Mini.zip ${dest_path}/
scp ${src_path}/Classification/TEX_Mini.zip ${dest_path}/

scp ${src_path}/Segmentation/FSS_Trn.zip ${dest_path}/
scp ${src_path}/Segmentation/FSS_Val.zip ${dest_path}/
scp ${src_path}/Segmentation/Vizwiz.zip ${dest_path}/
scp ${src_path}/Segmentation/FSS_Test.zip ${dest_path}/
scp ${src_path}/Segmentation/PASCAL.zip ${dest_path}/
scp ${src_path}/Segmentation/PH2.zip ${dest_path}/

scp ${src_path}/Keypoints/Animal_Pose_Trn.zip ${dest_path}/
scp ${src_path}/Keypoints/Animal_Pose_Val.zip ${dest_path}/
scp ${src_path}/Keypoints/Synthetic_Animal_Pose.zip ${dest_path}/
scp ${src_path}/Keypoints/Animal_Pose_Test.zip ${dest_path}/
scp ${src_path}/Keypoints/MPII.zip ${dest_path}/

scp ${src_path}/Regression/Distractor_Test.zip ${dest_path}/
scp ${src_path}/Regression/ShapeNet2D_Test.zip ${dest_path}/
scp ${src_path}/Regression/ShapeNet1D_Test.zip ${dest_path}/
scp ${src_path}/Regression/Pascal1D_Test.zip ${dest_path}/

unzip ${dest_path}/BCT_Mini_Trn.zip -d ${dest_path}
unzip ${dest_path}/BRD_Mini_Trn.zip -d ${dest_path}
unzip ${dest_path}/CRS_Mini_Trn.zip -d ${dest_path}
unzip ${dest_path}/BCT_Mini_Val.zip -d ${dest_path}
unzip ${dest_path}/BRD_Mini_Val.zip -d ${dest_path}
unzip ${dest_path}/CRS_Mini_Val.zip -d ${dest_path}
unzip ${dest_path}/FLW_Mini.zip -d ${dest_path}
unzip ${dest_path}/MD_MIX_Mini.zip -d ${dest_path}
unzip ${dest_path}/PLK_Mini.zip -d ${dest_path}
unzip ${dest_path}/BCT_Mini_Test.zip -d ${dest_path}
unzip ${dest_path}/BRD_Mini_Test.zip -d ${dest_path}
unzip ${dest_path}/CRS_Mini_Test.zip -d ${dest_path}
unzip ${dest_path}/PLT_VIL_Mini.zip -d ${dest_path}
unzip ${dest_path}/RESISC_Mini.zip -d ${dest_path}
unzip ${dest_path}/SPT_Mini.zip -d ${dest_path}
unzip ${dest_path}/TEX_Mini.zip -d ${dest_path}

unzip ${dest_path}/FSS_Trn.zip -d ${dest_path}
unzip ${dest_path}/FSS_Val.zip -d ${dest_path}
unzip ${dest_path}/Vizwiz.zip -d ${dest_path}
unzip ${dest_path}/FSS_Test.zip -d ${dest_path}
unzip ${dest_path}/PASCAL.zip -d ${dest_path}
unzip ${dest_path}/PH2.zip -d ${dest_path}

unzip ${dest_path}/Animal_Pose_Trn.zip -d ${dest_path}
unzip ${dest_path}/Animal_Pose_Val.zip -d ${dest_path}
unzip ${dest_path}/Synthetic_Animal_Pose.zip -d ${dest_path}
unzip ${dest_path}/Animal_Pose_Test.zip -d ${dest_path}
unzip ${dest_path}/MPII.zip -d ${dest_path}

unzip ${dest_path}/Distractor_Test.zip -d ${dest_path}
unzip ${dest_path}/ShapeNet2D_Test.zip -d ${dest_path}
unzip ${dest_path}/ShapeNet1D_Test.zip -d ${dest_path}
unzip ${dest_path}/Pascal1D_Test.zip -d ${dest_path}

# Make sure the extracted files were accessed recently
find ${dest_path}/ -type f -exec touch {} +

ls ${dest_path}/
