#!/bin/bash

set -x

# All the .yaml for HCP
wandb sweep --entity st-team wandb_sweeps/hcp/st_hcp_uni_gender_1_fmri_none_nodeedgemeta_diffpool_128.yaml # 2qpbpr51
wandb sweep --entity st-team wandb_sweeps/hcp/st_hcp_uni_gender_1_fmri_none_nodeedgemeta_mean_128.yaml # 3f2sxu40
wandb sweep --entity st-team wandb_sweeps/hcp/st_hcp_uni_gender_1_fmri_none_nodemeta_diffpool_128.yaml # qwghpo7t
wandb sweep --entity st-team wandb_sweeps/hcp/st_hcp_uni_gender_1_fmri_none_nodemeta_mean_128.yaml # 1ojprf8m
wandb sweep --entity st-team wandb_sweeps/hcp/st_hcp_uni_gender_2_fmri_none_nodeedgemeta_diffpool_128.yaml # smjj6lqm
wandb sweep --entity st-team wandb_sweeps/hcp/st_hcp_uni_gender_2_fmri_none_nodeedgemeta_mean_128.yaml # j3uuspl8
wandb sweep --entity st-team wandb_sweeps/hcp/st_hcp_uni_gender_2_fmri_none_nodemeta_diffpool_128.yaml # s0w2ec3p
wandb sweep --entity st-team wandb_sweeps/hcp/st_hcp_uni_gender_2_fmri_none_nodemeta_mean_128.yaml # twrl54yz
wandb sweep --entity st-team wandb_sweeps/hcp/st_hcp_uni_gender_3_fmri_none_nodeedgemeta_diffpool_128.yaml # udy992aj
wandb sweep --entity st-team wandb_sweeps/hcp/st_hcp_uni_gender_3_fmri_none_nodeedgemeta_mean_128.yaml # ur7fwbuu
wandb sweep --entity st-team wandb_sweeps/hcp/st_hcp_uni_gender_3_fmri_none_nodemeta_diffpool_128.yaml # 7o4xqpmb
wandb sweep --entity st-team wandb_sweeps/hcp/st_hcp_uni_gender_3_fmri_none_nodemeta_mean_128.yaml # 1jt1br96
wandb sweep --entity st-team wandb_sweeps/hcp/st_hcp_uni_gender_4_fmri_none_nodeedgemeta_diffpool_128.yaml # oeez4zhy
wandb sweep --entity st-team wandb_sweeps/hcp/st_hcp_uni_gender_4_fmri_none_nodeedgemeta_mean_128.yaml # p7j4xatk
wandb sweep --entity st-team wandb_sweeps/hcp/st_hcp_uni_gender_4_fmri_none_nodemeta_diffpool_128.yaml # qd4atco8
wandb sweep --entity st-team wandb_sweeps/hcp/st_hcp_uni_gender_4_fmri_none_nodemeta_mean_128.yaml # x8i95zux
wandb sweep --entity st-team wandb_sweeps/hcp/st_hcp_uni_gender_5_fmri_none_nodeedgemeta_diffpool_128.yaml # g7mz4mss
wandb sweep --entity st-team wandb_sweeps/hcp/st_hcp_uni_gender_5_fmri_none_nodeedgemeta_mean_128.yaml # fgsaz0o5
wandb sweep --entity st-team wandb_sweeps/hcp/st_hcp_uni_gender_5_fmri_none_nodemeta_diffpool_128.yaml # md8euckv
wandb sweep --entity st-team wandb_sweeps/hcp/st_hcp_uni_gender_5_fmri_none_nodemeta_mean_128.yaml # h0f8x15u

wandb sweep --entity st-team wandb_sweeps/hcp/st_hcp_uni_gender_1_fmri_none_diffpool_F_128.yaml # 3f5azgyh
wandb sweep --entity st-team wandb_sweeps/hcp/st_hcp_uni_gender_2_fmri_none_diffpool_F_128.yaml # 7ei3y5jw
wandb sweep --entity st-team wandb_sweeps/hcp/st_hcp_uni_gender_3_fmri_none_diffpool_F_128.yaml # u5sp6d1f
wandb sweep --entity st-team wandb_sweeps/hcp/st_hcp_uni_gender_4_fmri_none_diffpool_F_128.yaml # i8k7cp0b
wandb sweep --entity st-team wandb_sweeps/hcp/st_hcp_uni_gender_5_fmri_none_diffpool_F_128.yaml # 4pdzce1b

wandb sweep --entity st-team wandb_sweeps/hcp/st_hcp_uni_gender_1_fmri_none_no_gnn_F_128.yaml # vr1aww61
wandb sweep --entity st-team wandb_sweeps/hcp/st_hcp_uni_gender_2_fmri_none_no_gnn_F_128.yaml # xgc6zjq6
wandb sweep --entity st-team wandb_sweeps/hcp/st_hcp_uni_gender_3_fmri_none_no_gnn_F_128.yaml # iedni22n
wandb sweep --entity st-team wandb_sweeps/hcp/st_hcp_uni_gender_4_fmri_none_no_gnn_F_128.yaml # ndof5s8n
wandb sweep --entity st-team wandb_sweeps/hcp/st_hcp_uni_gender_5_fmri_none_no_gnn_F_128.yaml # elueatsy

wandb sweep --entity st-team wandb_sweeps/hcp/st_hcp_THRE100_uni_gender_1_fmri_none_nodeedgemeta_diffpool_128.yaml # 8yxahw3r
wandb sweep --entity st-team wandb_sweeps/hcp/st_hcp_THRE100_uni_gender_2_fmri_none_nodeedgemeta_diffpool_128.yaml # au66op5w
wandb sweep --entity st-team wandb_sweeps/hcp/st_hcp_THRE100_uni_gender_3_fmri_none_nodeedgemeta_diffpool_128.yaml # m9xeht0l
wandb sweep --entity st-team wandb_sweeps/hcp/st_hcp_THRE100_uni_gender_4_fmri_none_nodeedgemeta_diffpool_128.yaml # w1xll1i0
wandb sweep --entity st-team wandb_sweeps/hcp/st_hcp_THRE100_uni_gender_5_fmri_none_nodeedgemeta_diffpool_128.yaml # lyj2ou9o

wandb sweep --entity st-team wandb_sweeps/hcp/st_hcp_THRE100_uni_gender_1_fmri_none_nodeedgemeta_mean_128.yaml # 27612e5y
wandb sweep --entity st-team wandb_sweeps/hcp/st_hcp_THRE100_uni_gender_2_fmri_none_nodeedgemeta_mean_128.yaml # 4crn5mr1
wandb sweep --entity st-team wandb_sweeps/hcp/st_hcp_THRE100_uni_gender_3_fmri_none_nodeedgemeta_mean_128.yaml # r7e6f0q0
wandb sweep --entity st-team wandb_sweeps/hcp/st_hcp_THRE100_uni_gender_4_fmri_none_nodeedgemeta_mean_128.yaml # ge81rar8
wandb sweep --entity st-team wandb_sweeps/hcp/st_hcp_THRE100_uni_gender_5_fmri_none_nodeedgemeta_mean_128.yaml # pbocnxs2

wandb sweep --entity st-team wandb_sweeps/hcp/st_hcp_THRE100_uni_gender_1_fmri_none_nodemeta_diffpool_128.yaml # re6s6t3h
wandb sweep --entity st-team wandb_sweeps/hcp/st_hcp_THRE100_uni_gender_2_fmri_none_nodemeta_diffpool_128.yaml # 96dj2sfl
wandb sweep --entity st-team wandb_sweeps/hcp/st_hcp_THRE100_uni_gender_3_fmri_none_nodemeta_diffpool_128.yaml # mb6juvrt
wandb sweep --entity st-team wandb_sweeps/hcp/st_hcp_THRE100_uni_gender_4_fmri_none_nodemeta_diffpool_128.yaml # rjlbyiep
wandb sweep --entity st-team wandb_sweeps/hcp/st_hcp_THRE100_uni_gender_5_fmri_none_nodemeta_diffpool_128.yaml # tlxa46vv

wandb sweep --entity st-team wandb_sweeps/hcp/st_hcp_THRE100_uni_gender_1_fmri_none_nodemeta_mean_128.yaml # xl6z8exf
wandb sweep --entity st-team wandb_sweeps/hcp/st_hcp_THRE100_uni_gender_2_fmri_none_nodemeta_mean_128.yaml # 30sgxphi
wandb sweep --entity st-team wandb_sweeps/hcp/st_hcp_THRE100_uni_gender_3_fmri_none_nodemeta_mean_128.yaml # xm3ewgec
wandb sweep --entity st-team wandb_sweeps/hcp/st_hcp_THRE100_uni_gender_4_fmri_none_nodemeta_mean_128.yaml # czm6u5td
wandb sweep --entity st-team wandb_sweeps/hcp/st_hcp_THRE100_uni_gender_5_fmri_none_nodemeta_mean_128.yaml # ui3xti70