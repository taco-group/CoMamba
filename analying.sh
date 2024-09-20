



# model='/home/jinlongli/1.Detection_Set/Comamba/5.Comamba2024/test_Comamba_hw_layer2_2024_04_17_23_08_52'


# model='/home/jinlongli/1.Detection_Set/Comamba/5.Comamba2024/test_Comamba_hw_layer1_2024_04_17_23_06_04'


# model='/home/jinlongli/1.Detection_Set/Comamba/5.Comamba2024/test_Comamba_hw_layer3_2024_04_17_23_10_37'



# model='/home/jinlongli/1.Detection_Set/Comamba/5.Comamba2024/test_Comamba_bn_layer1_2024_04_17_22_56_07'


# model='/home/jinlongli/1.Detection_Set/Comamba/5.Comamba2024/test_Comamba_bn_layer2_2024_04_17_22_54_14'
# 
# model='/home/jinlongli/1.Detection_Set/Comamba/5.Comamba2024/test_Comamba_bn_layer3_2024_04_17_22_51_02'
# model='/home/jinlongli/1.Detection_Set/Comamba/5.Comamba2024/0.propsoed_comamba_multi-scale_V1_2024_04_30_17_22_16'
# model='/home/jinlongli/1.Detection_Set/Comamba/5.Comamba2024/0.proposed_comamba_multi-scale_4+4_mean_2024_05_01_13_01_48'







###---------------->
# model='/home/jinlongli/personal/personal_jinlongli/2.model_saved/1.da_v2vreal_model_trained2023/DA_CPVIT_model23/no-DA/CoBEVT'

# model='/home/jinlongli/1.Detection_Set/Comamba/5.Comamba2024/0.proposed_comamba_multi-scale_4+4_mean_max_2024_05_01_13_04_54'


# model='/home/jinlongli/1.Detection_Set/Comamba/5.Comamba2024/2.comamba_normal_multi-scale_only_2024_05_03_23_56_39'

# model='/home/jinlongli/1.Detection_Set/Comamba/5.Comamba2024/2.5x_comamba_normal_multi-scale_vssfusion_2024_05_04_01_23_36'

# model='/home/jinlongli/2.model_saved/1.da_v2vreal_model_trained2023/DA_CPVIT_model23/no-DA/v2x-vit'

# model='/home/jinlongli/2.model_saved/1.da_v2vreal_model_trained2023/DA_CPVIT_model23/no-DA/CoBEVT'
# model='/home/jinlongli/2.model_saved/1.da_v2vreal_model_trained2023/DA_CPVIT_model23/no-DA/attfuse'




#------------------------>

# model='/home/jinlongli/1.Detection_Set/Comamba/5.Comamba2024/3.opv2v_mamba_8448_3_2024_05_05_00_38_14'



# model='/home/jinlongli/1.Detection_Set/Comamba/5.Comamba2024/opv2v_2024_05_04_22_54_34'

# model='/home/jinlongli/2.model_saved/1.da_v2vreal_model_trained2023/DA_CPVIT_model23/no-DA/CoBEVT'


# model='/home/jinlongli/2.model_saved/1.da_v2vreal_model_trained2023/DA_CPVIT_model23/no-DA/v2x-vit'


# model='/home/jinlongli/1.Detection_Set/Comamba/5.Comamba2024/4.opv2v_mamba_N_1_max_mean_2024_05_06_22_08_37'




# model='/home/jinlongli/1.Detection_Set/Comamba/5.Comamba2024/4.opv2v_mamba_N_1_max_mean_2024_05_07_12_54_37'

# model='/home/jinlongli/1.Detection_Set/Comamba/5.Comamba2024/4.opv2v_mamba_N_HW_CrossScan_Ab_1direction_2024_05_07_08_47_27'



#---------------------->

# model='/home/jinlongli/1.Detection_Set/Comamba/5.Comamba2024/5.testing_opv2v_mamba_N_1_max_mean_continous_rf2_2024_05_14_14_14_45'
# model='/home/jinlongli/1.Detection_Set/Comamba/5.Comamba2024/5.testing_opv2v_mamba_N_1_max_mean_continous_rf3_8_2024_05_14_14_47_00'


hypes_yaml='/home/jinlongli/1.Detection_Set/Comamba/V2V/opencood/hypes_yaml/point_pillar_opv2v_comamba.yaml'
# hypes_yaml='/home/jinlongli/1.Detection_Set/Comamba/V2V/opencood/hypes_yaml/point_pillar_cobevt.yaml'
# hypes_yaml='/home/jinlongli/1.Detection_Set/Comamba/V2V/opencood/hypes_yaml/point_pillar_v2xvit.yaml'


# hypes_yaml='/home/jinlongli/1.Detection_Set/Comamba/V2V/opencood/hypes_yaml/point_pillar_opv2v.yaml'
# hypes_yaml='/home/jinlongli/1.Detection_Set/Comamba/V2V/opencood/hypes_yaml/point_pillar_opv2v_vit.yaml'


#conda activate comamba

# run python script
CUDA_VISIBLE_DEVICES=0 python3 opencood/tools/performance_analyze.py \
    --fusion_method intermediate \
    --hypes_yaml $hypes_yaml
    # --model_dir $model 