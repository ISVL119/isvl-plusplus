# 1. dataset processing
python split_crop.py
python copy_images.py

# 2. anomaly synthesis
cd sam3-260512
python 1_foreground_extraction.py
python 2_generate_synthetic_anomaly.py
mv ./log/synthesized/synthesized_mvtecad2_1024rgbl ../datasets/
cd ../

# 3. model training
python isvl_train_and_test++.py  --use_synth_anomalies --item_list can  --phase train --synth_anomaly_root ./datasets/synthesized_mvtecad2_1024rgbl/can
python isvl_train_and_test++.py  --use_synth_anomalies --item_list fabric --phase train --synth_anomaly_root ./datasets/synthesized_mvtecad2_1024rgbl/fabric
python isvl_train_and_test++.py  --use_synth_anomalies --item_list fruit_jelly  --phase train --synth_anomaly_root ./datasets/synthesized_mvtecad2_1024rgbl/fruit_jelly
python isvl_train_and_test++.py  --use_synth_anomalies --item_list sheet_metal --phase train --synth_anomaly_root ./datasets/synthesized_mvtecad2_1024rgbl/sheet_metal
python isvl_train_and_test++.py  --use_synth_anomalies --item_list vial --phase train --synth_anomaly_root ./datasets/synthesized_mvtecad2_1024rgbl/vial
python isvl_train_and_test++.py  --use_synth_anomalies --item_list walnuts --phase train --synth_anomaly_root ./datasets/synthesized_mvtecad2_1024rgbl/walnuts
python isvl_train_and_test++.py  --use_synth_anomalies --item_list wallplugs --phase train --synth_anomaly_root ./datasets/synthesized_mvtecad2_1024rgbl/wallplugs
python isvl_train_and_test++.py  --use_synth_anomalies --item_list rice --phase train --synth_anomaly_root ./datasets/synthesized_mvtecad2_1024rgbl/rice

# 4. model inferrence
python isvl_train_and_test++.py  --item_list can  --phase test 
python isvl_train_and_test++.py  --item_list fabric --phase test 
python isvl_train_and_test++.py  --item_list fruit_jelly  --phase test 
python isvl_train_and_test++.py  --item_list sheet_metal --phase test 
python isvl_train_and_test++.py  --item_list vial --phase test 
python isvl_train_and_test++.py  --item_list walnuts --phase test 
python isvl_train_and_test++.py  --item_list wallplugs --phase test 
python isvl_train_and_test++.py  --item_list rice --phase test 

# 5. select threshold
python isvl_select_threshold.py --item_list rice --phase true_val  --synthetic_k_enable --illumination_calibration_enable
python isvl_select_threshold.py --item_list fabric --phase true_val  --synthetic_k_enable --illumination_calibration_enable
python isvl_select_threshold.py --item_list can --phase true_val --synthetic_k_enable --illumination_calibration_enable
python isvl_select_threshold.py --item_list fruit_jelly --phase true_val  --synthetic_k_enable --illumination_calibration_enable
python isvl_select_threshold.py --item_list sheet_metal --phase true_val --synthetic_k_enable --illumination_calibration_enable
python isvl_select_threshold.py --item_list vial --phase true_val  --synthetic_k_enable --illumination_calibration_enable
python isvl_select_threshold.py --item_list walnuts --phase true_val --synthetic_k_enable --illumination_calibration_enable
python isvl_select_threshold.py --item_list wallplugs --phase true_val --synthetic_k_enable --illumination_calibration_enable

# 6. binary and data post processing
python threshold_map.py
python SAMHQ-Finer.py

# 7. submit
python convert_tiff_to_float16.py
python check_and_prepare_data_for_upload.py ./results/