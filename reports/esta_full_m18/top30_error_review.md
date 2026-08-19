# M18 首杀后高置信度错误复核

定义：预测错误且预测方概率不低于 0.80。以下模式是事后描述，不是因果结论。

1. `63e0d98d-fd95-4698-9d12-808508005ee2 / online:1f4b1506-6f45-43ee-8b4c-82961eb4bd5b / online:1f4b1506-6f45-43ee-8b4c-82961eb4bd5b_2`：预测 CT (0.989)，实际 T；首杀 CT，时间 56.17s，武器 MP9，装备差 CT +18850；`first_kill_and_equipment_agree`。
2. `90da2c53-5a02-4f16-8abe-f2235da5ffbd / online:478d378e-e7c1-4d64-a3f3-679ee18f27b5 / online:478d378e-e7c1-4d64-a3f3-679ee18f27b5_17`：预测 CT (0.985)，实际 T；首杀 CT，时间 28.70s，武器 MP9，装备差 CT +18150；`first_kill_and_equipment_agree`。
3. `47cef15a-cf52-476e-afba-abe0c33843f9 / lan:2a99d649-c6bb-4f90-8f2c-664b7738a2d9 / lan:2a99d649-c6bb-4f90-8f2c-664b7738a2d9_6`：预测 CT (0.985)，实际 T；首杀 CT，时间 27.33s，武器 AK-47，装备差 CT +29150；`first_kill_and_equipment_agree`。
4. `0f222f22-1d61-48c7-92a7-0578174959ba / online:6f87123d-f8ab-4b80-94fb-83afdb2632f9 / online:6f87123d-f8ab-4b80-94fb-83afdb2632f9_7`：预测 T (0.978)，实际 CT；首杀 T，时间 42.75s，武器 AK-47，装备差 CT -24600；`first_kill_and_equipment_agree`。
5. `c7f0e145-6303-40e6-9b79-ab0f825db57d / online:ba19b05b-3103-4cbd-9f57-41d222f71770 / online:ba19b05b-3103-4cbd-9f57-41d222f71770_28`：预测 T (0.972)，实际 CT；首杀 T，时间 19.37s，武器 AK-47，装备差 CT -19750；`first_kill_and_equipment_agree`。
6. `747526c1-f423-435f-8941-8064849b1c57 / online:34525ebe-5474-46a4-9eda-3b5dc358314d / online:34525ebe-5474-46a4-9eda-3b5dc358314d_26`：预测 CT (0.947)，实际 T；首杀 CT，时间 52.48s，武器 AWP，装备差 CT +26950；`first_kill_and_equipment_agree`。
7. `a997b20a-a1e1-4c8f-9624-8aa733a72ded / online:8a024690-6636-4b94-8211-683c104f6d84 / online:8a024690-6636-4b94-8211-683c104f6d84_22`：预测 T (0.946)，实际 CT；首杀 T，时间 13.67s，武器 AWP，装备差 CT -25100；`first_kill_and_equipment_agree`。
8. `8c516ddf-4403-4eaa-a17d-44a9feae80f5 / lan:9337dadd-c577-4043-a280-4c3b1f70c400 / lan:9337dadd-c577-4043-a280-4c3b1f70c400_11`：预测 T (0.931)，实际 CT；首杀 T，时间 22.40s，武器 AK-47，装备差 CT -23500；`first_kill_and_equipment_agree`。
9. `689b4e82-473b-4e35-a0c8-e23edc2dbb23 / lan:325cac78-4ca2-4a8d-bb7f-441320ce69d4 / lan:325cac78-4ca2-4a8d-bb7f-441320ce69d4_20`：预测 T (0.930)，实际 CT；首杀 T，时间 27.64s，武器 AK-47，装备差 CT -17400；`first_kill_and_equipment_agree`。
10. `bb7fb8a7-1a8b-4990-bb6b-b42109cb3259 / online:d02539dd-8ef5-4f24-a1ca-c1c39d93e0e9 / online:d02539dd-8ef5-4f24-a1ca-c1c39d93e0e9_19`：预测 CT (0.928)，实际 T；首杀 CT，时间 53.02s，武器 AK-47，装备差 CT +26450；`first_kill_and_equipment_agree`。
11. `15a7cd5c-2104-42cd-bac4-eae9d1083d19 / online:a314368f-f7f9-42b0-b497-fc60716dcaa7 / online:a314368f-f7f9-42b0-b497-fc60716dcaa7_14`：预测 T (0.925)，实际 CT；首杀 T，时间 23.07s，武器 AK-47，装备差 CT -13800；`first_kill_and_equipment_agree`。
12. `c061918a-87ee-477d-afcc-12dab5a769ae / online:1eedeeca-4400-41a5-a861-22e88435b473 / online:1eedeeca-4400-41a5-a861-22e88435b473_6`：预测 T (0.916)，实际 CT；首杀 T，时间 15.43s，武器 AK-47，装备差 CT -21700；`first_kill_and_equipment_agree`。
13. `8fadbb13-3728-4211-b0eb-cd4075a60218 / lan:44d5e289-b224-43ae-aaf2-3d79accc7909 / lan:44d5e289-b224-43ae-aaf2-3d79accc7909_7`：预测 T (0.913)，实际 CT；首杀 CT，时间 23.07s，武器 Knife，装备差 CT -19600；`equipment_only`。
14. `bc33710d-1499-4a28-b981-ad5b0eb061a1 / lan:05a57319-2185-4a18-9ad8-89b078bc48cb / lan:05a57319-2185-4a18-9ad8-89b078bc48cb_14`：预测 T (0.913)，实际 CT；首杀 T，时间 51.07s，武器 AK-47，装备差 CT -14250；`first_kill_and_equipment_agree`。
15. `c7f0e145-6303-40e6-9b79-ab0f825db57d / online:ba19b05b-3103-4cbd-9f57-41d222f71770 / online:ba19b05b-3103-4cbd-9f57-41d222f71770_10`：预测 T (0.912)，实际 CT；首杀 CT，时间 39.23s，武器 P250，装备差 CT -24200；`equipment_only`。
16. `5a4c75ca-ed01-4d01-8d2c-97409a9f5113 / lan:88551554-2186-4d7e-95c2-f7f22e543d9f / lan:88551554-2186-4d7e-95c2-f7f22e543d9f_12`：预测 T (0.912)，实际 CT；首杀 CT，时间 17.61s，武器 Desert Eagle，装备差 CT -24300；`equipment_only`。
17. `58e04be8-d9d5-4031-adda-bec6a4af79b9 / online:3f804500-94e5-46bb-9e90-d93b04df2acb / online:3f804500-94e5-46bb-9e90-d93b04df2acb_18`：预测 CT (0.908)，实际 T；首杀 CT，时间 13.10s，武器 M4A1，装备差 CT +17500；`first_kill_and_equipment_agree`。
18. `891751f3-228c-453e-8a36-b67ee9340cbd / online:b53069af-8ee0-4757-8500-448c6f4082d8 / online:b53069af-8ee0-4757-8500-448c6f4082d8_12`：预测 CT (0.905)，实际 T；首杀 CT，时间 27.83s，武器 AWP，装备差 CT +20350；`first_kill_and_equipment_agree`。
19. `bdada140-b48c-4be0-93c3-ce2cd7973eed / online:e941205c-0716-43ea-9836-3ad863fc2193 / online:e941205c-0716-43ea-9836-3ad863fc2193_3`：预测 T (0.905)，实际 CT；首杀 T，时间 35.39s，武器 AK-47，装备差 CT -14800；`first_kill_and_equipment_agree`。
20. `7661436d-fb0d-44ba-859a-02946733736c / lan:54d9b5f2-f079-4ce9-8a78-06beae802c29 / lan:54d9b5f2-f079-4ce9-8a78-06beae802c29_3`：预测 CT (0.903)，实际 T；首杀 CT，时间 17.50s，武器 HE Grenade，装备差 CT +18850；`first_kill_and_equipment_agree`。
21. `aca4acfc-efb2-4552-a503-2284be8c2a5f / online:d35b998d-716e-477b-a66c-16663c8a2a2b / online:d35b998d-716e-477b-a66c-16663c8a2a2b_22`：预测 T (0.900)，实际 CT；首杀 T，时间 69.42s，武器 AK-47，装备差 CT -15950；`first_kill_and_equipment_agree`。
22. `c7f0e145-6303-40e6-9b79-ab0f825db57d / online:f6dd9383-8d0b-46b6-b255-c3747d107355 / online:f6dd9383-8d0b-46b6-b255-c3747d107355_10`：预测 CT (0.897)，实际 T；首杀 T，时间 52.99s，武器 Desert Eagle，装备差 CT +26900；`equipment_only`。
23. `c061918a-87ee-477d-afcc-12dab5a769ae / online:1eedeeca-4400-41a5-a861-22e88435b473 / online:1eedeeca-4400-41a5-a861-22e88435b473_11`：预测 CT (0.895)，实际 T；首杀 CT，时间 14.80s，武器 M4A4，装备差 CT +28150；`first_kill_and_equipment_agree`。
24. `79e16b64-6722-4b8d-b38a-95463d6315b9 / lan:e8eff408-f08e-4798-9d7f-807375662fce / lan:e8eff408-f08e-4798-9d7f-807375662fce_14`：预测 T (0.895)，实际 CT；首杀 T，时间 16.86s，武器 AK-47，装备差 CT -18700；`first_kill_and_equipment_agree`。
25. `f107e704-85f7-44aa-9a35-ad7ac601c4d2 / lan:1aa2b4d0-da56-4760-8027-3c485531133f / lan:1aa2b4d0-da56-4760-8027-3c485531133f_25`：预测 CT (0.895)，实际 T；首杀 CT，时间 17.69s，武器 M4A1，装备差 CT +22500；`first_kill_and_equipment_agree`。
26. `28c6914e-e018-4d50-82f0-84082ae40d60 / lan:1cda2845-354c-4007-80f6-9c487a18d65d / lan:1cda2845-354c-4007-80f6-9c487a18d65d_13`：预测 T (0.891)，实际 CT；首杀 T，时间 19.46s，武器 AK-47，装备差 CT -18650；`first_kill_and_equipment_agree`。
27. `c7f0e145-6303-40e6-9b79-ab0f825db57d / online:f6dd9383-8d0b-46b6-b255-c3747d107355 / online:f6dd9383-8d0b-46b6-b255-c3747d107355_23`：预测 T (0.889)，实际 CT；首杀 CT，时间 23.86s，武器 CZ75 Auto，装备差 CT -21950；`equipment_only`。
28. `15a7cd5c-2104-42cd-bac4-eae9d1083d19 / online:a314368f-f7f9-42b0-b497-fc60716dcaa7 / online:a314368f-f7f9-42b0-b497-fc60716dcaa7_21`：预测 T (0.889)，实际 CT；首杀 T，时间 13.90s，武器 AK-47，装备差 CT -19200；`first_kill_and_equipment_agree`。
29. `8ddb1672-ae23-4ad0-b41b-25add199cd3b / lan:23976615-8f5e-46df-8d90-b37f3c3a717f / lan:23976615-8f5e-46df-8d90-b37f3c3a717f_6`：预测 T (0.888)，实际 CT；首杀 T，时间 12.84s，武器 AWP，装备差 CT -17800；`first_kill_and_equipment_agree`。
30. `48fb8ae5-5342-44cc-bfc5-8957068d565e / online:996ea94c-d525-4d3e-b54d-5b68be2b4bcf / online:996ea94c-d525-4d3e-b54d-5b68be2b4bcf_3`：预测 CT (0.888)，实际 T；首杀 CT，时间 19.94s，武器 MP9，装备差 CT +16400；`first_kill_and_equipment_agree`。
