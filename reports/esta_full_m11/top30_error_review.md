# M11 High-Confidence Error Review

These are post-hoc diagnostic patterns, not proven causal explanations.
First-kill fields are outcomes used only for error analysis and never model inputs.

1. `online:1f4b1506-6f45-43ee-8b4c-82961eb4bd5b_2`: predicted CT at 0.977, actual T; equipment diff CT +18850, first kill CT; favored_side_major_equipment_upset / predicted_favorite_lost_after_first_kill.
2. `lan:2a99d649-c6bb-4f90-8f2c-664b7738a2d9_6`: predicted CT at 0.977, actual T; equipment diff CT +29150, first kill CT; favored_side_major_equipment_upset / predicted_favorite_lost_after_first_kill.
3. `online:f6dd9383-8d0b-46b6-b255-c3747d107355_10`: predicted CT at 0.976, actual T; equipment diff CT +26900, first kill T; favored_side_major_equipment_upset / predicted_favorite_lost_first_kill.
4. `online:478d378e-e7c1-4d64-a3f3-679ee18f27b5_17`: predicted CT at 0.975, actual T; equipment diff CT +18150, first kill CT; favored_side_major_equipment_upset / predicted_favorite_lost_after_first_kill.
5. `online:f6dd9383-8d0b-46b6-b255-c3747d107355_14`: predicted CT at 0.975, actual T; equipment diff CT +30700, first kill T; favored_side_major_equipment_upset / predicted_favorite_lost_first_kill.
6. `lan:44d5e289-b224-43ae-aaf2-3d79accc7909_7`: predicted T at 0.963, actual CT; equipment diff CT -19600, first kill CT; favored_side_major_equipment_upset / predicted_favorite_lost_first_kill.
7. `online:ba19b05b-3103-4cbd-9f57-41d222f71770_10`: predicted T at 0.962, actual CT; equipment diff CT -24200, first kill CT; favored_side_major_equipment_upset / predicted_favorite_lost_first_kill.
8. `online:6f87123d-f8ab-4b80-94fb-83afdb2632f9_7`: predicted T at 0.957, actual CT; equipment diff CT -24600, first kill T; favored_side_major_equipment_upset / predicted_favorite_lost_after_first_kill.
9. `online:f6dd9383-8d0b-46b6-b255-c3747d107355_23`: predicted T at 0.957, actual CT; equipment diff CT -21950, first kill CT; favored_side_major_equipment_upset / predicted_favorite_lost_first_kill.
10. `lan:88551554-2186-4d7e-95c2-f7f22e543d9f_12`: predicted T at 0.957, actual CT; equipment diff CT -24300, first kill CT; favored_side_major_equipment_upset / predicted_favorite_lost_first_kill.
11. `lan:44c78b55-2961-40a1-ae4f-abe2d8309ebd_3`: predicted T at 0.940, actual CT; equipment diff CT -20050, first kill CT; favored_side_major_equipment_upset / predicted_favorite_lost_first_kill.
12. `lan:44c78b55-2961-40a1-ae4f-abe2d8309ebd_6`: predicted CT at 0.940, actual T; equipment diff CT +31200, first kill T; favored_side_major_equipment_upset / predicted_favorite_lost_first_kill.
13. `online:ba19b05b-3103-4cbd-9f57-41d222f71770_28`: predicted T at 0.924, actual CT; equipment diff CT -19750, first kill T; favored_side_major_equipment_upset / predicted_favorite_lost_after_first_kill.
14. `online:948ddde3-3a34-47ff-a72b-b385457ea9a6_18`: predicted T at 0.921, actual CT; equipment diff CT -19050, first kill CT; favored_side_major_equipment_upset / predicted_favorite_lost_first_kill.
15. `lan:c707514f-6eda-4e43-ade5-2a3056cfb38f_19`: predicted CT at 0.918, actual T; equipment diff CT +24950, first kill T; favored_side_major_equipment_upset / predicted_favorite_lost_first_kill.
16. `online:0240b0da-b2d9-40fb-9332-f3134243d2b6_22`: predicted CT at 0.918, actual T; equipment diff CT +26950, first kill T; favored_side_major_equipment_upset / predicted_favorite_lost_first_kill.
17. `online:25d1ece6-8169-419d-a231-9ad4b920ace3_25`: predicted T at 0.915, actual CT; equipment diff CT -25150, first kill CT; favored_side_major_equipment_upset / predicted_favorite_lost_first_kill.
18. `online:8a024690-6636-4b94-8211-683c104f6d84_22`: predicted T at 0.905, actual CT; equipment diff CT -25100, first kill T; favored_side_major_equipment_upset / predicted_favorite_lost_after_first_kill.
19. `lan:5e987f3c-2a75-4c63-8f04-0fac59abcc92_13`: predicted CT at 0.893, actual T; equipment diff CT +30150, first kill T; favored_side_major_equipment_upset / predicted_favorite_lost_first_kill.
20. `lan:59610ddd-4588-4379-8b1e-b8a84a678ed9_9`: predicted CT at 0.893, actual T; equipment diff CT +25450, first kill T; favored_side_major_equipment_upset / predicted_favorite_lost_first_kill.
21. `online:34525ebe-5474-46a4-9eda-3b5dc358314d_26`: predicted CT at 0.883, actual T; equipment diff CT +26950, first kill CT; favored_side_major_equipment_upset / predicted_favorite_lost_after_first_kill.
22. `lan:f982160d-7649-4460-b25f-38fd94343529_4`: predicted CT at 0.883, actual T; equipment diff CT +24600, first kill T; favored_side_major_equipment_upset / predicted_favorite_lost_first_kill.
23. `lan:7bd3ccfd-d101-4972-8e54-230434c33ceb_12`: predicted CT at 0.879, actual T; equipment diff CT +30650, first kill T; favored_side_major_equipment_upset / predicted_favorite_lost_first_kill.
24. `lan:f912d2b7-3bec-47a2-9609-0ab73fb263e3_9`: predicted CT at 0.878, actual T; equipment diff CT +25900, first kill T; favored_side_major_equipment_upset / predicted_favorite_lost_first_kill.
25. `online:297643ce-df7a-470e-8c22-904f4595c3d6_14`: predicted CT at 0.877, actual T; equipment diff CT +26750, first kill T; favored_side_major_equipment_upset / predicted_favorite_lost_first_kill.
26. `lan:252b2257-340c-42c8-8210-5e62674bcd52_9`: predicted CT at 0.877, actual T; equipment diff CT +26700, first kill T; favored_side_major_equipment_upset / predicted_favorite_lost_first_kill.
27. `online:05964cbb-8d63-4751-9749-e72e67a4202e_19`: predicted T at 0.875, actual CT; equipment diff CT -18200, first kill CT; favored_side_major_equipment_upset / predicted_favorite_lost_first_kill.
28. `online:2e7b5b87-ad49-40d5-a783-99ee519d419b_8`: predicted CT at 0.873, actual T; equipment diff CT +24450, first kill T; favored_side_major_equipment_upset / predicted_favorite_lost_first_kill.
29. `online:1ed468fc-dd85-4f84-9247-92be9a11af5b_7`: predicted CT at 0.870, actual T; equipment diff CT +26050, first kill T; favored_side_major_equipment_upset / predicted_favorite_lost_first_kill.
30. `lan:9337dadd-c577-4043-a280-4c3b1f70c400_11`: predicted T at 0.869, actual CT; equipment diff CT -23500, first kill T; favored_side_major_equipment_upset / predicted_favorite_lost_after_first_kill.
