// Synthetic metadata for state-only tests; browser acceptance uses actual model responses.
module.exports = selection => ({
  model_version:'test-frozen-model', feature_version:'test-features', calibration_method:'uncalibrated', inference_ms:1,
  model_sha256:'a'.repeat(64), calibrator_sha256:'b'.repeat(64),
  identity:{series_id:'test-series',game_id:'test-game',round_id:`test-round-${selection.example_id}`},
  source:{split:'test',purchase_state:'purchase_end'},
  validation:{status:'passed',required_input_field_count:selection.stage==='pre_round'?27:31,encoded_model_feature_count:selection.stage==='pre_round'?43:82}
});
