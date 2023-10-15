import json
args = parse_args()
ap.print_arguments(args)
args.model = "meta_gru4rec_ood_uncertainty_separate"
model_uncertainty = model.get_model_meta(args.model_uncertainty)  # type: model.ModelMeta
model_uncertainty_conf, raw_model_uncertainty_conf = ap.parse_arch_config_from_args(model_uncertainty, args)
model_uncertainty_obj = model_meta.model_builder(model_conf=model_uncertainty_conf)
model_conf = {
    "id_dimension": 32,
    "id_vocab": 8000,
    "classifier": [128, 64],
    "mlp_layers": 2
}
model = GRU4Rec(model_conf)