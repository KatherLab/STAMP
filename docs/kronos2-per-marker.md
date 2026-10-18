# Independent KRONOS2 marker features and MarkerFusion

The `kronos2_per_marker` extractor encodes each configured channel separately with
KRONOS2, using the channel's marker name and upstream normalization. All channels
share the same SPImage tile grid, dtype scaling, and tile order. Normalization and
forward passes receive one channel at a time. The pretrained extractor is frozen;
MarkerFusion is trained subsequently using clinical labels.

Per-slide HDF5 output:

- `marker_embeddings`: `(tiles, markers, 768)`, in the `marker_names` attribute order.
- `coord_x`, `coord_y`: the shared tile coordinates, in pixels.
- `feats` / `patch_embeddings`: mean of the independent marker vectors (hard links).
- `marker_embedding_method`: `independent_single_channel_cls`.
- `patch_embedding_method`: `mean_of_marker_embeddings`.

The mean is not KRONOS2's joint-panel CLS representation. MarkerFusion training
must select `marker_embeddings`. There are no native per-marker spatial token
outputs. The existing `kronos2` extractor continues to produce joint-panel features.