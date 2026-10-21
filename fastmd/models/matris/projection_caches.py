"""MatRIS's frozen projection caches: the module fields of each cache group its projection builders (in
``functions``, ``interaction_block`` and ``feature_embed``) assign, registered with the engine's provenance seals
(``fastmd.engine.cache_provenance``) when this module is imported (``functions`` imports it)."""
from __future__ import annotations

from fastmd.engine.cache_provenance import register_projection_cache_groups

PROJECTION_CACHE_GROUPS = {
    "indexed_split": ("_indexed_cat_linear_split_cache_key",
                      "_indexed_cat_linear_split_weights", "_indexed_cat_linear_split_bias"),
    "attention_merged": ("_merged_attn_proj_key", "_merged_attn_proj_weight",
                         "_merged_attn_proj_splits"),
    "combined_swish": ("_combined_swish_cache_key", "_combined_swish_weight"),
}
register_projection_cache_groups(PROJECTION_CACHE_GROUPS)
