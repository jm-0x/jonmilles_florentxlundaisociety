export type TokenPrediction = { token: string; prob: number };

export type ZeroHeadIntervention = {
  type: "zero_head";
  layer: number;
  head: number;
};

export type ZeroMLPIntervention = {
  type: "zero_mlp";
  layer: number;
};

export type PatchHeadIntervention = {
  type: "patch_head";
  layer: number;
  head: number;
  source_prompt: string;
};

export type Intervention =
  | ZeroHeadIntervention
  | ZeroMLPIntervention
  | PatchHeadIntervention;

export type Trace = {
  prompt: string;
  tokens: string[];
  token_ids: number[];
  n_layers: number;
  n_heads: number;
  d_model: number;
  seq_len: number;
  logit_lens_top_k: TokenPrediction[][][]; // [n_layers+1][seq_len][top_k]
  final_top_k: TokenPrediction[][]; // [seq_len][top_k]
  attention_patterns: number[][][][]; // [n_layers][n_heads][seq_len][seq_len]
  interventions_applied?: Intervention[];
};

export type DetailTab = "predictions" | "attention";

export type TraceViewState = {
  selectedTokenIdx: number;
  selectedLayerIdx: number;
  detailTab: DetailTab;
};

export type Comparison = {
  prompt_a: string;
  prompt_b: string;
  n_layers: number;
  kl_symmetric: number[]; // length n_layers + 1, index 0 = embedding
  cosine_distance: number[];
};

export type AttentionCompareMode = "A" | "B" | "diff";

export type CompareViewState = {
  selectedTokenIdx: number;
  selectedLayerIdx: number;
  detailTab: DetailTab;
};

export type TraceEffect = {
  // Signed change at the final token position:
  //   ablation: p(base_top1 in intervened) − p(base_top1 in base) — usually negative
  //   patching: p(source_top1 in intervened) − p(source_top1 in base) — usually positive
  magnitude: number;
  trackedToken: string;
  direction: "down" | "up"; // which arrow to draw in the badge
};

export type TraceMetadata = {
  baseTraceId: string | null; // trace this one was derived from via intervention
  effect?: TraceEffect;
};

export type SweepScope = "current_layer" | "layer_range" | "all_layers";

export type SweepRequestBody = {
  prompt: string;
  scope: SweepScope;
  layer?: number;
  layer_start?: number;
  layer_end?: number;
  target_token?: string;
  base_interventions?: Intervention[];
};

export type SweepResultItem = {
  layer: number;
  head: number;
  baseline_prob: number;
  ablated_prob: number;
  effect: number;
};

export type SweepResponse = {
  prompt: string;
  target_token: string;
  target_token_baseline_prob: number;
  n_layers_swept: number;
  n_heads_per_layer: number;
  total_ablations: number;
  results: SweepResultItem[];
};

export type SweepState = {
  isRunning: boolean;
  progress: { current: number; total: number } | null;
  results: SweepResultItem[] | null;
  targetToken: string | null;
  baselineProb: number | null;
  scopeUsed: SweepScope;
};
