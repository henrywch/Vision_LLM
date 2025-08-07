> Github: https://github.com/henrywch/Vision_LLM  
> Huggingface: https://huggingface.co/henrywch2huggingface/llavanext-scaled-0.5b

## LLaVA-NeXT Reproduction Scaled

### Configs

- Devices (AutoDL)
  - RTX 4090 24G * 2 (Stage 2 RTX 4090 24G * 4)
  - 64 vCPU Intel(R) Xeon(R) Gold 6430
- Models
  - LLM: Qwen2.5-0.5B-Instruct
  - ViT: SigLIP-SO400M-Patch14-384
- Datasets
  - blip_laion_cc_sbu_558k
  - coco30k_stage1.5_finetune_w_prompt (sampled from `coco118k_stage1.5_finetune_w_prompt`)
  - llava_v1_5_mix60k (pulled source: `llava_v1_5_mix665k`)
- Tools
  - DeepSpeed: For GPU Memory Optimization
  - OpenCompass/VLMEvalKit: For VLM Evaluation

### Scripts

#### Stage 1 (Pretrain): Vision-Text Alignment

> LLaVA-NeXT/scripts/train/pretrain_siglip.sh

```bash

export NUM_GPUS=2
export NNODES=1
export RANK=0
export ADDR=localhost
export PORT=29500

export OMP_NUM_THREADS=8
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_SOCKET_IFNAME=eth0
export NCCL_DEBUG=INFO

LLM_VERSION="/root/autodl-tmp/models/qwen2.5-0.5b-instr"
LLM_VERSION_CLEAN="${LLM_VERSION//\//_}"
VISION_MODEL_VERSION="/root/autodl-tmp/models/siglip-so400m-p14-384"
VISION_MODEL_VERSION_CLEAN="${VISION_MODEL_VERSION//\//_}"

############### Pretrain ################

PROMPT_VERSION=plain

BASE_RUN_NAME="llavanext-${VISION_MODEL_VERSION_CLEAN}-${LLM_VERSION_CLEAN}-mlp2x_gelu-pretrain_blip558k_plain"
echo "BASE_RUN_NAME: ${BASE_RUN_NAME}"

ACCELERATE_CPU_AFFINITY=1 torchrun --nproc_per_node="${NUM_GPUS}" --nnodes="${NNODES}" --node_rank="${RANK}" --master_addr="${ADDR}" --master_port="${PORT}" \
    llava/train/train_mem.py \
    --deepspeed scripts/zero3.json \
    --model_name_or_path ${LLM_VERSION} \
    --version ${PROMPT_VERSION} \
    --data_path /root/autodl-tmp/datasets/LLaVA_Train/LLaVA_PT/blip_laion_cc_sbu_558k.json \
    --image_folder /root/autodl-tmp/datasets/LLaVA_Train/LLaVA_PT/images \
    --vision_tower ${VISION_MODEL_VERSION} \
    --mm_tunable_parts="mm_mlp_adapter" \
    --mm_vision_select_layer -2 \
    --mm_projector_type mlp2x_gelu \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir checkpoints/projectors/${BASE_RUN_NAME} \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "no" \
    --save_strategy "no" \
    --save_steps 10000 \
    --learning_rate 1e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
    --tf32 True \
    --model_max_length 8192 \
    --gradient_checkpointing True \
    --dataloader_num_workers 16 \
    --lazy_preprocess True \
    --report_to tensorboard \
    --run_name $BASE_RUN_NAME \
    --attn_implementation sdpa

# You can delete the sdpa attn_implementation if you want to use flash attn

```

- Loss

```json

{
  "epoch": 1.0,
  "step": 8720,
  "total_flos": 91701433851904.0,
  "train_loss": 2.9441032562780816,
  "train_runtime": 22781.75,
  "train_samples_per_second": 24.499,
  "train_steps_per_second": 0.383
}

```

#### Stage 1.5 (Mid_Stage_Finetune): High-Quality Knowledge Learning

> LLaVA-NeXT/scripts/train/direct_finetune_siglip_a4.sh

```bash

export NUM_GPUS=2
export NNODES=1
export RANK=0
export ADDR=localhost
export PORT=29500

export OMP_NUM_THREADS=8
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_SOCKET_IFNAME=eth0
export NCCL_DEBUG=INFO

LLM_VERSION="/root/autodl-tmp/models/qwen2.5-0.5b-instr"
LLM_VERSION_CLEAN="${LLM_VERSION//\//_}"
VISION_MODEL_VERSION="/root/autodl-tmp/models/siglip-so400m-p14-384"
VISION_MODEL_VERSION_CLEAN="${VISION_MODEL_VERSION//\//_}"

############### Pretrain ################

PROMPT_VERSION="qwen_1_5"

BASE_RUN_NAME="llavanext-${VISION_MODEL_VERSION_CLEAN}-${LLM_VERSION_CLEAN}-mlp2x_gelu-pretrain_blip558k_plain"
MID_RUN_NAME="llavanext-${VISION_MODEL_VERSION_CLEAN}-${LLM_VERSION_CLEAN}-mlp2x_gelu-stage1.5_co118kf_qwen_1_5"
echo "BASE_RUN_NAME: ${BASE_RUN_NAME}"
echo "MID_RUN_NAME: ${MID_RUN_NAME}"

CKPT_PATH=$LLM_VERSION # this could also be the previous stage checkpoint

ACCELERATE_CPU_AFFINITY=1 torchrun --nproc_per_node="${NUM_GPUS}" --nnodes="${NNODES}" --node_rank="${RANK}" --master_addr="${ADDR}" --master_port="${PORT}" \
    llava/train/train_mem.py \
    --deepspeed scripts/zero3.json \
    --model_name_or_path ${CKPT_PATH} \
    --version ${PROMPT_VERSION} \
    --data_path="/root/autodl-tmp/datasets/LLaVA_Train/LLaVA_S1_5/coco30k_stage1.5_finetune_w_prompt.json" \
    --image_folder /root/autodl-tmp/datasets/LLaVA_Train/LLaVA_S1_5/ \
    --pretrain_mm_mlp_adapter="/root/autodl-tmp/models/projectors/${BASE_RUN_NAME}/mm_projector.bin" \
    --mm_tunable_parts="mm_vision_tower,mm_mlp_adapter,mm_language_model" \
    --mm_vision_tower_lr=2e-6 \
    --vision_tower ${VISION_MODEL_VERSION} \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --group_by_modality_length True \
    --image_aspect_ratio anyres \
    --image_grid_pinpoints "[(384, 768), (768, 384), (768, 768), (1152, 384), (384, 1152)]" \
    --mm_patch_merge_type spatial_unpad \
    --bf16 True \
    --run_name ${MID_RUN_NAME} \
    --output_dir "checkpoints/${MID_RUN_NAME}" \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 1 \
    --learning_rate 1e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 2 \
    --tf32 True \
    --model_max_length 32768 \
    --dataloader_num_workers 16 \
    --lazy_preprocess True \
    --report_to tensorboard \
    --dataloader_drop_last True \
    --attn_implementation flash_attention_2 \
    --torch_compile False
    # --torch_compile_backend "eager"

# You can delete the sdpa attn_implementation if you want to use flash attn

```

- Loss

```json

{
  "epoch": 1.0,
  "step": 937,
  "total_flos": 46654632226816.0,
  "train_loss": 0.8997930888178,
  "train_runtime": 17224.4313,
  "train_samples_per_second": 1.742,
  "train_steps_per_second": 0.054
}

```

#### Stage 2 (Finetune): Visual Instruction Tuning

> LLaVA-NeXT/scripts/train/direct_finetune_siglip_a4.sh

```bash

export NUM_GPUS=4
export NNODES=1
export RANK=0
export ADDR=localhost
export PORT=29500

export OMP_NUM_THREADS=8
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_SOCKET_IFNAME=eth0
export NCCL_DEBUG=INFO

LLM_VERSION="/root/autodl-tmp/models/qwen2.5-0.5b-instr" 
# for 7b model we recommend bs=1, accum=2, 16 nodes, 128 gpus, lr=1e-5, warmup=0.03
# for 72b model we recommend bs=1, accum=1, 32 nodes, 256 gpus, lr=1e-5, warmup=0.03
LLM_VERSION_CLEAN="${LLM_VERSION//\//_}"
VISION_MODEL_VERSION="/root/autodl-tmp/models/siglip-so400m-p14-384"
VISION_MODEL_VERSION_CLEAN="${VISION_MODEL_VERSION//\//_}"

############### Pretrain ################

BASE_RUN_NAME="llavanext-_root_autodl-tmp_models_siglip-so400m-p14-384-_root_autodl-tmp_models_qwen2.5-0.5b-instr-mlp2x_gelu-pretrain_blip558k_plain"
echo "BASE_RUN_NAME: ${BASE_RUN_NAME}"
MID_RUN_NAME="llavanext-${VISION_MODEL_VERSION_CLEAN}-${LLM_VERSION_CLEAN}-mlp2x_gelu-stage1.5_co118kf_qwen_1_5"
echo "MID_RUN_NAME: ${MID_RUN_NAME}"

############### Finetune ################

# Stage 2
PROMPT_VERSION="qwen_1_5"
llavanext-${VISION_MODEL_VERSION_CLEAN}-${LLM_VERSION_CLEAN}-mlp2x_gelu-stage2_mix665k_plain
NEXT_RUN_NAME="llavanext-${VISION_MODEL_VERSION_CLEAN}-${LLM_VERSION_CLEAN}-mlp2x_gelu-stage2_mix665k_qwen_1_5" 
PREV_STAGE_CHECKPOINT="/root/autodl-tmp/models/llavanext-mid-0.5b" # replace it with your last checkpoint training from mid stage
echo "PREV_STAGE_CHECKPOINT: ${PREV_STAGE_CHECKPOINT}"
echo "NEXT_RUN_NAME: ${NEXT_RUN_NAME}"

ACCELERATE_CPU_AFFINITY=1 torchrun --nproc_per_node="${NUM_GPUS}" --nnodes="${NNODES}" --node_rank="${RANK}" --master_addr="${ADDR}" --master_port="${PORT}" \
    llava/train/train_mem.py \
    --deepspeed scripts/zero3.json \
    --model_name_or_path ${PREV_STAGE_CHECKPOINT} \
    --version ${PROMPT_VERSION} \
    --data_path /root/autodl-tmp/datasets/LLaVA_Train/LLaVA_SFT/llava_v1_5_mix60k.json \
    --image_folder /root/autodl-tmp/datasets/LLaVA_Train/LLaVA_SFT \
    --mm_tunable_parts="mm_vision_tower,mm_mlp_adapter,mm_language_model" \
    --mm_vision_tower_lr=2e-6 \
    --vision_tower ${VISION_MODEL_VERSION} \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --group_by_modality_length True \
    --image_aspect_ratio anyres_max_9 \
    --image_grid_pinpoints  "(1x1),...,(6x6)" \
    --mm_patch_merge_type spatial_unpad \
    --bf16 True \
    --run_name ${NEXT_RUN_NAME} \
    --output_dir /root/autodl-tmp/models/checkpoints/llavanext/${NEXT_RUN_NAME} \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 1 \
    --learning_rate 1e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 2 \
    --tf32 True \
    --model_max_length 32768 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to tensorboard \
    --dataloader_drop_last True \
    --torch_compile False \
    --attn_implementation flash_attention_2
    # --video_folder your_video_dataset_path \
    # --frames_upbound 32 \
    # --torch_compile_backend "inductor" 
exit 0;

```

- Loss

```json

{
  "epoch": 1.0,
  "step": 937,
  "total_flos": 111946760044544.0,
  "train_loss": 0.7040835454853266,
  "train_runtime": 20346.5666,
  "train_samples_per_second": 2.949,
  "train_steps_per_second": 0.046
}

```

> Appendix: DeepSpeed/ZeRO3 Config

```json

{
    "fp16": {
        "enabled": "auto",
        "loss_scale": 0,
        "loss_scale_window": 1000,
        "initial_scale_power": 16,
        "hysteresis": 2,
        "min_loss_scale": 1
    },
    "bf16": {
        "enabled": "auto"
    },

    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": true
        },
        "offload_param": {
            "device": "cpu",
            "pin_memory": true
        },
        "overlap_comm": true,
        "contiguous_gradients": true,
        "sub_group_size": 1e9,
        "reduce_bucket_size": "auto",
        "stage3_prefetch_bucket_size": "auto",
        "stage3_param_persistence_threshold": "auto",
        "stage3_max_live_parameters": 1e9,
        "stage3_max_reuse_distance": 1e9,
        "stage3_gather_16bit_weights_on_model_save": true
    },

    "gradient_accumulation_steps": "auto",
    "gradient_clipping": "auto",
    "steps_per_print": 100,
    "train_batch_size": "auto",
    "train_micro_batch_size_per_gpu": "auto",
    "wall_clock_breakdown": false
}


```

### Results (MMBench_DEV_EN)

```bash

--------------------------------------  --------------------
split                                   dev
Overall                                 0.5420962199312714
AR                                      0.6331658291457286
CP                                      0.6554054054054054
FP-C                                    0.44755244755244755
FP-S                                    0.5426621160409556
LR                                      0.2966101694915254
RR                                      0.4608695652173913
action_recognition                      0.8703703703703703
attribute_comparison                    0.22727272727272727
attribute_recognition                   0.6486486486486487
celebrity_recognition                   0.6767676767676768
function_reasoning                      0.7088607594936709
future_prediction                       0.4
identity_reasoning                      0.8888888888888888
image_emotion                           0.72
image_quality                           0.018867924528301886
image_scene                             0.9326923076923077
image_style                             0.6226415094339622
image_topic                             0.75
nature_relation                         0.3958333333333333
object_localization                     0.2222222222222222
ocr                                     0.6666666666666666
physical_property_reasoning             0.4
physical_relation                       0.20833333333333334
social_relation                         0.6744186046511628
spatial_relationship                    0.15555555555555556
structuralized_imagetext_understanding  0.24358974358974358
--------------------------------------  --------------------

```

### Notations

> The model config has been aligned with lmms-lab/llava-onevision-qwen2-0.5b-si (num_attention_heads, max_window_layers, processor_config...) for compatibility in VLMEvalKit Evaluation, special thanks to **LLaVA-NeXT**

> LLaVA-NeXT/model/multimodal_encoder/build.py line 16 was modified to `    if "clip" in vision_tower or (vision_tower.startswith("openai") or vision_tower.startswith("laion") or "ShareGPT4V" in vision_tower):` for local *vision_tower* load. Additionally, 
> ```python
> 
> 565 if "/root/autodl-tmp/models/llavanext-scaled-0.5b" in model_path:
> 566     model_name = "llava-onevision-qwen2-0.5b-si"
> 567 else: 
> 568     model_name = get_model_name_from_path(model_path)
> 
> ```
> VLMEvalKit/vlmeval/vlm/llava/llava.py was adapted as above to include my reproduction into the supported.

> As datasets (Stage 1.5 & 2) are reduced, the result may not as satisfying as the official model, which achieves: 
> ```bash
> 
> --------------------------------------  --------------------
> Overall                                 0.5506872852233677
> AR                                      0.5979899497487438
> CP                                      0.6959459459459459
> FP-C                                    0.3776223776223776
> FP-S                                    0.590443686006826
> LR                                      0.23728813559322035
> RR                                      0.5304347826086957
> action_recognition                      0.8703703703703703
> attribute_comparison                    0.0
> attribute_recognition                   0.7432432432432432
> celebrity_recognition                   0.7171717171717171
> function_reasoning                      0.7468354430379747
> future_prediction                       0.375
> identity_reasoning                      0.9333333333333333
> image_emotion                           0.82
> image_quality                           0.05660377358490566
> image_scene                             0.9326923076923077
> image_style                             0.7358490566037735
> image_topic                             0.7222222222222222
> nature_relation                         0.3541666666666667
> object_localization                     0.24691358024691357
> ocr                                     0.6923076923076923
> physical_property_reasoning             0.24
> physical_relation                       0.4166666666666667
> social_relation                         0.7906976744186046
> spatial_relationship                    0.15555555555555556
> structuralized_imagetext_understanding  0.16666666666666666
> --------------------------------------  --------------------
> 
> ```
> Its loss is:
> ```json
> 
> {
>   "epoch": 1.0,
>   "step": 12464,
>   "total_flos": 7038884969086976.0,
>   "train_loss": 0.8383206710774112,
>   "train_runtime": 88593.6851,
>   "train_samples_per_second": 36.017,
>   "train_steps_per_second": 0.141
> }
> 
> ```
>
