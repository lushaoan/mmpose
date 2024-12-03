default_scope = 'mmpose'
# custom hooks
custom_hooks = [
    # Synchronize model buffers such as running_mean and running_var in BN
    # at the end of each epoch
    dict(type='SyncBuffersHook')
]

# multi-processing backend
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)

# visualizer
vis_backends = [
    dict(type='LocalVisBackend'),
    # dict(type='TensorboardVisBackend'),
    # dict(type='WandbVisBackend'),
]
visualizer = dict(
    type='PoseLocalVisualizer', vis_backends=vis_backends, name='visualizer')

# logger
log_processor = dict(
    type='LogProcessor', window_size=50, by_epoch=True, num_digits=6)
log_level = 'INFO'
load_from = None
resume = False

# file I/O backend
backend_args = dict(backend='local')

# training/validation/testing progress
# val_cfg = None

# runtime
train_cfg = dict(type="EpochBasedTrainLoop", max_epochs=210, val_interval=10)

# optimizer
optim_wrapper = dict(optimizer=dict(
    type='Adam',
    lr=5e-4,
))

# learning policy
param_scheduler = [
    dict(
        type='LinearLR', begin=0, end=500, start_factor=0.001,
        by_epoch=False),  # warm-up
    dict(
        type='MultiStepLR',
        begin=0,
        end=210,
        milestones=[170, 200],
        gamma=0.1,
        by_epoch=True)
]

# automatically scaling LR based on the actual training batch size
auto_scale_lr = dict(base_batch_size=512)

# hooks
default_hooks = dict(
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=10))

# codec settings
codec = dict(
    type='MSRAHeatmap', input_size=(320, 320), heatmap_size=(80, 80), sigma=3)

# model settings
model = dict(
    type='TopdownPoseEstimator',
    data_preprocessor=dict(
        type='PoseDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True),
    backbone=dict(
        type='MobileNetV2',
        widen_factor=1.,
        out_indices=(7, )),
    head=dict(
        type='HeatmapHead',
        in_channels=1280,
        out_channels=4,
        loss=dict(type='KeypointMSELoss', use_target_weight=True),
        decoder=codec),
    test_cfg=dict(
        flip_test=False,
        flip_mode='heatmap',
        shift_heatmap=True)
    )

# base dataset settings
dataset_type = 'BarcodeDataset'

# pipelines
train_pipeline = [
    dict(type='LoadImage'),
    dict(type='GetBBoxCenterScale'),
    # dict(type='RandomFlip', direction='horizontal'),
    dict(type='RandomBBoxTransform'),
    dict(type='TopdownAffine', input_size=codec['input_size']),
    dict(type='GenerateTarget', encoder=codec),
    dict(type='PackPoseInputs')
]
val_pipeline = [
    dict(type='LoadImage'),
    dict(type='GetBBoxCenterScale'),
    dict(type='TopdownAffine', input_size=codec['input_size']),
    dict(type='PackPoseInputs')
]

# data loaders
train_dataloader = dict(
    batch_size=64,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        train_file_infos=[
            {
                "root": "/dataset/sharedir/industrial/PUBLIC/Datacode/newdata/datacode_precloc/240902/",
                "path": "/dataset/shaoanlu/github/mmlab/train_set/precloc/2d/train.txt"
            }
        ],
        pipeline=train_pipeline,
    ))
val_dataloader = dict(
    batch_size=32,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=dict(
        type=dataset_type,
        train_file_infos=[
            {
                "root": "/dataset/sharedir/industrial/PUBLIC/Datacode/newdata/datacode_precloc/240902/",
                "path": "/dataset/shaoanlu/github/mmlab/train_set/precloc/2d/train.txt"
            }
        ],
        pipeline=val_pipeline,
    ))
test_dataloader = val_dataloader

# # evaluators
# val_evaluator = dict(
#     type='CocoMetric',
#     ann_file=data_root + 'annotations/person_keypoints_val2017.json')
# test_evaluator = val_evaluator
