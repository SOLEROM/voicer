# torchinfo summary

```
==============================================================================================================
Layer (type:depth-idx)                                       Output Shape              Param #
==============================================================================================================
ReDimNetWrap                                                 [1, 192]                  --
├─MelBanks: 1-1                                              [1, 72, 201]              --
│    └─Sequential: 2-1                                       [1, 72, 201]              --
│    │    └─NormalizeAudio: 3-1                              [1, 1, 32000]             --
│    │    └─PreEmphasis: 3-2                                 [1, 32000]                --
│    │    └─MelSpectrogram: 3-3                              [1, 72, 201]              --
├─ReDimNet: 1-2                                              [1, 1728, 201]            --
│    └─Sequential: 2-2                                       [1, 1728, 201]            --
│    │    └─Conv2d: 3-4                                      [1, 24, 72, 201]          240
│    │    └─LayerNorm: 3-5                                   [1, 24, 72, 201]          48
│    │    └─to1d: 3-6                                        [1, 1728, 201]            --
│    └─Sequential: 2-3                                       [1, 1728, 201]            --
│    │    └─weigth1d: 3-7                                    [1, 1728, 201]            (1)
│    │    └─to2d: 3-8                                        [1, 24, 72, 201]          --
│    │    └─Conv2d: 3-9                                      [1, 48, 72, 201]          1,200
│    │    └─ConvBlock2d: 3-10                                [1, 48, 72, 201]          11,808
│    │    └─ConvBlock2d: 3-11                                [1, 48, 72, 201]          11,808
│    │    └─ConvBlock2d: 3-12                                [1, 48, 72, 201]          11,808
│    │    └─ConvBlock2d: 3-13                                [1, 48, 72, 201]          11,808
│    │    └─Sequential: 3-14                                 [1, 24, 72, 201]          4,128
│    │    └─to1d: 3-15                                       [1, 1728, 201]            --
│    │    └─TimeContextBlock1d: 3-16                         [1, 1728, 201]            312,840
│    └─Sequential: 2-4                                       [1, 1728, 201]            --
│    │    └─weigth1d: 3-17                                   [1, 1728, 201]            3,456
│    │    └─to2d: 3-18                                       [1, 24, 72, 201]          --
│    │    └─Conv2d: 3-19                                     [1, 96, 36, 201]          4,704
│    │    └─ConvBlock2d: 3-20                                [1, 96, 36, 201]          32,832
│    │    └─ConvBlock2d: 3-21                                [1, 96, 36, 201]          32,832
│    │    └─ConvBlock2d: 3-22                                [1, 96, 36, 201]          32,832
│    │    └─ConvBlock2d: 3-23                                [1, 96, 36, 201]          32,832
│    │    └─ConvBlock2d: 3-24                                [1, 96, 36, 201]          32,832
│    │    └─Sequential: 3-25                                 [1, 48, 36, 201]          9,408
│    │    └─to1d: 3-26                                       [1, 1728, 201]            --
│    │    └─TimeContextBlock1d: 3-27                         [1, 1728, 201]            312,840
│    └─Sequential: 2-5                                       [1, 1728, 201]            --
│    │    └─weigth1d: 3-28                                   [1, 1728, 201]            5,184
│    │    └─to2d: 3-29                                       [1, 48, 36, 201]          --
│    │    └─Conv2d: 3-30                                     [1, 96, 36, 201]          4,704
│    │    └─ConvBlock2d: 3-31                                [1, 96, 36, 201]          32,832
│    │    └─ConvBlock2d: 3-32                                [1, 96, 36, 201]          32,832
│    │    └─ConvBlock2d: 3-33                                [1, 96, 36, 201]          32,832
│    │    └─ConvBlock2d: 3-34                                [1, 96, 36, 201]          32,832
│    │    └─ConvBlock2d: 3-35                                [1, 96, 36, 201]          32,832
│    │    └─ConvBlock2d: 3-36                                [1, 96, 36, 201]          32,832
│    │    └─Sequential: 3-37                                 [1, 48, 36, 201]          9,408
│    │    └─to1d: 3-38                                       [1, 1728, 201]            --
│    │    └─TimeContextBlock1d: 3-39                         [1, 1728, 201]            312,840
│    └─Sequential: 2-6                                       [1, 1728, 201]            --
│    │    └─weigth1d: 3-40                                   [1, 1728, 201]            6,912
│    │    └─to2d: 3-41                                       [1, 48, 36, 201]          --
│    │    └─Conv2d: 3-42                                     [1, 96, 18, 201]          9,312
│    │    └─ConvBlock2d: 3-43                                [1, 96, 18, 201]          32,832
│    │    └─ConvBlock2d: 3-44                                [1, 96, 18, 201]          32,832
│    │    └─ConvBlock2d: 3-45                                [1, 96, 18, 201]          32,832
│    │    └─ConvBlock2d: 3-46                                [1, 96, 18, 201]          32,832
│    │    └─ConvBlock2d: 3-47                                [1, 96, 18, 201]          32,832
│    │    └─ConvBlock2d: 3-48                                [1, 96, 18, 201]          32,832
│    │    └─ConvBlock2d: 3-49                                [1, 96, 18, 201]          32,832
│    │    └─ConvBlock2d: 3-50                                [1, 96, 18, 201]          32,832
│    │    └─to1d: 3-51                                       [1, 1728, 201]            --
│    │    └─TimeContextBlock1d: 3-52                         [1, 1728, 201]            312,840
│    └─Sequential: 2-7                                       [1, 1728, 201]            --
│    │    └─weigth1d: 3-53                                   [1, 1728, 201]            8,640
│    │    └─to2d: 3-54                                       [1, 96, 18, 201]          --
│    │    └─Conv2d: 3-55                                     [1, 96, 18, 201]          9,312
│    │    └─ConvBlock2d: 3-56                                [1, 96, 18, 201]          32,832
│    │    └─ConvBlock2d: 3-57                                [1, 96, 18, 201]          32,832
│    │    └─ConvBlock2d: 3-58                                [1, 96, 18, 201]          32,832
│    │    └─ConvBlock2d: 3-59                                [1, 96, 18, 201]          32,832
│    │    └─ConvBlock2d: 3-60                                [1, 96, 18, 201]          32,832
│    │    └─ConvBlock2d: 3-61                                [1, 96, 18, 201]          32,832
│    │    └─ConvBlock2d: 3-62                                [1, 96, 18, 201]          32,832
│    │    └─ConvBlock2d: 3-63                                [1, 96, 18, 201]          32,832
│    │    └─to1d: 3-64                                       [1, 1728, 201]            --
│    │    └─TimeContextBlock1d: 3-65                         [1, 1728, 201]            312,840
│    └─Sequential: 2-8                                       [1, 1728, 201]            --
│    │    └─weigth1d: 3-66                                   [1, 1728, 201]            10,368
│    │    └─to2d: 3-67                                       [1, 96, 18, 201]          --
│    │    └─Conv2d: 3-68                                     [1, 192, 9, 201]          37,056
│    │    └─ConvBlock2d: 3-69                                [1, 192, 9, 201]          102,528
│    │    └─ConvBlock2d: 3-70                                [1, 192, 9, 201]          102,528
│    │    └─ConvBlock2d: 3-71                                [1, 192, 9, 201]          102,528
│    │    └─to1d: 3-72                                       [1, 1728, 201]            --
│    │    └─TimeContextBlock1d: 3-73                         [1, 1728, 201]            312,840
│    └─weigth1d: 2-9                                         [1, 1728, 201]            12,096
│    └─Identity: 2-10                                        [1, 1728, 201]            --
│    └─Identity: 2-11                                        [1, 1728, 201]            --
├─ASTP: 1-3                                                  [1, 3456]                 --
│    └─Conv1d: 2-12                                          [1, 128, 201]             663,680
│    └─Conv1d: 2-13                                          [1, 1728, 201]            222,912
├─BatchNorm1d: 1-4                                           [1, 3456]                 6,912
├─Linear: 1-5                                                [1, 192]                  663,744
==============================================================================================================
Total params: 4,811,745
Trainable params: 4,811,744
Non-trainable params: 1
Total mult-adds (G): 6.62
==============================================================================================================
Input size (MB): 0.13
Forward/backward pass size (MB): 926.91
Params size (MB): 19.25
Estimated Total Size (MB): 946.28
==============================================================================================================
```