# Flexible-Conv-Module

<p>
    <a href="#"><img alt="Python" src="https://img.shields.io/badge/Python-14354C.svg?logo=python&logoColor=white"></a>
    <a href="#"><img alt="OnnxRuntime" src="https://img.shields.io/badge/OnnxRuntime-FF6F00.svg?logo=onnx&logoColor=white"></a>
    <a href="#"><img alt="Markdown" src="https://img.shields.io/badge/Markdown-000000.svg?logo=markdown&logoColor=white"></a>
    <a href="#"><img alt="Visual Studio Code" src="https://img.shields.io/badge/Visual%20Studio%20Code-ad78f7.svg?logo=visual-studio-code&logoColor=white"></a>
    <a href="#"><img alt="Linux" src="https://img.shields.io/badge/Linux-0078D6?logo=linux&logoColor=white"></a>
    <a href="#"><img alt="Windows" src="https://img.shields.io/badge/Windows-0078D6?logo=windows&logoColor=white"></a>

</p>

The `Flexible-Conv-Module` library is a flexible toolbox of a series of CV algorithms based on PyTorch. Used to combine different modules to build different networks.

# ➤ Contents
1) [Requirements](#Requirements)

2) [Main Components](#MainComponents)
    - [Conv Basic Series](#ConvBasicSeries)
    - [Conv Block Series](#ConvBlockSeries)
    - [Conv ReceptionField Series](#ConvReceptionFieldSeries)
    - [Conv Backbone Series](#ConvBackboneSeries)
    - [ReParameter Series](#ReParameterSeries)
    - [Vision Transformer Series](#ViTSeries)
    - [Attention Series](#AttentionSeries)


<h1 id="Requirements">➤ Requirements🚀</h1>

* **Install :**

    The `requirements.txt` file should list all Python libraries that your notebooks
    depend on, and they will be installed using:

    ```
    pip install -r requirements.txt
    ```


<h1 id="MainComponents">➤ Main Components🚀</h1>

<div align="center">
  <b>Conv Module Components</b>
</div>

<table align="center">
  <tbody>
    <tr align="center" valign="bottom">
      <td>
        <b>Basic</b>
      </td>
      <td>
        <b>Block</b>
      </td>
      <td>
        <b>ReParameter</b>
      </td>
      <td>
        <b>ReceptionField</b>
      </td>
      <td>
        <b>Backbone</b>
      </td>
    </tr>
    <tr valign="top">
      <td>
        <ul>
            <li>CondConv</li>
            <li>DoConv</li>
            <li>DynamicConv</li>
            <li>PConv</li>
            <li>PsConv</li>
            <li>PyConv</li>
        </ul>
      </td>
      <td>
        <ul>
            <li>BsConv</li>
            <li>DwsConv</li>
            <li>GhostConv</li>
            <li>GnConv</li>
            <li>MixConv</li>
            <li>ScConv</li>
        </ul>
      </td>
      <td>
        <ul>
            <li>AC Block</li>
            <li>DiverseBranch Block</li>
            <li>Mobileone Block</li>
            <li>RepLK Block</li>
            <li>RepMLP Block</li>
            <li>RepVGG Block</li>
        </ul>
      </td>
      <td>
        <ul>
            <li>SPP Module</li>
            <li>SPPF Module</li>
            <li>ASPP Module</li>
            <li>SPPCSPC Module</li>
            <li>SPPFCSPC Module</li>
            <li>RFB Module</li>
        </ul>
      </td>
      <td>
        <ul>
            <li>BottleNeck Block</li>
            <li>Resnet Block</li>
            <li>Resnext Block</li>
            <li>VarGroup Block</li>
            <li>VoV Block</li>
            <li>Pelee Block</li>
            <li>EPSA Block</li>
        </ul>
      </td>
    </tr>
  </tbody>
</table>


<div align="center">
  <b>Transformer Module Componens</b>
</div>

<table align="center">
  <tbody>
    <tr align="center" valign="bottom">
      <td>
        <b>Attention</b>
      </td>
      <td>
        <b>ViT</b>
      </td>
    </tr>
    <tr valign="top">
      <td>
        <ul>
            <li>CBAModule</li>
            <li>coordAttModule</li>
            <li>EAModule</li>
            <li>ECAModule</li>
            <li>GAModule</li>
            <li>LKAModule</li>
            <li>SEModule</li>
            <li>shuffleAttModule</li>
            <li>simAModule</li>
            <li>SKModule</li>
            <li>ULSAModule</li>
            <li>PSAModule</li>
        </ul>
      </td>
      <td>
        <ul>
            <li>ConvNext Block</li>
            <li>EdgeViT Block</li>
            <li>Hor Block</li>
            <li>MobileVit Block</li>
            <li>PVTv1 Block</li>
            <li>PVTv2 Block</li>
            <li>TnT Block</li>
            <li>WaveMLP Block</li>
            <li>STViT Block</li>
        </ul>
      </td>
    </tr>
  </tbody>
</table>


<details open>
<summary> <b id="ConvBasicSeries">👉 Conv Basic Series</b> </summary>

- 1 - [CondConv: Conditionally Parameterized Convolutions for Efficient Inference](https://arxiv.org/abs/1904.04971)
  - NeurIPS 2019
  - Usage : `CondConv2d`

- 2 - [DO-Conv: Depthwise Over-parameterized Convolutional Layer](https://arxiv.org/abs/2006.12030)
  - TIP 2022
  - Usage : `DoConv2d`

- 3 - [Dynamic Convolution: Attention over Convolution Kernels](https://arxiv.org/abs/1912.03458)
  - CVPR 2020 
  - Usage : `DynamicConv2d`

- 4 - [PSConv: Squeezing Feature Pyramid into One Compact Poly-Scale Convolutional Layer](https://arxiv.org/abs/2007.06191)
  - ECCV 2020
  - Usage : `PSConv2d`

- 5 - [Pyramidal Convolution: Rethinking Convolutional Neural Networks for Visual Recognition](https://arxiv.org/pdf/2006.11538)
  - CVPR 2020
  - Usage : `PyConv2d`

- 6 - [Run, Don’t Walk: Chasing Higher FLOPS for Faster Neural Networks](https://arxiv.org/abs/2303.03667)
  - CVPR 2023
  - Usage : `PConv2d`

</details>

<details open>
<summary> <b id="ConvBlockSeries">👉 Conv Block Series</b> </summary>

- 1 - [Blueprint Separable Residual Network for Efficient Image Super-Resolution](https://arxiv.org/abs/2205.05996)
  - CVPR 2022
  - Usage : `BlueprintSeparableUConv2d`, `BlueprintSeparableSConv2d`

- 2 - [PP-LCNet: A Lightweight CPU Convolutional Neural Network](https://arxiv.org/pdf/2109.15099)
  - CVPR 2021
  - Improve by DW Conv
  - Usage : `DepthwiseSeparableConv2d`

- 3.1 - [GhostNet: More Features from Cheap Operations](​​​​​​https://arxiv.org/abs/1911.11907v2)
  - CVPR 2020
  - Usage : `GhostConv2d`

- 3.2 - [GhostNetV2: Enhance Cheap Operation with Long-Range Attention](https://arxiv.org/abs/2211.12905)
  - NeurIPS 2022
  - Usage : `Ghostv2Conv2d`

- 4 - [HorNet: Efficient High-Order Spatial Interactions with Recursive Gated Convolutions](https://arxiv.org/abs/2207.14284)
  - NeurIPS 2022
  - Usage : `RecursiveGatedConv2d`

- 5 - [MixConv: Mixed Depthwise Convolutional Kernels](https://arxiv.org/abs/1907.09595)
  - BMVC 2019
  - Usage : `MixedDepthwiseConv2d`

- 6 - SCConv: Spatial and Channel Reconstruction Convolution for Feature Redundancy
  - CVPR 2023
  - Usage : `SCReconstructConv2d`

</details>

<details open>
<summary> <b id="ConvReceptionFieldSeries">👉 Conv ReceptionField Series</b> </summary>

- 1 - [Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition](https://arxiv.org/abs/1406.4729)
  - CVPR 2015
  - Usage : `SPP`, `SPPBlock`

- 2.1 - [YOLOv5](https://github.com/ultralytics/yolov5)
  - Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher
  - Usage : `SPPF`, `SPPFBlock`

- 2.2 - [YOLOv6: A Single-Stage Object Detection Framework for Industrial Applications](https://arxiv.org/abs/2209.02976)
  - CVPR 2022
  - Usage : `SimSPPFBlock`

- 3.1 - [DeepLab: Semantic Image Segmentation with Deep Convolutional Nets, Atrous Convolution, and Fully Connected CRFs](https://arxiv.org/abs/1606.00915v2)
  - CVPR 2017
  - Usage : `ASPPv2`

- 3.2 - [Rethinking Atrous Convolution for Semantic Image Segmentation](https://arxiv.org/abs/1706.05587)
  - CVPR 2017
  - Usage : `ASPPv3`, `ASPPv3Block`

- 4 - [Receptive Field Block Net for Accurate and Fast Object Detection](https://arxiv.org/abs/1711.07767)
  - ECCV 2018
  - Usage : `RFBBlock`, `RFBsBlock`

- 5 - [YOLOv7: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors](https://arxiv.org/abs/2207.02696)
  - CVPR 2022
  - Usage : `SPPCSPCBlock`

- 6 - [YOLOv6 v3.0: A Full-Scale Reloading](https://arxiv.org/abs/2301.05586)
  - CVPR 2023
  - Usage : `SPPFCSPCBlock`

</details>

<details open>
<summary> <b id="ConvBackboneSeries">👉 Conv Backbone Series</b> </summary>

- 1 - [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
  - CVPR 2015
  - Usage : `ResnetUnit`, `ResnetStage`

- 2 - [Aggregated Residual Transformations for Deep Neural Networks](https://arxiv.org/abs/1611.05431)
  - CVPR 2017
  - Usage : `ResnextUnit`, `ResnextStage`

- 3 - [VarGFaceNet: An Efficient Variable Group Convolutional Neural Network for Lightweight Face Recognition](https://arxiv.org/pdf/1910.04985)
  - ICCV 2019
  - Usage : `VarGroupConv`, `VarGroupBlock`, `VarGroupBlock_DownSampling`, `VarGroupStage`

- 4 - [An Energy and GPU-Computation Efficient Backbone Network for Real-Time Object Detection]( https://arxiv.org/abs/1904.09730 )
  - CVPR 2019 
  - Usage : `OSAUnit`, `DWsOSAUnit`, `OSAStage`

- 5 - [Pelee: A Real-Time Object Detection System on Mobile Devices](https://arxiv.org/abs/1804.06882)
  - CVPR 2019
  - Usage : `PeleeUnit`, `PeleeStage`

- 6 - [EPSANet: An Efficient Pyramid Squeeze Attention Block on Convolutional Neural Network](https://arxiv.org/abs/2105.14447)
  - CVPR 2021
  - Usage : `EPSAUnit`, `EPSAStage`

</details>

<details open>
<summary> <b id="ReParameterSeries">👉 ReParameter Series</b> </summary>

- 1 - [ACNet: Strengthening the Kernel Skeletons for Powerful CNN via Asymmetric](https://arxiv.org/abs/1908.03930)
  - ICCV 2019
  - Usage : `ACBlock`

- 2 - [Diverse Branch Block: Building a Convolution as an Inception-like Unit](https://arxiv.org/abs/2103.13425)
  - CVPR 2021
  - Usage : `DBBlock`

- 3 - [RepVGG: Making VGG-style ConvNets Great Again](https://arxiv.org/abs/2101.03697)
  - CVPR 2021
  - Usage : `RepVGGBlock`, `RepVGGStage`

- 4 - [MobileOne: An Improved One millisecond Mobile Backbone](https://arxiv.org/abs/2206.04040v2)
  - CVPR 2022
  - Usage : `MobileOneBlock`, `MobileOneUnit`, `MobileOneStage`

- 5 - [RepMLPNet: Hierarchical Vision MLP with Re-parameterized Locality](https://arxiv.org/abs/2112.11081)
  - CVPR 2022
  - Usage : `RepMLPBlock`, `RepMLPUnit`

- 6 - [Scaling Up Your Kernels to 31x31: Revisiting Large Kernel Design in CNNs](https://arxiv.org/abs/2203.06717)
  - CVPR 2022
  - Usage : `RepLKBlock`, `RepLKUnit`, `RepLKStage`

</details>

<details open>
<summary> <b id="ViTSeries">👉 Vision Transformer Series</b> </summary>

- 1 - [Pyramid Vision Transformer: A Versatile Backbone for Dense Prediction without Convolutions](https://arxiv.org/abs/2102.12122)
  - ICCV 2021
  - Usage : `PVTv1Unit`, `PVTv1Stage`

- 2 - [Transformer in Transformer](https://arxiv.org/abs/2103.00112)
  - NeurIPS 2021
  - Usage : `TnTUnit`

- 3 - [MobileViT: Light-weight, General-purpose, and Mobile-friendly Vision Transformer](https://arxiv.org/abs/2110.02178)
  - ICLR 2021
  - Usage : `MobileViTUnit`, `MobileViTStage`

- 4 - [PVT v2: Improved Baselines with Pyramid Vision Transformer](https://arxiv.org/abs/2106.13797)
  - CVMJ 2022
  - Usage : `PVTv2Unit`, `PVTv2Stage`

- 5 - [An Image Patch is a Wave: Quantum Inspired Vision MLP]( https://arxiv.org/abs/2111.12294 )
  - CVPR 2022
  - Usage : `WaveUnit`, `WaveStage`

- 6 - [HorNet: Efficient High-Order Spatial Interactions with Recursive Gated Convolutions](https://arxiv.org/abs/2207.14284)
  - NeurIPS 2022
  - Usage : `HorUnit`, `HorStage`

- 7 - [EdgeViTs: Competing Light-weight CNNs on Mobile Devices with Vision Transformers](https://arxiv.org/abs/2205.03436)
  - ECCV 2022
  - Usage : `EdgeViTUnit`, `EdgeViTStage`

- 8 - [A ConvNet for the 2020s](https://arxiv.org/abs/2201.03545)
  - CVPR 2022
  - Usage : `ConvNeXtUnit`, `ConvNeXtStage`

- 9 - [Vision Transformer with Super Token Sampling](https://arxiv.org/abs/2211.11167)
  - CVPR 2023
  - Usage : `STViTUnit`, `STViTStage`

</details>


<details open>
<summary> <b id="AttentionSeries">👉 Attention Series</b> </summary>

- 1 - [CBAM: Convolutional Block Attention Module](https://arxiv.org/abs/1807.06521)
  - ECCV 2018
  - Usage : `CAMBlock_FC`, `CAMBlock_Conv`, `SAMBlock`, `CBAMBlock`

- 2 - [Squeeze-and-Excitation Networks](https://arxiv.org/abs/1709.01507)
  - CVPR 2018
  - Usage : `SEBlock`, `SEBlock_Conv`, `ESEBlock_Conv`

- 3 - [Selective Kernel Networks](https://arxiv.org/abs/1903.06586)
  - CVPR 2019
  - Usage : `SKBlock`

- 4 - [ULSAM: Ultra-Lightweight Subspace Attention Module for Compact Convolutional Neural Networks](https://arxiv.org/abs/2006.15102)
  - CVPR 2020
  - Usage : `ULSAMBlock`

- 5 - [ECA-Net: Efficient Channel Attention for Deep Convolutional Neural Networks](https://arxiv.org/abs/1910.03151)
  - CVPR 2020
  - Usage : `ECABlock`

- 6 - [Coordinate Attention for Efficient Mobile Network Design](https://arxiv.org/abs/2103.02907)
  - CVPR 2021
  - Usage : `CoordAttBlock`

- 7 - [Beyond Self-attention: External Attention using Two Linear Layers for Visual Tasks](https://arxiv.org/abs/2105.02358)
  - CVPR 2021
  - Usage : `EABlock`

- 8 - [Global Attention Mechanism: Retain Information to Enhance Channel-Spatial Interactions](https://arxiv.org/abs/2112.05561)
  - CVPR 2021
  - Usage : `GAMBlock`

- 9 - [SA-NET: SHUFFLE ATTENTION FOR DEEP CONVOLUTIONAL NEURAL NETWORKS](https://arxiv.org/abs/2102.00240)
  - ICASSP 2021
  - Usage : `ShuffleAttBlock`

- 10 - [EPSANet: An Efficient Pyramid Squeeze Attention Block on Convolutional Neural Network](https://arxiv.org/abs/2105.14447)
  - CVPR 2021
  - Usage : `PSABlock`

- 11 - [Simple Attention Module based Speaker Verification with Iterative noisy label detection](https://arxiv.org/abs/2110.06534)
  - ICASSP 2022
  - Usage : `simAMBlock`

- 12 - [Beyond Self-Attention: Deformable Large Kernel Attention for Medical Image Segmentation](https://arxiv.org/abs/2309.00121)
  - CVPR 2023
  - Usage : `LKAConv`, `deformable_LKAConv`, `LKABlock`


</details>
