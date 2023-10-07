# Deep-Tutorials-for-MindSpore

The code of this repository is referenced to [Deep-Tutorial-for-PyTorch](https://github.com/sgrvinod/Deep-Tutorials-for-PyTorch)

---

These tutorials is the implementation of some typical papers. Below is the code directories and their corresponding papers.

Tutorial | Paper
:---: | :---:
Image Captioning | [_Show, Attend, and Tell_](https://arxiv.org/abs/1502.03044)
Sequence Labeling | [_Empower Sequence Labeling with Task-Aware Neural Language Model_](https://arxiv.org/abs/1709.04109)
Object Detection | [_SSD: Single Shot MultiBox Detector_](https://arxiv.org/abs/1512.02325)
Text Classification | [_Hierarchical Attention Networks for Document Classification_](https://www.semanticscholar.org/paper/Hierarchical-Attention-Networks-for-Document-Yang-Yang/1967ad3ac8a598adc6929e9e6b9682734f789427)
Super-Resolution | [_Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network_](https://arxiv.org/abs/1609.04802)
Machine Translation | [_Attention Is All You Need_](https://arxiv.org/abs/1706.03762)

---

Take ImageCaptioning as an example to introduce the file dictionary structure, the others are similar.
```
.
|--ImageCaptioning
|    |--create_input_files.py // Process source data files
|    |--utils.py              // Utility module
|    |--datasets.py           // Create data source for GeneratorDataset
|    |--models.py             // Model file
|    |--train.py              // Train the model
|    |--eval.py               // Evaluate the model
|    |--caption.py            // Caption the input image
```