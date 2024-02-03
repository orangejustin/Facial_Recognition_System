# README for Facial Recognition Model Training

## Overview
This README provides instructions on how to run the code for training facial recognition models, details about the experiments conducted, architectures used, hyperparameters, data loading schemes, and additional insights gained during the project.

## File Structure
- `hw2pw.ipynb`: Jupyter notebook that replicates the experiment.
- `run.py`: Python script for running the experiment.
- `test.py`: Python script for testing the model.
- Python scripts for various components of the project.

## Instructions

### Running Python Scripts
To run the experiment:
```bash
python3 run.py
```
To test the model:
```bash
python3 test.py
```

### Running Jupyter Notebook
Execute the cells in the following order:
1. Preliminaries
2. TODOs
3. Download Data from Kaggle
4. Configs
5. Classification Dataset
6. Data visualization
7. MyNetwork (resnet)
8. resnet + SE
9. Let's train!
10. Setup everything for training
11. Pretrain

**Note**: The pretrain model can be skipped initially. If interested, the pretrained self-resnetSE model is available at [Google Drive](https://drive.google.com/file/d/1-1cCmf0KAUzxISjM6x6uPEq9ClP1uJGi/view?usp=sharing).

### Wandb Setup
Set up your Wandb to record the experiment. The 'Experiments' session is crucial for training the model. The 'CutMix' session is optional and can be used if you wish to try CutMix in training.

### Further Fine-tuning
Sessions for 'verification Task: Validation', 'Constrative Loss', 'CenterLoss', 'ArcFace', 'Pretrain', and 'finetune' are provided to help further fine-tune your pretrained model using Contrastive Losses like CenterLoss and ArcFace.

The 'New data finetune' session is not necessary for this scenario but could be useful in the future when new data is provided.

## Experiments and Architecture
Two models were experimented with:
1. **self-Resnet**: Inspired by [ResNet Paper](https://arxiv.org/abs/1512.03385).
2. **self-resnetSE**: Based on [SENet Paper](https://ieeexplore.ieee.org/document/9771436).

Both models were trained end-to-end and achieved testing accuracy above 90%. The main data augmentation techniques used were:

```python
train_transforms = torchvision.transforms.Compose([
    torchvision.transforms.RandomCrop(224, padding=8, padding_mode='reflect'),
    torchvision.transforms.RandomGrayscale(p=0.12),
    torchvision.transforms.RandAugment(num_ops=2, magnitude=9),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.5103,0.4014,0.3508], std=[0.3077,0.2701,0.2591]),
    torchvision.transforms.RandomErasing(p=0.25, scale=(0.02, 0.2), ratio=(0.3, 3.3), value=0, inplace=False)
])
```

Normalization was used for self-resnetSE but not for self-Resnet. RandomErasing was found to be particularly helpful.

## Hyperparameters and Training
The optimizer used was RADAM, which provided more robust convergence than ADAM. CutMix improved the self-resnet model from 87% to 90%, and it is believed that a combination of CutMix and MixUp could further enhance model robustness.

A key learning was the use of dropout before the fully connected layer, which helped the model to not rely on specific neurons for prediction. The optimal dropout probability was found to be 0.49.

ArcFace contrastive loss was used to fine-tune the self-resnetSE model, improving it from 90.07% to 90.75% accuracy in less than 30 epochs. The chosen hyperparameters for ArcFace were a margin of 0.5 and a scaler of 72, which was aggressive but effective.

## Further Improvements

### Potential Enhancements
1. **Extended Training**: Increasing the number of epochs beyond 30 could potentially improve the accuracy of the models, especially when fine-tuning with contrastive losses.
2. **Hyperparameter Tuning**: Further experimentation with dropout rates, learning rates, and other hyperparameters could yield better performance.
3. **Data Augmentation**: Exploring additional data augmentation techniques like MixUp or more aggressive forms of CutMix could enhance the model's generalization capabilities.
4. **Ensemble Methods**: Combining predictions from multiple models or different versions of the same model could improve accuracy and robustness.
5. **Advanced Architectures**: Investigating newer or more complex architectures like ConvNeXt or Transformers designed for vision tasks could lead to better feature extraction and performance.
6. **Regularization Techniques**: Implementing other regularization methods such as label smoothing or batch normalization adjustments might provide benefits.
7. **Loss Function Exploration**: Trying out other contrastive or triplet loss functions could offer insights into better embedding spaces for facial recognition.

### Wandb Logs
The training process and results can be further explored through the Weights & Biases logs provided below:

- **self-resnet**: The training logs for the self-resnet model can be found at [Wandb - self-resnet](https://wandb.ai/11785_orangeli/hw2p2-ablations-early?workspace=user-zechengl).

- **self-resnet-se**: The training logs for the self-resnetSE model, especially the runs with center loss, can be viewed at [Wandb - self-resnet-se](https://wandb.ai/idlf23-22/hw2p2-ablations?workspace=user-zechengl), mainly under the run titled 'resnet_se_center'.

These logs provide a detailed account of the training process, including the hyperparameters used, the epochs, and the performance metrics for each run. They serve as a valuable resource for understanding the model's behavior during training and for identifying potential areas for improvement.

## Conclusion
The project demonstrated the effectiveness of data augmentation, the RADAM optimizer, and contrastive loss functions in improving facial recognition model performance. The use of dropout and droppath techniques also contributed to preventing overfitting and promoting model robustness.

For detailed logs and results, refer to the 'apply arcface' session in the notebook.