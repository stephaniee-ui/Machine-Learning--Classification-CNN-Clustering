## Classification/Clustering of AI-Generated vs. Human-Generated Images
This project applies CNN (ResNet-50), Vision Transformer (ViT), and PCA + GMM clustering to classify or analyze AI vs. human-generated images.

```plaintext
.
├── cnn/                              # CNN model (ResNet-50)
│   ├── CNN.py                        # Model, training, evaluation, checkpointing
│   ├── CNN_training.ipynb            # CNN training (baseline)
│   ├── CNN_training_aug.ipynb        # CNN training with augmentation
│   ├── CNN_evaluation.ipynb          # Evaluation with metrics and plots
│   └── CNN_evaluation_aug.ipynb      # Evaluation for augmented model
│
├── vit/                              # Vision Transformer (ViT)
│   ├── ViT.py                        # ViT model definition
│   ├── ViT_training.ipynb            # ViT training
│   ├── ViT_evaluation.ipynb          # ViT evaluation
│   ├── ViT_training_aug.ipynb        # ViT training with augmentation
│   └── ViT_evaluation_aug.ipynb      # ViT evaluation with augmentation
│
├── unsupervised/                     # Unsupervised clustering
│   └── Unsupervised_PCA_GMM.ipynb    # PCA + GMM clustering analysis
│
├── image.py                          # ImageDataset loader from CSV
├── util.py                           # Logging, plot styling, checksum
├── get_model_params.sh               # Copies models/CSVs from scratch directory
│           
├── backup/                           # Folder to include the outdated code
│   └── ...
└── README.md                         # Project overview (this file)
