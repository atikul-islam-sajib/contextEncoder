# Context Encoders: Feature Learning by Inpainting

<img src="https://github.com/atikul-islam-sajib/contextEncoder/blob/main/research/artifacts/IMG_0336.jpg" alt="Context Encoder GAN">

This project provides a complete framework for training and testing a  Context Encoders: Feature Learning by Inpainting Generative Adversarial Network . It includes functionality for data preparation, model training, testing, and inference to enhance feature learning by Inpainting

<img src="https://miro.medium.com/v2/resize:fit:2000/1*xZ9-q0bGxhi18RAbiBykSQ.png" alt="AC-GAN - Medical Image Dataset Generator: Generated Image with labels">

## Features

| Feature                          | Description                                                                                                                                                                                                           |
| -------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Efficient Implementation**     | Utilizes an optimized Context Encoder model architecture for superior performance on diverse image segmentation tasks.                                                                                                          |
| **Custom Dataset Support**       | Features easy-to-use data loading utilities that seamlessly accommodate custom datasets, requiring minimal configuration.                                                                                             |
| **Training and Testing Scripts** | Provides streamlined scripts for both training and testing phases, simplifying the end-to-end workflow.                                                                                                               |
| **Visualization Tools**          | Equipped with tools for tracking training progress and visualizing segmentation outcomes, enabling clear insight into model effectiveness.                                                                            |
| **Custom Training via CLI**      | Offers a versatile command-line interface for personalized training configurations, enhancing flexibility in model training.                                                                                          |
| **Import Modules**               | Supports straightforward integration into various projects or workflows with well-documented Python modules, simplifying the adoption of Context Encoder functionality.                                                         |
| **Multi-Platform Support**       | Guarantees compatibility with various computational backends, including MPS for GPU acceleration on Apple devices, CPU, and CUDA for Nvidia GPU acceleration, ensuring adaptability across different hardware setups. |

## Getting Started

## Installation Instructions

Follow these steps to get the project set up on your local machine:

| Step | Instruction                                  | Command                                                       |
| ---- | -------------------------------------------- | ------------------------------------------------------------- |
| 1    | Clone this repository to your local machine. | **git clone https://github.com/atikul-islam-sajib/contextEncoder.git** |
| 2    | Navigate into the project directory.         | **cd contextEncoder**                                                  |
| 3    | Install the required Python packages.        | **pip install -r requirements.txt**                           |

## Project Structure

This project is thoughtfully organized to support the development, training, and evaluation of the srgan model efficiently. Below is a concise overview of the directory structure and their specific roles:

- **checkpoints/**
  - Stores model checkpoints during training for later resumption.
- **best_model/**

  - Contains the best-performing model checkpoints as determined by validation metrics.

- **train_models/**

  - Houses all model checkpoints generated throughout the training process.

- **data/**

  - **processed/**: Processed data ready for modeling, having undergone normalization, augmentation, or encoding.
  - **raw/**: Original, unmodified data serving as the baseline for all preprocessing.

- **logs/**

  - **Log** files for debugging and tracking model training progress.

- **metrics/**

  - Files related to model performance metrics for evaluation purposes.

- **outputs/**

  - **test_images/**: Images generated during the testing phase, including segmentation outputs.
  - **train_gif/**: GIFs compiled from training images showcasing the model's learning progress.
  - **train_images/**: Images generated during training for performance visualization.

- **research/**

  - **notebooks/**: Jupyter notebooks for research, experiments, and exploratory analyses conducted during the project.

- **src/**

  - Source code directory containing all custom modules, scripts, and utility functions for the U-Net model.

- **unittest/**
  - Unit tests ensuring code reliability, correctness, and functionality across various project components.

### Dataset Organization for ContextEncoder

The dataset is organized into three categories for ContextEncoder. Each category directly contains paired images and their corresponding images, stored together to simplify the association images .

## Directory Structure:

```
dataset/
├── X/
│ │ ├── 1.png
│ │ ├── 2.png
│ │ ├── ...
├── y/
│ │ ├── 1.png
│ │ ├── 2.png
│ │ ├── ...
```

### User Guide Notebook - CLI

For detailed documentation on the implementation and usage, visit the -> [contextEncoder Notebook - CLI](https://github.com/atikul-islam-sajib/ESRGAN/blob/main/research/notebooks/ESRGAN_CLI.ipynb).

### User Guide Notebook - Custom Modules

For detailed documentation on the implementation and usage, visit the -> [contextEncoder Notebook - CM](https://github.com/atikul-islam-sajib/ESRGAN/blob/main/research/notebooks/ESRGAN_CM.ipynb).

## Data Versioning with DVC
To ensure you have the correct version of the dataset and model artifacts.

Reproducing the Pipeline
To reproduce the entire pipeline and ensure consistency, use:

```bash
dvc repro
```

### Command Line Interface

The project is controlled via a command line interface (CLI) which allows for running different operational modes such as training, testing, and inference.

#### CLI Arguments
| Argument          | Description                                  | Type   | Default |
|-------------------|----------------------------------------------|--------|---------|
| `--image_path`    | Path to the image dataset                    | str    | None    |
| `--batch_size`    | Number of images per batch                   | int    | 1       |
| `--image_size`    | Size to resize images to                     | int    | 64      |
| `--epochs`        | Number of training epochs                    | int    | 100     |
| `--lr`            | Learning rate                                | float  | 0.0002  |
| `--lr_scheduler`| Enable learning rate scheduler              | bool   | False   |
| `--is_weight_init`| Apply weight initialization                  | bool   | False   |
| `--device`        | Computation device ('cuda', 'mps', 'cpu')    | str    | 'mps'   |
| `--adam`          | Use Adam optimizer                           | bool   | True    |
| `--SGD`           | Use Stochastic Gradient Descent optimizer    | bool   | False   |
| `--beta1`         | Beta1 parameter for Adam optimizer           | float  | 0.5     |
| `--train`         | Flag to initiate training mode               | action | N/A     |
| `--model`         | Path to a saved model for testing            | str    | None    |
| `--test`          | Flag to initiate testing mode                | action | N/A     |

### CLI Command Examples

| Task                     | CUDA Command                                                                                                              | MPS Command                                                                                                              | CPU Command                                                                                                              |
|--------------------------|---------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------|
| **Training a Model**     | `python cli.py --train --image_path "/path/to/dataset" --batch_size 32 --image_size 128 --epochs 50 --lr 0.001 --adam True --is_l1 True --device "cuda"` | `python cli.py --train --image_path "/path/to/dataset" --batch_size 32 --image_size 128 --epochs 50 --lr 0.001 --adam True --is_l1 True --device "mps"` | `python main.py --train --image_path "/path/to/dataset" --batch_size 32 --image_size 128 --epochs 50 --lr 0.001 --adam True --is_l1 True --device "cpu"` |
| **Testing a Model**      | `python cli.py --test --model "/path/to/saved_model.pth" --device "cuda"`                                              | `python cli.py --test --model "best" --device "mps"`                                              | `python cli.py --test --model "best" --device "cpu"`                                              |

### Notes:
- **CUDA Command**: For systems with NVIDIA GPUs, using the `cuda` device will leverage GPU acceleration.
- **CPU Command**: Suitable for systems without dedicated GPU support or for testing purposes on any machine.

**Configure the Project**:
   Update the `config.yml` file with the appropriate paths and settings. An example `config.yml`:
   ```yaml

        dataloader:
          image_path: "./data/raw/dataset.zip"
          image_size: 128            # Paper suggested to use 128, try to use different like 256, 512, 1024 as well
          channels: 3
          batch_size: 4              # Paper suggested to use 8, try to use different like 1, 16, 32, 64 as well
          split_size: 0.30

        trainer:
          name: "trainer"
          adam: True
          SGD: False
          lr: 0.0002
          beta1: 0.5                   # Used in the Adam optimizer
          beta2: 0.999                 # Used in the Adam optimizer
          momentum: 0.9                # Used for SGD optimizer
          weight_decay: 0.0001         # Used for Adam optimizer
          epochs: 5
          adversarial_lambda: 0.001    # Lmbda value added in the netG
          pixelwise_lambda: 0.999      # Lmbda value added in the netG
          steps: 100
          step_size: 10                # Used for learning rate scheduler
          gamma: 0.5                   # Used for learning rate scheduler
          device: mps                  # Can be used "cpu", "mps" as well
          lr_scheduler: False          # Used StepLR
          l1_regularization: False
          l2_regularization: False
          MLFlow: True                 # To visualize the training process in local - "mlflow ui"
          display: True                # Show the progress in each epoch
          is_weight_init: True

        tester:
          model: best                  # Define the model as "best" or select the model from chackpoints
          dataloader: train            # Can be "train", "valid" as well
          device: mps                  # Can be used "mps" or "cuda" as well

```


#### Initializing Data Loader - Custom Modules
```python
loader = Loader(image_path="path/to/dataset", batch_size=32, image_size=128)
loader.unzip_folder()
loader.create_dataloader()
```

##### To details about dataset
```python
loader.plot_images()           # It will display the images from dataset
```

#### Training the Model
```python
trainer = Trainer(
    epochs=100,                # Number of epochs to train the model
    lr=0.0002,                 # Learning rate for optimizer
    device='cuda',             # Computation device ('cuda', 'mps', 'cpu')
    adam=True,                 # Use Adam optimizer; set to False to use SGD if implemented
    SGD=False,                 # Use Stochastic Gradient Descent optimizer; typically False if Adam is True
    beta1=0.5,                 # Beta1 parameter for Adam optimizer
    lr_scheduler=False,        # Enable a learning rate scheduler
    weight_init=False,         # Enable custom weight initialization for the models
    display=True               # Display training progress and statistics

    ... ... ... 
    ... ... ...                # Check the trainer.py
)

# Start training
trainer.train()
```

#### Testing the Model
```python
tester = Tester(device="cuda", model="best") # use mps, cpu
test.plot()
```


### Configuration for MLFlow

1. **Generate a Personal Access Token on DagsHub**:
   - Log in to [DagsHub](https://dagshub.com).
   - Go to your user settings and generate a new personal access token under "Personal Access Tokens".

2. **Set environment variables**:
   Set the following environment variables with your DagsHub credentials:
   ```bash
   export MLFLOW_TRACKING_URI="https://dagshub.com/<username>/<repo_name>.mlflow"
   export MLFLOW_TRACKING_USERNAME="<your_dagshub_username>"
   export MLFLOW_TRACKING_PASSWORD="<your_dagshub_token>"
   ```

   Replace `<username>`, `<repo_name>`, `<your_dagshub_username>`, and `<your_dagshub_token>` with your actual DagsHub username, repository name, and personal access token.

### Running the Training Script

To start training and logging the experiments to DagsHub, run the following command:

```bash
python src/cli.py --train 
```

### Accessing Experiment Tracking

You can access the MLflow experiment tracking UI hosted on DagsHub using the following link:

[ESRGAN Experiment Tracking on DagsHub](https://dagshub.com/atikul-islam-sajib/contextEncoder/experiments/#/)

### Using MLflow UI Locally

If you prefer to run the MLflow UI locally, use the following command:

```bash
mlflow ui
```


## Contributing
Contributions to improve this implementation of context Encoder are welcome. Please follow the standard fork-branch-pull request workflow.

## License
Specify the license under which the project is made available (e.g., MIT License).


# will updated soon .....
