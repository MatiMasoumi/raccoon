Raccoon Detection with Faster R-CNN** This repository demonstrates how to detect raccoons in images using **Faster R-CNN**.
The project leverages a pre-trained ResNet-50 backbone and fine-tunes it on the **Raccoon Dataset**. The model achieves strong performance and visualizes detected bounding boxes on test images.
-- ## **Overview** This project includes: - Training the Faster R-CNN model on the **Raccoon Dataset**. 
- Evaluating the model using metrics like **mAP** and **mAR**. - Running inference on custom images and visualizing the results.
- ## **Features** - Fine-tunes a pre-trained **Faster R-CNN ResNet-50** model. - Includes custom preprocessing for the **Raccoon Dataset**.
 Provides evaluation metrics like **mean Average Precision (mAP)**. - Visualizes predictions with bounding boxes and scores
**Dataset** The **Raccoon Dataset** is sourced from [experiencor](https://github.com/experiencor/raccoon_dataset).
   -  It contains labeled images of raccoons with bounding boxes. To use the dataset: After training, the model is evaluated using **mAP** (mean Average Precision) and **mAR**
(mean Average Recall). Below are example metrics: - mAP: 0.17 - mAP@50: 0.24 - mAR (large objects): 0.75 - mAR (medium objects): 0.40 --- ##
**Results** The following is an example of the model's output on a test image, where raccoons are detected with bounding boxes: ![Sample Detection](path_to_sample_image.jpg)
--- ## **Project Structure** ```plaintext raccoon_detection/ ├── dataset/ # Contains the Raccoon Dataset ├── models/ # Contains the Faster R-CNN model setup
├── utils/ # Utility functions for preprocessing and visualization ├── train.py # Script for training the model ├── inference.py # Script for running inference
 ├── requirements.txt # List of dependencies └── README.md # Project documentation ``` --- ## **Credits** This project uses the following resources:
 - **Dataset**: [Raccoon Dataset](https://github.com/experiencor/raccoon_dataset) - **Framework**: [PyTorch](https://pytorch.org/) --- ## **License** This project is licensed under the **MIT License**.
 See the `LICENSE` file for details. 
