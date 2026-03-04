# **QIE-Object-Remover-Bbox**

QIE-Object-Remover-Bbox is an advanced, AI-powered image editing application specifically designed to perform precise object removal and background inpainting based on user-defined bounding box coordinates. By leveraging sophisticated vision-language models—likely from the Qwen family, as indicated by the project structure—the tool allows users to accurately isolate unwanted elements within an image. Once a target region is designated, the underlying model seamlessly removes the object and intelligently fills the void with contextually appropriate textures and pixels, ensuring that the final image looks natural and undisturbed. The application is built entirely in Python and features a user-friendly Gradio web interface, making high-precision image manipulation accessible without requiring complex manual editing skills.

## Features

* **BBox-Based Target Identification:** Accurately specify the exact objects to be removed by providing bounding box coordinates.
* **Seamless Background Inpainting:** Automatically reconstructs and fills the removed area with context-aware textures that match the surrounding environment.
* **Interactive Web Interface:** Provides an intuitive, browser-based user interface powered by Gradio for easy image uploading and processing.
* **High-Precision Vision Logic:** Utilizes advanced Qwen-based vision processing to understand image context and segment objects cleanly.

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/PRITHIVSAKTHIUR/QIE-Object-Remover-Bbox.git
cd QIE-Object-Remover-Bbox
```

### 2. Install Pre-requirements

System-level or heavy dependencies should be installed first:

```bash
pip install -r pre-requirements.txt
```

### 3. Install Standard Dependencies

```bash
pip install -r requirements.txt
```

## How to Run

Start the application by executing the main script:

```bash
python app.py
```

Once the script is running, the terminal will provide a local URL (typically `http://127.0.0.1:7860`). Open this URL in your web browser to access the interactive Gradio interface.

## Project Structure

* `app.py`: The main entry point of the application containing the Gradio interface logic.
* `qwenimage/`: The core module housing the model processing and vision-language integration logic.
* `requirements.txt`: A list of standard Python dependencies required to run the application.
* `pre-requirements.txt`: Initial setup dependencies (such as PyTorch and torchvision) needed before standard requirements.
* `LICENSE.txt`: The licensing details for the repository.

## Workflow

1. Upload the target image via the web interface.
2. Input the bounding box coordinates to define the area containing the unwanted object.
3. The model processes the specified region, removes the object, performs inpainting, and returns the cleaned image.

## License

This project is open-source and licensed under the Apache License 2.0. Please refer to the `LICENSE.txt` file within the repository for full terms and conditions.

## Contributing

Contributions to the project are highly encouraged. You are welcome to submit a Pull Request or open an issue on GitHub to report bugs, suggest new features, or improve the existing codebase.