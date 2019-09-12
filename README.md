General solution of table detection task with Faster R-CNN using Tensorflow Object Detection API
## Installation
Mind following [API documentation](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/install.html)
## Usage
1. Prepare data via creating PASCAL VOC format dataset in _data_dir/experiment_  
(mind using parsers from _utils_ to extract xml data)
2. Extract TFRecords from your data using _create_tf_record.py_
3. Configure your pipeline config in _data_dir/experiment_ folder
4. Train network via using _train.ipynb_
5. Execute inference with _infer.ipynb_ and evaluate obtained results using _metrics.ipynb_
