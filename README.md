# Introduction to TensorFlow-TensorRT (TF-TRT)

## Overview

This is a project-based course on optimizing [TensorFlow (TF)](https://www.tensorflow.org/) models for deployment using [TensorRT](https://developer.nvidia.com/tensorrt).

- **Instructor**: Snehan Kekre  
- **Certificate**: Awarded upon completion  
- **Duration**: ~<2 hours  

## Course Objectives

By the end of this course, you will achieve the following objectives:

- Optimize TensorFlow models using [TensorRT (TF-TRT)](https://docs.nvidia.com/deeplearning/frameworks/tf-trt-user-guide/index.html).  
- Optimize deep learning models at FP32, FP16, and INT8 precision using TF-TRT.  
- Analyze how tuning TF-TRT parameters impacts `performance` and `inference throughput`.  

## Course Outline

This course is divided into three parts:

1. **Course Overview**: Introductory reading material.  
2. **Optimize TensorFlow Models for Deployment with TensorRT**: A hands-on project.  
3. **Graded Quiz**: A final assignment required to successfully complete the course.  

## About this Project

This hands-on project guides you in optimizing [TensorFlow (TF)](https://www.tensorflow.org/) models for inference with NVIDIA's [TensorRT (TRT)](https://developer.nvidia.com/tensorrt).

By the end of this project, you will:  
- Optimize TensorFlow models using TensorRT (TF-TRT).  
- Work with models at `FP32`, `FP16`, and `INT8` precision, observing how TF-TRT parameters affect performance and inference throughput.

### Prerequisites  

To complete this project successfully, you should have:  
- Competency in Python programming.  
- An understanding of deep learning concepts and inference.  
- Experience building deep learning models using TensorFlow and its Keras API.  

## Project Structure  

| Task   | Description                                  |
|--------|----------------------------------------------|
| Task 1 | Introduction and Project Overview           |
| Task 2 | Set up TensorFlow and TensorRT Runtime      |
| Task 3 | Load Data and Pre-trained InceptionV3 Model |
| Task 4 | Create Batched Input                        |
| Task 5 | Load the TensorFlow SavedModel              |
| Task 6 | Benchmark Prediction Throughput and Accuracy |
| Task 7 | Convert TensorFlow SavedModel to TF-TRT Float32 Graph |
| Task 8 | Benchmark TF-TRT Float32                   |
| Task 9 | Convert to TF-TRT Float16 and Benchmark     |
| Task 10 | Work with TF-TRT INT8 Models               |
| Task 11 | Convert to TF-TRT INT8                     |

## Lab: Notebook  

| Description                    | Notebook                                                                                                               | Demo         |
|--------------------------------|-----------------------------------------------------------------------------------------------------------------------|--------------|
| Intro to TensorFlow-TensorRT   | [![Open notebook in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/afondiel/Computer-Vision-Kaggle-Free-Course/blob/main/lab/notebooks/Intro-to-TensorFlow-TensorRT.ipynb) | HF/Gradio Space |

## References  

### Courses
- [Main Course - Coursera](https://www.coursera.org/projects/tensorflow-tensorrt).  
- [Deep Learning Optimization and Deployment Using TensorFlow and TensorRT - NVIDIA DLI](https://learn.nvidia.com/courses/course-detail?course_id=course-v1:DLI+L-FX-18+V2).  

### Videos
- [NVAITC Webinar: Deploying Models with TensorRT](https://www.youtube.com/watch?v=67ev-6Xn30U).  
- [NVAITC AI Webinar Series Playlist](https://www.youtube.com/watch?v=azLCUayJJoQ&list=PL5B692fm6--sJLzBmCpUSpP36xUWwuO8c&index=1).  

### Documentation
- [TensorFlow (TF)](https://www.tensorflow.org/).  
- [LiteRT (Lite Runtime)](https://ai.google.dev/edge/litert).  
- [NVIDIA TensorRT (Ecosystem)](https://developer.nvidia.com/tensorrt).  
    - [TensorRT - Getting Started](https://developer.nvidia.com/tensorrt-getting-started).  
    - [TensorRT - Quick Start Guide](https://docs.nvidia.com/deeplearning/tensorrt/quick-start-guide/index.html).  
- [TensorFlow-TensorRT (TF-TRT)](https://docs.nvidia.com/deeplearning/frameworks/tf-trt-user-guide/index.html).  
    - [TF-TRT Getting Started Video](https://www.youtube.com/watch?v=w7871kMiAs8).  
    - [Increase Inference Performance with TensorFlow-TensorRT](https://blog.tensorflow.org/2018/04/speed-up-tensorflow-inference-on-gpus-tensorRT.html). 

### **Deep Learning Model Optimization**

- Core concepts: 
    - [Efficient Deep Learning: A Comprehensive Overview of Optimization Techniques 👐 📚 - Hugging Face](https://huggingface.co/blog/Isayoften/optimization-rush)     

- Quantization in Signal Processing:
    - [Quantization Signal Processing - Wikipedia](https://en.wikipedia.org/wiki/Quantization_(signal_processing))   
            
- Computer Arithmetic:
    - Core: 
        - [Number Representation and Computer Arithmetic](https://web.ece.ucsb.edu/Faculty/Parhami/pubs_folder/parh02-arith-encycl-infosys.pdf)
    - Data types & conversion:  
        - [Decimal](https://en.wikipedia.org/wiki/Decimal)
        - [Floating-point Arithmetic](https://en.wikipedia.org/wiki/Floating-point_arithmetic)
        - [Using Data in C++](https://runestone.academy/ns/books/published/cpp4python/AtomicData/AtomicData.html)
        - [Data Type Ranges - C++](https://learn.microsoft.com/en-us/cpp/cpp/data-type-ranges?view=msvc-170)
        - [Standard Conversion - C++](https://learn.microsoft.com/en-us/cpp/cpp/standard-conversions?view=msvc-170) 
        - [Number System and Base Conversions - GeeksforGeeks](https://www.geeksforgeeks.org/number-system-and-base-conversions/)
    - Maths & Algebra: 
        - Tensor (Maths): https://en.wikipedia.org/wiki/Tensor_(intrinsic_definition)
        - Matrix Multiplication: https://en.wikipedia.org/wiki/Matrix_multiplication
    - ML Tensor: https://en.wikipedia.org/wiki/Tensor_(machine_learning)

### Additional Resources  

- **Model Zoo**: [Edge AI Model Zoo](https://github.com/afondiel/EdgeAI-Model-Zoo).  
- **Blogs**:  
    - [2019 June 13] [High-Performance Inference with TensorRT Integration](https://blog.tensorflow.org/2019/06/high-performance-inference-with-TensorRT.html).  
    - [2019 June 03] [High performance inference with TensorRT Integration - TensorFlow Medium](https://medium.com/tensorflow/high-performance-inference-with-tensorrt-integration-c4d78795fbfe)
    - [2018 April 18] [Speed Up TensorFlow Inference on GPUs with TensorRT](https://blog.tensorflow.org/2018/04/speed-up-tensorflow-inference-on-gpus-tensorRT.html). 
    - [2017 April 08] [Advanced Spark and TensorFlow Meetup 2017-05-06 Reduced Precision (FP16, INT8) Inference on Convolutional Neural Networks with TensorRT and NVIDIA Pascal from Chris Gottbrath, Nvidia](https://www.slideshare.net/slideshow/advanced-spark-and-tensorflow-meetup-20170506-reduced-precision-fp16-int8-inference-on-convolutional-neural-networks-with-tensorrt-and-nvidia-pascal-from-chris-gottbrath-nvidia/74738726)



### "It's hardware that makes a machine fast. It's software that makes a fast machine slow." - [Craig Bruce](http://www.csbruce.com/)