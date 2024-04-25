# trafficFlowVis
This repository is used for the final project of NYU Visualization for Machine Learning course.

# 5/2 presentation #
-  same section as the final paper
    -  intro
    - related work
    - methodology
    - evaluation
    - explainability
    - future work
    - interactive demo
    - pre-recorded video
    - Q&A

# workflow #
-  each step should preview the video to select the most informative episode

- classes: bicycle, scooter, truck, car, human.  
## select chase 2 sensor 4 right as the main video ##
- Spatial
    - time-best: 3:45-4:06 
    - smaller frame step size, finer time period
    - only capture the number of classes
    - visualize via time/classes distribution
    - refer to MoReVis

- Temporal
    - time-best: 3:45-4:06 
    - larger frame step size, coarser time period
    - matching the bounding box by distance in two sequential frame
    - visualize by the time/ distance with camera. 

# TODO-let's say "Maybe" # 
- Explainability
    - training YOLO and explain by CNN, get some insights on model's decision.
