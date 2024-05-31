<a href="https://github.com/ccrijnen">
   <img src="https://avatars.githubusercontent.com/u/20086110?v=4" width="100px;" alt=""/>
   <br/><sub><b>Cornelius Crijnen</b></sub>
</a>

I am a PhD student in Psychology at Université de Montréal under the guidance of Prof. Shahab Bakhtiari. My research is at intersection of artificial intelligence and neuroscience, particularly focusing on modelling the visual system. I use artificial neural networks trained with self-supervised learning to simulate and understand the complexities of the visual system.

# Evaluating ANN models of the visual system with Representational Similarity Analysis
This project aims to evaluate the similarity between the representations of artificial neural networks (ANNs) and the visual system in the mouse brain using Representational Similarity Analysis (RSA).

## Project definition
### Main Question
* How similar are the representations of ANNs trained on treeshrew and rat's pov videos to each other?
* How similar are the representations of ANNs (treeshrew and rat) to those in the mouse visual system?

### Background
* ANN models are often used to understand the visual system in the brain, specifically the ventral pathway.
* First, image classification models were shown to predict neural responses in the ventral pathway (Yamins et al., 2014. https://doi.org/10.1073/pnas.1403112111).
* Similarly, the representations learned by self-supervised models produce good matches to the ventral pathway (Talia Konkle, George Alvarez, 2020. https://doi.org/10.1167/jov.20.11.498).
* We trained self-supervised models on videos from treeshrew and rat's point of view.
* RSA is used to compare two representational spaces:
  * Response matrices are created for every brain area and every layer of the ANNs, where each element represents the response of a neuron to a video sequence.
  * Use Pearson correlation to calculate the similarity of every pair of columns in the response matrix, forming a Representation Similarity Matrix (RSM).
  * The RSMs describe the representation space in a network, either a brain area or an ANN layer.
  * Kendall’s τ is used between the vectorized RSMs to quantify the similarity between the two representations.

### Objectives
* Become familiar with a Allan Brain Observatory dataset containing calcium imaging data from the mouse brain.
* Learn how to do RSA between ANNs and mouse brain data.
* Visualize the results in a way that is easy to interpret.

## Methodology
* Generate response matrices of my ANNs from video stimuli
  * Load the trained models
  * Save activations at each layer for a short video (response matrix)
* Download response matrices from the AllenSDK
  * Bring the data into a format that can be used for RSA
* Feed them into rsatoolbox to calculate RSMs
* Calculate similarity between RSMs
* Visualize the results
  * Heatmaps of the RSMs
  * Plots of the similarity between RSMs

### Dataset
* Calcium imaging data from the Allen Brain Observatory (https://observatory.brain-map.org/visualcoding/).
* Our collaborators videos from treeshrew and rat's point of view.

### Tools
* AllenSDK for mouse brain data
* PyTorch to retrieve the representations of the ANNs
* RSA toolbox for the RSA analysis
* Plotly / Matplotlib / Seaborn for visualizations

## Deliverables
* A Github repository with documented code to reproduce your analyses and results.
* A jupyter notebook of the analysis codes and visualisations to display the results.

## Collaborators
* Shahab Bakhtiari, Madineh Sedigh-Sarvestani
