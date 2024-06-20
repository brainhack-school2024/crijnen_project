<a href="https://github.com/ccrijnen">
   <img src="https://avatars.githubusercontent.com/u/20086110?v=4" width="100px;" alt=""/>
   <br/><sub><b>Cornelius Crijnen</b></sub>
</a>

I am a PhD student in Psychology at Université de Montréal under the guidance of Prof. Shahab Bakhtiari. My research is at intersection of artificial intelligence and neuroscience, particularly focusing on modelling the visual system. I use artificial neural networks trained with self-supervised learning to simulate and understand the complexities of the visual system.

# Evaluating ANN models of the visual system with Representational Similarity Analysis
This project aims to evaluate the similarity between the representations of artificial neural networks (ANNs) and the visual system in the mouse brain using Representational Similarity Analysis (RSA).

## Project definition
### Main Question
How similar are the representations of ANNs (treeshrew and rat) to those in the mouse visual system?

### Background
* ANN models are often used to understand the visual system in the brain, specifically the ventral pathway.
* First, image classification models were shown to predict neural responses in the ventral pathway (Yamins et al., 2014. https://doi.org/10.1073/pnas.1403112111).
* Similarly, the representations learned by self-supervised models produce good matches to the ventral pathway (Talia Konkle, George Alvarez, 2020. https://doi.org/10.1167/jov.20.11.498).
* Recently, it was shown that it is not only possible to predict neural responses in the ventral pathway but also in the dorsal pathway with a model that has two parallel pathways (Bakhtiari et al., 2021. https://doi.org/10.1101/2021.06.18.448989).
* We trained self-supervised models on videos from treeshrew and rat's point of view with two pathways.
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
* Generate response matrices from visual areas in the mouse brain
  1. Download calcium recordings of cell activities with AllenSDK
  2. Find the responses to each image / video in the dataset
  3. Save response matrices
* Generate response matrices of my ANNs from video stimuli
  1. Load the trained models
  2. Feed stimuli (images/videos) to ANN
  3. Save response matrices for each layer in the ANN
* Use rsatoolbox to calculate RDMs of all response matrices
* Calculate similarity between RDMs (using rsatoolbox)
* Visualize the results
  * Plots of the similarity between RDMs for different brain areas and ANN layers

### Dataset
* Our collaborators videos from treeshrew and rat's point of view, which were used to train the ANN models of the visual system using SSL.
* Calcium imaging data from the Allen Brain Observatory (https://observatory.brain-map.org/visualcoding/).

### Tools
* AllenSDK for mouse brain data
* PyTorch to retrieve the representations of the ANNs
* rsatoolbox for the RSA analysis
* Plotly, Matplotlib and Seaborn for visualizations

### Deliverables
* This GitHub repository containing the code and model checkpoints to reproduce my results.
* A jupyter notebook of the RSA: [rsa notebook](notebooks/analysis.ipynb)
* A jupyter notebook containing the visualisations of my analysis: [rsa notebook](notebooks/results.ipynb)

## Results

### Visualizing the Representational Similarity Analysis

The following figures show the results of the RSA between the visual system of the mouse brain and the ANNs of the visual system trained on treeshrew and rat videos. 
Each figure shows the similarity between the representations of the visual system and the ANNs for five different brain areas. 
The noise ceiling is shown in grey, the similarity between the representations of the visual system and the rat ANNs is shown in blue, and the similarity between the representations of the visual system and the treeshrew ANNs is shown in red. 
Since the ANN models have two parallel pathways, the similarity between the representations of the visual system and the two pathways of the ANNs are shown separately.
Two stimuli were used: Natural Movie One and Natural Scenes. For each stimulus, I used two depths of mouse brain recordings from the allen brain observatory: 175 and 275, respectively.

#### Natural Movie One Stimuli

![depth 175](nm_175.png)
![depth 275](nm_275.png)

#### Natural Scenes Stimuli
![depth 175](ns_175.png)
![depth 275](ns_275.png)

## Conclusion and acknowledgement

The results show that the representations of the visual system in the mouse brain are more similar to the representations of the ANNs trained on rat videos than to the ones trained on treeshrew videos. 

I would like to thank the brainhack school for providing me with the opportunity to work on this project. I would also like to thank my collaborators for providing me with the data and the guidance to complete this project.

## Collaborators
* Shahab Bakhtiari
* Madineh Sedigh-Sarvestani

## References

* [Yamins et al., 2014. Performance-optimized hierarchical models predict neural responses in higher visual cortex](https://doi.org/10.1073/pnas.1403112111)
* [Talia Konkle, George Alvarez, 2020. Deepnets do not need category supervision to predict visual system responses to objects](https://doi.org/10.1167/jov.20.11.498)
* [Bakhtiari et al., 2021. The functional specialization of visual cortex emerges from training parallel pathways with self-supervised predictive learning](https://doi.org/10.1101/2021.06.18.448989)
