# Effects of random weights in deep architectures

Building on the findings from - [On Random Weights and Unsupervised Feature Learning](http://www.robotics.stanford.edu/~ang/papers/nipsdlufl10-RandomWeights.pdf), this experiment focuses on understanding the effects of multiple randomly initialized layers in a convolutional neural network. To be more specific,

For a given architecture X with n layers, the following procedure is used for training it :
* Train n layers of X normally (i.e. train X)
* Randomize layer 1 (lowest layer)
  * Train the top n-1 layers
* Randomize layer 1, 2 
  * Train the top n-2 layer
* Randomize layer 1, 2, ..,i
  * Train the top n-i layers 
  
In these experiments, the LeNet architecture is used on the MNIST dataset.
![Image of LeNet](/images/lenet.png)

For further details and results, please refer to the documentation [here](https://docs.google.com/document/d/1aB84TNnqIferAaw2JsLEp2fTc1ilPJdweg3Q_5Jmr0U/edit?usp=sharing).

