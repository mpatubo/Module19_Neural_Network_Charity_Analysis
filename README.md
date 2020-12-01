# Module19_Neural_Network_Charity_AnalysisOverview of the loan prediction risk analysis:



Overview of the analysis: 
The purpose of the analysis is to use machine learning and neural networks to crate a binary classifier capable of predicting whether applicants will be successl if they receive funding from Alphabet Soup.  More than 34,000 past recipients were used for data.  



Data Preprocessing
What variable(s) are considered the target(s) for your model?
y = application_df["IS_SUCCESSFUL"].values
 

What variable(s) are considered to be the features for your model?
Application categories which are:

'APPLICATION_TYPE',
 'AFFILIATION',
 'CLASSIFICATION',
 'USE_CASE',
 'ORGANIZATION',
 'INCOME_AMT',
 'SPECIAL_CONSIDERATIONS']

What variable(s) are neither targets nor features, and should be removed from the input data?
X = application_df.drop(["IS_SUCCESSFUL"],1).values

Compiling, Training, and Evaluating the Model
How many neurons, layers, and activation functions did you select for your neural network model, and why?

number_input_features = len(X_train[0])
hidden_nodes_layer1 =  80
hidden_nodes_layer2 = 40

Model structure summary:

_________________________________________________________________
Layer (type)                 Output Shape           Param #   
=================================================================
 Layer (type) 
dense_3 (Dense)              (None, 80)                8480      
_________________________________________________________________
dense_4 (Dense)              (None, 40)                3240      
_________________________________________________________________
Dense_5 (Dense)              (None, 1)                 41        
=================================================================
Total params: 11,761
Trainable params: 11,761
Non-trainable params: 0

First layer: Activation function = relu
Second layer: Activation function = relu
Outer layer: Activation function = sigmoid


Model evaluation:
268/268 - 0s - loss: 0.7676 - accuracy: 0.5391
Loss: 0.7676177024841309, Accuracy: 0.539125382900238

Were you able to achieve the target model performance? Yes.
The model correctly classifies all the training data, although model performance loss is .76 with accuracy of .53.   It is sufficient for our needs. Can't expect 100% accuracy. 


What steps did you take to try and increase model performance?
Add more neurons to a hidden layer.
Add additional hidden layers.
Use a different activation function for the hidden layers.
Add additional epochs to the training regimen.


Summary: Summarize the overall results of the deep learning model. Include a recommendation for how a different model could solve this classification problem, and explain your recommendation.

I recommend experimenting using the following other activation functions and see how different the results are and if the numbers are improvements of what you already have.  Although. It is best to identify (ahead of time) your threshholds and what you would consider success/failure or good/bad.   


1) The linear function 
2) The tanh function
3) The Leaky ReLU



