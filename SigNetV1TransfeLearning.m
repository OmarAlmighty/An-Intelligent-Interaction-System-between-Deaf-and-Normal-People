clc;clear;close all;
addpath('frames');
%%
images = imageDatastore('frames',...
    'IncludeSubfolders',true,...
    'LabelSource','foldernames');%,'ReadFcn',@(loc)imresize(imread(loc),[277 277]));
%%
[designImages,testImages] = splitEachLabel(images,0.75,'randomized');
[trainingImages,validationImages] = splitEachLabel(designImages,0.75,'randomized');

%% Load Pretrained Network
net = importKerasLayers('CovidNetDBV3-360KParam.h5');%alexnet();
%% Transfer Learning The DNN
layersTransfer = net(1:end-4);
numClasses = numel(categories(trainingImages.Labels))
layers = [
    layersTransfer
    fullyConnectedLayer(numClasses,'WeightLearnRateFactor',20,'BiasLearnRateFactor',20)
    softmaxLayer
    classificationLayer];
%% Define Training Parameters
miniBatchSize =32;
numIterationsPerEpoch = floor(numel(trainingImages.Labels)/miniBatchSize);
options = trainingOptions('rmsprop',...
    'MiniBatchSize',miniBatchSize,...
    'MaxEpochs',9,...
    'InitialLearnRate',1e-4,...
    'Verbose',true,...
    'Plots','training-progress',...
    'ValidationData',validationImages,...
    'ValidationFrequency',numIterationsPerEpoch);
%% Train The DNN Model
[SignNet,tr] = trainNetwork(trainingImages,layers,options);
%% Extract Image Features
layer = 'fc';
disp('Extracting Training Features')
trainingFeatures4D = activations(SignNet,trainingImages,layer);
validationFeatures4D = activations(SignNet,validationImages,layer);
disp('Extracting Test Features')
testFeatures4D = activations(SignNet,testImages,layer);
%

%%
% Extract the class labels from the training and test data.
trainingLabels = trainingImages.Labels;
validationLabels = validationImages.Labels;
testLabels = testImages.Labels;
trainingFeatures=reshape(trainingFeatures4D,[],length(trainingLabels));
validaionFeatures=reshape(validationFeatures4D,[],length(validationLabels));
testFeatures=reshape(testFeatures4D,[],length(testLabels));
disp('Model Trained and features were extracted')
%% Fit Image Classifier
% Use the features extracted from the training images as predictor
% variables and fit a multiclass support vector machine (SVM) using
% |fitcecoc| (Statistics and Machine Learning Toolbox).
% disp('Start trining Classifiers')
disp('Training SVM on Features')
classifier = fitcecoc(trainingFeatures',trainingLabels);
disp('Training KNN on Features')
classifierknn = fitcknn(trainingFeatures',trainingLabels);
disp('Training Decision Tree on Features')
classifierTree = fitctree(trainingFeatures',trainingLabels);
disp('Training Naive Bayseian on Features')
classifierNB = fitcnb(trainingFeatures',trainingLabels);

%% Classify Test Images
% Classify the test images using the trained SVM model the features
% extracted from the test images.
disp('Testing Trained Classifiers on Test Features')
predictedLabels = predict(classifier,testFeatures');
predictedLabelsDnn = classify(SignNet,testImages);
predictedLabelsknn = predict(classifierknn,testFeatures');
predictedLabelsTree = predict(classifierTree,testFeatures');
predictedLabelsNB = predict(classifierNB,testFeatures');

%%
% Display four sample test images with their predicted labels.
idx = [10 30 90 100];
figure
for i = 1:numel(idx)
    subplot(2,2,i)
    I = readimage(testImages,idx(i));
    label = predictedLabelsDnn(idx(i));
    imshow(I)
    title(char(label))
end
%
%%
% Calculate the classification accuracy on the test set. Accuracy is the
% fraction of labels that the network predicts correctly.
accuracySVM = mean(predictedLabels == testLabels)
accuracyDNN = mean(predictedLabelsDnn == testLabels)
accuracyKNN = mean(predictedLabelsknn == testLabels)
accuracyTree = mean(predictedLabelsTree == testLabels)
accuracyCNB = mean(predictedLabelsNB == testLabels)
%savefile='maryim.mat';
%save(savefile,'netTransfer','trainingFeatures','testFeatures','trainingLabels','testLabels');

%% x=grp2idx(trainingLabels);y=grp2idx(validationLabels);z=grp2idx(testLabels);
% dataLabels=[x;y;z];
% This SVM has high accuracy. If the accuracy is not high enough using
% feature extraction, then try transfer learning instead.