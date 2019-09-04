dataFolder = 'TrainData';
categories = {'cat', 'dog'};
imds = imageDatastore(fullfile(dataFolder, categories), 'LabelSource', 'foldernames');
imds.ReadFcn = @(filename)readAndPreprocessImage(filename);
layers = [
    imageInputLayer([227 227 3])
    
    convolution2dLayer(3,8,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,16,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,32,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    fullyConnectedLayer(2)
    softmaxLayer
    classificationLayer];
%%

options = trainingOptions('sgdm','MaxEpochs',70,'InitialLearnRate',0.001,'MiniBatchSize',64,'Momentum',0.9);
convnet = trainNetwork(imds,layers,options);

testData = imageDatastore('TestData','IncludeSubFolders',true,'LabelSource', 'foldernames');
testData.ReadFcn = @(filename)readAndPreprocessImage(filename);
YTest = classify(convnet,testData);
TTest = testData.Labels;
accuracy = sum(YTest == TTest)/numel(TTest);
plotconfusion(TTest,YTest)


%% Train
testImg = readimage(imds,15);
classify(convnet,testImg)
imshow(testImg);

%% Test
testImg = readimage(testData,10);
classify(convnet,testImg)
imshow(testImg);



