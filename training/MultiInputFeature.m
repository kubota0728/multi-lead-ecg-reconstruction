%% *Prediction ECG*
%% *Load data*
% Lead index reference:
% 1–12 correspond to the following: 
% I1, I2, I3, aVr, aVl, aVf, V1, V2, V3, V4, V5, V6

cd('your_directory_here');   % Set working directory

inputIndexesStr = inputdlg('Input Indexes', 'Input Indexes');
outputIndexesStr = inputdlg('Output Indexes', 'Output Indexes');

selectedIndexes_input = str2num(inputIndexesStr{1});
selectedIndexes_output = str2num(outputIndexesStr{1});

FolderName_onehot = {'VerticularRate','QRSDuration',...
                     'RAxis','TAxis','QRSCount'};  

[selectedIndices, ok] = listdlg( ...
                        'PromptString', 'Select one or more folders:', ...
                        'SelectionMode', 'multiple', ...
                        'ListString', FolderName_onehot, ...
                        'Name', 'Folder Selection', ...
                        'ListSize', [200, 150]);

if ok && ~isempty(selectedIndices)
    selectedFolders = FolderName_onehot(selectedIndices);

    % Display selected folder names
    disp('Selected folders:');
    disp(selectedFolders)
else
    errordlg('Please select at least one folder.', 'Error');
end

numSelected = length(selectedFolders);

names = {'I1', 'I2', 'I3', 'aVr', 'aVl', 'aVf', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6'};

%% Load input signals
data.input = struct();
for i = 1:length(selectedIndexes_input)
    selectedIndex = selectedIndexes_input(i);
    
    if selectedIndex >= 1 && selectedIndex <= 12
        selectedName = names{selectedIndex};
        filePath = fullfile('matData', [selectedName, '.mat']);

        if exist(filePath, 'file')
            loadedData = load(filePath);
            data.input.(selectedName) = loadedData.(selectedName);
        end
    end
end

%% Load output signals
data.output = struct();
for i = 1:length(selectedIndexes_output)
    selectedIndex = selectedIndexes_output(i);

    if selectedIndex >= 1 && selectedIndex <= 12
        selectedName = names{selectedIndex};
        filePath = fullfile('matData', [selectedName, '.mat']);

        if exist(filePath, 'file')
            loadedData = load(filePath);
            data.output.(selectedName) = loadedData.(selectedName);
        end
    end
end

%% Load one-hot feature data
data.onehot = struct();
for i = 1:numSelected
    filePath = fullfile(['matData\', selectedFolders{i}, '\onehot.mat']);
    if exist(filePath, 'file')
        onehot_temp = load(filePath);
        data.onehot.(selectedFolders{i}) = onehot_temp;
    else
        disp('Folder does not exist.')
        break
    end
end

%% Transpose input/output signals
inputFields = fieldnames(data.input);
for i = 1:length(inputFields)
    fieldName = inputFields{i};
    fieldData = data.input.(fieldName);
    processed.input.(fieldName) = cellfun(@transpose, fieldData, 'UniformOutput', false);
end

onehotFields = fieldnames(data.onehot);
for i = 1:length(onehotFields)
    fieldName = onehotFields{i};
    fieldData = data.onehot.(fieldName).onehot;
    processed.onehot.(fieldName) = fieldData;
end

outputFields = fieldnames(data.output);
for i = 1:length(outputFields)
    fieldName = outputFields{i};
    fieldData = data.output.(fieldName);
    processed.output.(fieldName) = cellfun(@transpose, fieldData, 'UniformOutput', false);
end

%% Concatenate multiple leads into a single input matrix
tempo_inputData = cell(length(processed.output.(fieldName)),1);
inputData = processed.input.(names{selectedIndexes_input(1)});

for i = 1:length(processed.output.(fieldName))
    for j = 1:length(selectedIndexes_input)-1
        tempo_inputData{i} = vertcat(inputData{i}, processed.input.(names{selectedIndexes_input(j+1)}){i});
        inputData{i} = tempo_inputData{i};
    end
end

%% Concatenate one-hot features
tempo_onehotData = cell(length(processed.onehot.(fieldName)),1);
onehotData = processed.onehot.(selectedFolders{1});

for i = 1:length(processed.onehot.(fieldName))
    for j = 1:numSelected-1
        tempo_onehotData{i} = vertcat(onehotData{i}, processed.onehot.(selectedFolders{j+1}){i});
        onehotData{i} = tempo_onehotData{i};
    end
end

%% Concatenate output leads
tempo_outputData = cell(length(processed.output.(fieldName)),1);
outputData = processed.output.(names{selectedIndexes_output(1)});

for i = 1:length(processed.output.(fieldName))
    for j = 1:length(selectedIndexes_output)-1
        tempo_outputData{i} = vertcat(outputData{i}, processed.output.(names{selectedIndexes_output(j+1)}){i});
        outputData{i} = tempo_outputData{i};
    end
end

%% Dataset split: train/validation/test
prompt1 = {'Seed value:', 'Train ratio:', 'Validation ratio:'};
dlgtitle1 = 'Dataset';
dims = [1 35];
defaultans1 = {'1','0.7','0.15'};
ans_ratio = inputdlg(prompt1, dlgtitle1, dims, defaultans1);

seedValue = str2double(ans_ratio{1});
trainRatio = str2double(ans_ratio{2});
valRatio = str2double(ans_ratio{3});
testRatio = 1 - trainRatio - valRatio;

rng(seedValue);

numSamples = length(inputData);
numTrain = floor(trainRatio * numSamples);
numVal = floor(valRatio * numSamples);

idx = randperm(numSamples);
trainIdx = idx(1:numTrain);
valIdx = idx(numTrain+1:numTrain+numVal);
testIdx = idx(numTrain+numVal+1:end);

trainDataIN = inputData(trainIdx);
trainDataOUT = outputData(trainIdx);

valDataIN = inputData(valIdx);
valDataOUT = outputData(valIdx);

testDataIN = inputData(testIdx);
testDataOUT = outputData(testIdx);

trainData_one = onehotData(trainIdx);
testData_one = onehotData(testIdx);
valData_one = onehotData(valIdx);

%% Expand one-hot vectors to match sequence length
trainData_one = cellfun(@(x) repmat(x,1,512), trainData_one, 'UniformOutput', false);
trainecgDS = arrayDatastore(trainDataIN, 'OutputType', 'same');
trainfeatDS = arrayDatastore(trainData_one, 'OutputType', 'same');
traininputDS = combine(trainecgDS, trainfeatDS);
trainoutputDS = arrayDatastore(trainDataOUT, 'OutputType', 'same');
trainDS = combine(traininputDS, trainoutputDS);

valData_one = cellfun(@(x) repmat(x,1,512), valData_one, 'UniformOutput', false);
valecgDS = arrayDatastore(valDataIN, 'OutputType', 'same');
valfeatDS = arrayDatastore(valData_one, 'OutputType', 'same');
valinputDS = combine(valecgDS, valfeatDS);
valoutputDS = arrayDatastore(valDataOUT, 'OutputType', 'same');
valDS = combine(valinputDS, valoutputDS);

testData_one = cellfun(@(x) repmat(x,1,512), testData_one, 'UniformOutput', false);

%% Layer settings
prompt2 = {'CNN_ecg: filter size, num of filters', ...
           'CNN_com: filter size, num of filters', ...
           'Bi-LSTM units', ...
           'Dropout ratio:'};
dlgtitle2 = 'Layers';
defaultans2 = {'7,128','3,8','128','0.3'};
layerset = inputdlg(prompt2, dlgtitle2, dims, defaultans2);

CNNpara_ecg = strsplit(layerset{1}, ',');
CNNpara_com = strsplit(layerset{2}, ',');

inputSize = length(selectedIndexes_input);
length_ecg = size(trainDataIN{1},2);
numClasses_ecg = length(selectedIndexes_output);
numClasses_onehot = size(onehotData{1},1);

filterSize_ecg = str2double(CNNpara_ecg(1));
numFilters_ecg = str2double(CNNpara_ecg(2));
filterSize_com = str2double(CNNpara_com(1));
numFilters_com = str2double(CNNpara_com(2));

%% Hyperparameters
prompt3 = {'Epoch', 'Batch size', 'Learning rate', ...
           'L2 regularization (lambda)', 'Validation frequency'};
dlgtitle3 = 'Hyperparameter';
defaultans3 = {'150','128','0.001','0.001','20'};
hypara = inputdlg(prompt3, dlgtitle3, dims, defaultans3);

epoch = str2double(hypara{1});
batchsize = str2double(hypara{2});
lr = str2double(hypara{3});
lamda = str2double(hypara{4});
vf = str2double(hypara{5});

%% ----- Network Architecture -----

% ECG input branch
sequenceInput = sequenceInputLayer(inputSize, 'Name', 'ecg_input');
featureInput = sequenceInputLayer(numClasses_onehot, 'Name', 'label_input');

% ECG branch
ecgBranch = [
    sequenceInput
    convolution1dLayer(filterSize_ecg, numFilters_ecg, 'Padding', 'same')
    reluLayer('Name', 'relu_ecg')
    bilstmLayer(numHiddenUnits, 'OutputMode', 'sequence', 'Name', 'bilstm_ecg') % summarize sequence
    dropoutLayer(0.2, 'Name', 'dropout_ecg')
];

% One-hot feature branch
labelBranch = [
    featureInput
    fullyConnectedLayer(256, 'Name', 'fc0')
    reluLayer('Name', 'relu_feat0')
    dropoutLayer(0.1, 'Name', 'dropout_label0')

    fullyConnectedLayer(128, 'Name', 'fc1')
    reluLayer('Name', 'relu_feat1')
    dropoutLayer(0.1, 'Name', 'dropout_label1')

    fullyConnectedLayer(64, 'Name', 'fc2')
    reluLayer('Name', 'relu_feat2')
    dropoutLayer(0.1, 'Name', 'dropout_label2')

    fullyConnectedLayer(32, 'Name', 'fc2_2')
    reluLayer('Name', 'relu_feat2_2')
    dropoutLayer(0.1, 'Name', 'dropout_label2_2')

    fullyConnectedLayer(16, 'Name', 'fc2_3')
    reluLayer('Name', 'relu_feat2_3')
    dropoutLayer(0.1, 'Name', 'dropout_label2_3')
];

% Combined final layers
finalLayers = [
    concatenationLayer(1, 2, 'Name', 'Concat')
    dropoutLayer(0.3, 'Name', 'dropout_combined')
    convolution1dLayer(filterSize_com, numFilters_com, 'Padding', 'same')
    fullyConnectedLayer(numClasses_ecg, 'Name', 'output')  % regression output
];

% Build entire graph
layers = layerGraph();
layers = addLayers(layers, ecgBranch);
layers = addLayers(layers, labelBranch);
layers = addLayers(layers, finalLayers);

% Connect branches
layers = connectLayers(layers, 'dropout_ecg', 'Concat/in1');
layers = connectLayers(layers, 'dropout_label2_3', 'Concat/in2');

%% Training options
options = trainingOptions('adam', ...
    'MaxEpochs', epoch, ...
    'MiniBatchSize', batchsize, ...
    'InitialLearnRate', lr, ...
    'Shuffle', 'every-epoch', ...
    'Verbose', true, ...
    'ValidationFrequency', vf, ...
    'ValidationData', valDS, ...
    'Plots', 'training-progress', ...
    'ExecutionEnvironment', 'auto', ...
    'InputDataFormats', ["CTB","CTB"], ... % Required for sequence-to-sequence
    'TargetDataFormats', "CTB", ...
    OutputNetwork="best-validation");

%% Train the network
dlnet = dlnetwork(layers);
[net, info] = trainnet(trainDS, dlnet, 'mse', options);
