%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Collect historical data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

GSR = reshape(GSR, [1, numel(GSR)]);
PV = reshape(PV, [1, numel(PV)]);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Preprocess historical data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% training set percentage from the whole set
train_per = 0.9;

% obtains training set from the whole set
numtrainsamples = floor(train_per * numel(GSR));

% training inputs and targets
xtrain = GSR(1:numtrainsamples);
ytrain = PV(1:numtrainsamples);

% testing inputs and targets
xtest = GSR(numtrainsamples+1:end);
ytest = PV(numtrainsamples+1:end);

% input standardization
muGSR = mean(GSR);
sigGSR = std(GSR);

norm_xtrain = (xtrain - muGSR) / sigGSR;
norm_xtest = (xtest - muGSR) / sigGSR;

% input standardization
muPV = mean(PV);
sigPV = std(PV);

norm_ytrain = (ytrain - muPV) / sigPV;
norm_ytest = (ytest - muPV) / sigPV;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Build model: Fully-Connected Neural Network Architecture
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
num_input_features = 1;
num_output_targets = 1;

model = [
    sequenceInputLayer(num_input_features)
    fullyConnectedLayer(10)
    reluLayer
    fullyConnectedLayer(num_output_targets)
    regressionLayer
];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Identify parameters: https://www.mathworks.com/help/deeplearning/ref/trainingoptions.html
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
parameters = trainingOptions(...
    'rmsprop',... %optimizer
    'MaxEpochs',100,...
    'VerboseFrequency',1,...
    'ExecutionEnvironment','gpu', ...
    'InitialLearnRate',0.05,...
    'LearnRateSchedule','piecewise',...
    'LearnRateDropFactor',0.2,...
    'MiniBatchSize',100,...
    'Plots','training-progress',...
    'Shuffle', 'every-epoch'...
);

trained_net_ffnn = trainNetwork(norm_xtrain, norm_ytrain, model, parameters);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Forecast load
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
predicted_ffnn = predict(trained_net_ffnn, norm_xtest(6:29));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Analyze performance
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% accuracy measures
disp("Accuracy Measures");
RSME_ffnn = rmse(norm_ytest(6:29), predicted_ffnn);
MAPE_ffnn = mape(norm_ytest(6:29), predicted_ffnn);
MAE_ffnn = mae(norm_ytest(6:29), predicted_ffnn);

% plotting curves

ytest_denormalized_lstm = predicted_ffnn * sigPV + muPV;
plot_af(ytest(6:29), ytest_denormalized_lstm, 3)

% function declarations
function plot_af(actual, forecasted, axis_exponent)
    len = numel(actual);

    t = linspace(0, len, len);
    plot(t, actual);

    hold on

    plot(t, forecasted);

    hold off
    
    % plot axis settings 
    title('Actual and Forecasted Plots');
    legend('Actual','Forecasted');
    xlabel('Time (h)');
    ylabel('Power (W)');
    xlim([0 len]);
    ax = gca;
    ax.YAxis.Exponent = axis_exponent;
end

function RMSE = rmse(actual, forecasted)
    RMSE = sqrt(mean((actual - forecasted).^2));
    disp(["RMSE", RMSE]);
end

function MAPE = mape(actual, forecasted)
    MAPE = 100 * mean(abs((actual - forecasted)/actual));
    disp(["MAPE", MAPE, "%"]);
end

function MAE = mae(actual, forecasted)
    MAE = mean(abs(actual - forecasted));
    disp(["MAE", MAE]);
end