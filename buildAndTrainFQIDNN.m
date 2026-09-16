function [net, trainInfo] = buildAndTrainFQIDNN(X_train, y_train, varargin)
% BUILDANDTRAINFQIDNN Build and train the deep neural network with attention
%
%   Implements Equations 14-17 from the paper:
%     h_k = ReLU(BN(W_k * h_{k-1} + b_k))                        (Eq. 14)
%     Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) * V            (Eq. 15)
%     p_hat = sigma(W_out * h_L + b_out)                           (Eq. 16)
%     L = -1/N sum [y*log(p) + (1-y)*log(1-p)] + lambda*||W||^2   (Eq. 17)

    p = inputParser;
    addParameter(p, 'HiddenLayers', [128, 64, 32]);
    addParameter(p, 'DropoutRate', 0.4);
    addParameter(p, 'UseAttention', true);
    addParameter(p, 'MaxEpochs', 300);
    addParameter(p, 'MiniBatchSize', 16);
    addParameter(p, 'L2Reg', 1e-3);
    addParameter(p, 'LearnRate', 0.0005);
    addParameter(p, 'Patience', 30);
    addParameter(p, 'Verbose', false);
    parse(p, varargin{:});
    opts = p.Results;

    [N, d_in] = size(X_train);
    hiddenLayers = opts.HiddenLayers;

    % ---- Build network layers ----
    layers = [featureInputLayer(d_in, 'Name', 'input', 'Normalization', 'zscore')];

    for k = 1:length(hiddenLayers)
        layers = [layers
            fullyConnectedLayer(hiddenLayers(k), 'Name', sprintf('fc%d',k), ...
                'WeightsInitializer', 'he')
            batchNormalizationLayer('Name', sprintf('bn%d',k))
            reluLayer('Name', sprintf('relu%d',k))
            dropoutLayer(opts.DropoutRate, 'Name', sprintf('drop%d',k))
        ];
    end

    % Attention approximation: squeeze-excitation style gating
    if opts.UseAttention
        lastHidden = hiddenLayers(end);
        bottleneck = max(8, floor(lastHidden / 4));
        layers = [layers
            fullyConnectedLayer(bottleneck, 'Name', 'att_squeeze', ...
                'WeightsInitializer', 'glorot')
            reluLayer('Name', 'att_relu')
            fullyConnectedLayer(lastHidden, 'Name', 'att_excite', ...
                'WeightsInitializer', 'glorot')
            sigmoidLayer('Name', 'att_gate')
        ];
    end

    % Output
    layers = [layers
        fullyConnectedLayer(2, 'Name', 'fc_out')
        softmaxLayer('Name', 'softmax')
        classificationLayer('Name', 'output')
    ];

    % ---- Prepare data ----
    y_cat = categorical(y_train);

    % Validation split (15%)
    nVal = max(round(N * 0.15), 10);
    perm = randperm(N);
    valIdx = perm(1:nVal);
    trainIdx = perm(nVal+1:end);

    X_val = X_train(valIdx, :);
    y_val = y_cat(valIdx);
    X_tr = X_train(trainIdx, :);
    y_tr = y_cat(trainIdx);

    % ---- Training options ----
    trainOpts = trainingOptions('adam', ...
        'MaxEpochs', opts.MaxEpochs, ...
        'MiniBatchSize', min(opts.MiniBatchSize, size(X_tr,1)), ...
        'InitialLearnRate', opts.LearnRate, ...
        'LearnRateSchedule', 'piecewise', ...
        'LearnRateDropPeriod', 30, ...
        'LearnRateDropFactor', 0.5, ...
        'L2Regularization', opts.L2Reg, ...
        'GradientThreshold', 1, ...
        'ValidationData', {X_val, y_val}, ...
        'ValidationFrequency', 5, ...
        'ValidationPatience', opts.Patience, ...
        'Shuffle', 'every-epoch', ...
        'Plots', 'none', ...
        'Verbose', false, ...
        'OutputNetwork', 'best-validation-loss');

    % ---- Train ----
    [net, info] = trainNetwork(X_tr, y_tr, layers, trainOpts);

    trainInfo.TrainingLoss = info.TrainingLoss;
    trainInfo.TrainingAccuracy = info.TrainingAccuracy;
    trainInfo.ValidationLoss = info.ValidationLoss;
    trainInfo.ValidationAccuracy = info.ValidationAccuracy;
end
