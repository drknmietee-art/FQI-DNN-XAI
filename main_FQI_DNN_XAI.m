%% ========================================================================
%  FQI-DNN-XAI: A Fuzzy Quantum-Inspired Deep Neural Network with
%  Explainable AI for Ischemic Heart Disease Prediction
%  ========================================================================
%  Main Script - Runs the complete pipeline
%
%  Requirements:
%    - MATLAB R2021b or later
%    - Deep Learning Toolbox
%    - Statistics and Machine Learning Toolbox
%
%  Dataset: Cleveland Heart Disease (UCI ML Repository)
%    https://archive.ics.uci.edu/dataset/45/heart+disease
%
%  Author: [Your Name]
%  Date: July 2026
%  ========================================================================

clc; clear; close all;
rng(42, 'twister'); % For reproducibility

fprintf('=============================================================\n');
fprintf(' FQI-DNN-XAI Framework for Ischemic Heart Disease Prediction\n');
fprintf('=============================================================\n\n');

%% ========================= 1. LOAD DATASET =============================
fprintf('[1/8] Loading Cleveland Heart Disease dataset...\n');

dataFile = fullfile(pwd, 'data', 'cleveland.csv');
if ~exist(dataFile, 'file')
    fprintf('  Downloading dataset from UCI repository...\n');
    if ~exist(fullfile(pwd, 'data'), 'dir'), mkdir(fullfile(pwd, 'data')); end
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data';
    websave(dataFile, url);
end

% Read data - handle '?' as missing
opts = detectImportOptions(dataFile, 'NumHeaderLines', 0);
opts.VariableNames = {'age','sex','cp','trestbps','chol','fbs','restecg',...
                      'thalach','exang','oldpeak','slope','ca','thal','target'};
opts = setvartype(opts, opts.VariableNames, 'char');
rawData = readtable(dataFile, opts);

numData = zeros(size(rawData));
for j = 1:width(rawData)
    numData(:, j) = str2double(rawData{:, j}); % '?' -> NaN
end

featureNames = {'age','sex','cp','trestbps','chol','fbs','restecg',...
                'thalach','exang','oldpeak','slope','ca','thal'};
X = numData(:, 1:13);
y = numData(:, 14);
y(y > 0) = 1; % Binarize

fprintf('  Dataset loaded: %d samples, %d features\n', size(X,1), size(X,2));
fprintf('  Class distribution: %d healthy (0), %d IHD (1)\n', sum(y==0), sum(y==1));

%% ========================= 2. PREPROCESSING ============================
fprintf('[2/8] Preprocessing data...\n');

% Remove rows with NaN
validRows = ~any(isnan(X), 2) & ~isnan(y);
X = X(validRows, :);
y = y(validRows);
fprintf('  After removing missing: %d samples\n', size(X,1));

% Store original for explainability
X_original = X;

% Min-Max Normalization (Eq. 1)
X_min = min(X); X_max = max(X);
X_norm = (X - X_min) ./ (X_max - X_min + eps);

%% ========================= 3. FUZZY ENCODING ===========================
fprintf('[3/8] Fuzzy Membership Encoding (Eqs. 2-7)...\n');

numTerms = 3; % Low, Medium, High
[N, d] = size(X_norm);

% DATA-DRIVEN centers: use 25th, 50th, 75th percentiles per feature
centers = zeros(d, numTerms);
spreads = zeros(d, numTerms);
for j = 1:d
    pcts = prctile(X_norm(:, j), [25, 50, 75]);
    centers(j, :) = pcts;
    % Spread = distance between adjacent centers / 2 (ensures overlap)
    spreads(j, 1) = max(abs(pcts(2) - pcts(1)) / 1.5, 0.1);
    spreads(j, 2) = max(abs(pcts(3) - pcts(1)) / 3.0, 0.1);
    spreads(j, 3) = max(abs(pcts(3) - pcts(2)) / 1.5, 0.1);
end

% Compute fuzzy features (Eq. 2-4)
X_fuzzy = fuzzyMembershipEncoding(X_norm, centers, spreads, numTerms);
fprintf('  Fuzzy: %d -> %d features\n', d, size(X_fuzzy, 2));

% Compute firing strengths for visualization (Eqs. 5-7)
[firingStrengths, normalizedFS] = computeFiringStrengths(X_norm, centers, spreads, numTerms);

%% =================== 4. QUANTUM-INSPIRED TRANSFORM ====================
fprintf('[4/8] Quantum-Inspired Feature Transformation (Eqs. 8-13)...\n');

numQubits = 8;
numLayers = 4;

% Combine original + fuzzy features, then apply quantum-inspired transform
X_combined = [X_norm, X_fuzzy];
[X_quantum, quantumParams] = quantumInspiredTransform(X_combined, numQubits, numLayers);

% FINAL features: concatenate original + fuzzy + quantum
X_final = [X_norm, X_fuzzy, X_quantum];
fprintf('  Final feature dim: %d (orig=%d + fuzzy=%d + quantum=%d)\n', ...
        size(X_final,2), d, size(X_fuzzy,2), size(X_quantum,2));

%% ================ 5. 10-FOLD CROSS-VALIDATION =========================
fprintf('[5/8] Running 10-fold stratified cross-validation...\n');
fprintf('  Using ensemble of 3 DNNs per fold for robust prediction.\n');

K = 10;
cv = cvpartition(y, 'KFold', K, 'Stratify', true);
nEnsemble = 3; % Ensemble size

results = struct();
results.accuracy = zeros(K, 1);
results.sensitivity = zeros(K, 1);
results.specificity = zeros(K, 1);
results.precision = zeros(K, 1);
results.f1score = zeros(K, 1);
results.auc = zeros(K, 1);
results.allTrue = [];
results.allPred = [];
results.allScores = [];

allSHAPValues = [];
allSHAPSamples = [];

for fold = 1:K
    fprintf('  Fold %d/%d: ', fold, K);

    trainIdx = training(cv, fold);
    testIdx = test(cv, fold);

    X_train_raw = X_norm(trainIdx, :);
    y_train = y(trainIdx);
    X_test_raw = X_norm(testIdx, :);
    y_test = y(testIdx);

    % --- SMOTE on training data ---
    [X_train_smote, y_train_smote] = applySMOTE(X_train_raw, y_train, 'k', 7);

    % --- Fuzzy Encoding ---
    X_train_fuzzy = fuzzyMembershipEncoding(X_train_smote, centers, spreads, numTerms);
    X_test_fuzzy = fuzzyMembershipEncoding(X_test_raw, centers, spreads, numTerms);

    % --- Quantum-Inspired Transform ---
    X_train_comb = [X_train_smote, X_train_fuzzy];
    X_test_comb = [X_test_raw, X_test_fuzzy];
    [X_train_q, ~] = quantumInspiredTransform(X_train_comb, numQubits, numLayers);
    [X_test_q, ~] = quantumInspiredTransform(X_test_comb, numQubits, numLayers);

    % --- Final feature set ---
    X_train_final = [X_train_smote, X_train_fuzzy, X_train_q];
    X_test_final = [X_test_raw, X_test_fuzzy, X_test_q];

    % --- Feature selection: variance + correlation-based ---
    % 1. Remove near-zero-variance features
    feat_var = var(X_train_final);
    keepIdx = feat_var > 0.01;
    X_train_final = X_train_final(:, keepIdx);
    X_test_final = X_test_final(:, keepIdx);

    % 2. Keep features with meaningful correlation to target (|r| > 0.05)
    feat_corr = abs(corr(X_train_final, y_train_smote));
    keepCorr = feat_corr > 0.05;
    X_train_final = X_train_final(:, keepCorr);
    X_test_final = X_test_final(:, keepCorr);

    % --- Ensemble of DNNs for robust prediction ---
    nEns = nEnsemble + 2; % Use 5 ensemble members for better averaging
    ensembleScores = zeros(length(y_test), nEns);
    for e = 1:nEns
        rng(42 + e*17 + fold*7); % Different seed per member & fold

        [net_e, trainInfo_e] = buildAndTrainFQIDNN(X_train_final, y_train_smote, ...
            'HiddenLayers', [64, 32], ...
            'DropoutRate', 0.45, ...
            'UseAttention', true, ...
            'MaxEpochs', 400, ...
            'MiniBatchSize', 16, ...
            'L2Reg', 3e-3, ...
            'LearnRate', 0.001, ...
            'Patience', 35, ...
            'Verbose', false);

        [~, scores_e] = predictFQIDNN(net_e, X_test_final);
        ensembleScores(:, e) = scores_e;

        % Save last ensemble member for SHAP & convergence plots
        if fold == K && e == nEns
            net = net_e;
            lastTrainInfo = trainInfo_e;
        end
    end

    % --- Ensemble averaging ---
    y_scores = mean(ensembleScores, 2);
    y_pred = double(y_scores >= 0.5);

    % --- Metrics ---
    [acc, sen, spe, pre, f1, aucVal] = computeMetrics(y_test, y_pred, y_scores);

    results.accuracy(fold) = acc;
    results.sensitivity(fold) = sen;
    results.specificity(fold) = spe;
    results.precision(fold) = pre;
    results.f1score(fold) = f1;
    results.auc(fold) = aucVal;
    results.allTrue = [results.allTrue; y_test];
    results.allPred = [results.allPred; y_pred];
    results.allScores = [results.allScores; y_scores];

    fprintf('Acc=%.2f%%, Sen=%.2f%%, Spe=%.2f%%, F1=%.2f%%, AUC=%.4f\n', ...
            acc*100, sen*100, spe*100, f1*100, aucVal);

    % SHAP values (only on last fold to save time)
    if fold == K
        shapValues = computeKernelSHAP(net, X_test_final, X_train_final, 30);
        allSHAPValues = [allSHAPValues; shapValues];
        allSHAPSamples = [allSHAPSamples; X_test_raw];
    end
end

fprintf('\n--- Overall Results (10-Fold CV) ---\n');
fprintf('  Accuracy:    %.2f +/- %.2f%%\n', mean(results.accuracy)*100, std(results.accuracy)*100);
fprintf('  Sensitivity: %.2f +/- %.2f%%\n', mean(results.sensitivity)*100, std(results.sensitivity)*100);
fprintf('  Specificity: %.2f +/- %.2f%%\n', mean(results.specificity)*100, std(results.specificity)*100);
fprintf('  Precision:   %.2f +/- %.2f%%\n', mean(results.precision)*100, std(results.precision)*100);
fprintf('  F1-Score:    %.2f +/- %.2f%%\n', mean(results.f1score)*100, std(results.f1score)*100);
fprintf('  AUC-ROC:     %.4f +/- %.4f\n', mean(results.auc), std(results.auc));

%% ================== 6. ABLATION STUDY ==================================
fprintf('\n[6/8] Running Ablation Study...\n');

ablationNames = {'Full FQI-DNN-XAI', 'w/o Fuzzy', 'w/o Quantum', ...
                 'w/o Attention', 'DNN only'};
ablationConfigs = {
    struct('useFuzzy', true,  'useQuantum', true,  'useAttention', true),
    struct('useFuzzy', false, 'useQuantum', true,  'useAttention', true),
    struct('useFuzzy', true,  'useQuantum', false, 'useAttention', true),
    struct('useFuzzy', true,  'useQuantum', true,  'useAttention', false),
    struct('useFuzzy', false, 'useQuantum', false, 'useAttention', false),
};
ablationResults = zeros(length(ablationNames), 6);

% Use 5 folds for ablation (faster, still valid)
Ka = 5;
cv_ab = cvpartition(y, 'KFold', Ka, 'Stratify', true);

for a = 1:length(ablationConfigs)
    cfg = ablationConfigs{a};
    aAcc = zeros(Ka,1); aSen = zeros(Ka,1); aSpe = zeros(Ka,1);
    aPre = zeros(Ka,1); aF1 = zeros(Ka,1);  aAUC = zeros(Ka,1);

    for fold = 1:Ka
        trainIdx = training(cv_ab, fold);
        testIdx = test(cv_ab, fold);
        X_tr = X_norm(trainIdx,:); y_tr = y(trainIdx);
        X_te = X_norm(testIdx,:);  y_te = y(testIdx);
        [X_tr, y_tr] = applySMOTE(X_tr, y_tr, 'k', 5);

        % Build features based on config
        Xtr_feat = X_tr; Xte_feat = X_te;
        if cfg.useFuzzy
            Xtr_f = fuzzyMembershipEncoding(X_tr, centers, spreads, numTerms);
            Xte_f = fuzzyMembershipEncoding(X_te, centers, spreads, numTerms);
            Xtr_feat = [Xtr_feat, Xtr_f];
            Xte_feat = [Xte_feat, Xte_f];
        end
        if cfg.useQuantum
            Xtr_f2 = fuzzyMembershipEncoding(X_tr, centers, spreads, numTerms);
            Xte_f2 = fuzzyMembershipEncoding(X_te, centers, spreads, numTerms);
            [Xtr_q, ~] = quantumInspiredTransform([X_tr, Xtr_f2], numQubits, numLayers);
            [Xte_q, ~] = quantumInspiredTransform([X_te, Xte_f2], numQubits, numLayers);
            Xtr_feat = [Xtr_feat, Xtr_q];
            Xte_feat = [Xte_feat, Xte_q];
        end

        rng(42 + fold*7 + a*3);
        [net_ab, ~] = buildAndTrainFQIDNN(Xtr_feat, y_tr, ...
            'HiddenLayers', [64, 32], ...
            'DropoutRate', 0.5, ...
            'UseAttention', cfg.useAttention, ...
            'MaxEpochs', 200, 'MiniBatchSize', 16, ...
            'L2Reg', 5e-3, 'LearnRate', 0.001, ...
            'Patience', 25, 'Verbose', false);

        [yp, ys] = predictFQIDNN(net_ab, Xte_feat);
        [ac,sn,sp,pr,f1v,au] = computeMetrics(y_te, yp, ys);
        aAcc(fold)=ac; aSen(fold)=sn; aSpe(fold)=sp;
        aPre(fold)=pr; aF1(fold)=f1v; aAUC(fold)=au;
    end

    ablationResults(a,:) = [mean(aAcc), mean(aSen), mean(aSpe), ...
                            mean(aPre), mean(aF1), mean(aAUC)];
    fprintf('  %s: Acc=%.2f%%\n', ablationNames{a}, mean(aAcc)*100);
end

%% ================= 7. BASELINE COMPARISONS =============================
fprintf('\n[7/8] Running Baseline Comparisons...\n');
baselineResults = runBaselineComparisons(X_norm, y, cv);

%% ================= 8. GENERATE ALL FIGURES =============================
fprintf('\n[8/8] Generating Figures and Tables...\n');

outputDir = fullfile(pwd, 'results');
if ~exist(outputDir, 'dir'), mkdir(outputDir); end

generateFigure2_MembershipFunctions(X_norm, centers, spreads, featureNames, outputDir);
generateFigure3_FuzzyRuleActivation(firingStrengths, normalizedFS, y, outputDir);
generateFigure5_SHAPAnalysis(allSHAPValues, allSHAPSamples, featureNames, outputDir);
generateFigure6_LIMEExplanations(X_norm, y, centers, spreads, numTerms, ...
    numQubits, numLayers, featureNames, outputDir);
generateFigure7_ROCCurves(results, baselineResults, outputDir);
generateFigure8_ConfusionMatrices(results, baselineResults, outputDir);
generateFigure9_AblationStudy(ablationNames, ablationResults, outputDir);
generateFigure10_FeatureInteractions(X_quantum, y, featureNames, outputDir);
generateFigure11_Convergence(lastTrainInfo, outputDir);
generateFigure12_StatisticalAnalysis(results, baselineResults, outputDir);

% Save all results
save(fullfile(outputDir, 'FQI_DNN_XAI_Results.mat'), ...
    'results', 'ablationResults', 'ablationNames', 'baselineResults', ...
    'allSHAPValues', 'featureNames', 'centers', 'spreads', 'quantumParams');

generateResultsTables(results, ablationResults, ablationNames, baselineResults, outputDir);

fprintf('\n=============================================================\n');
fprintf(' All results saved to: %s\n', outputDir);
fprintf('=============================================================\n');
