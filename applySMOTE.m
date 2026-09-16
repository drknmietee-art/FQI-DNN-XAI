function [X_new, y_new] = applySMOTE(X, y, varargin)
% APPLYSMOTE Synthetic Minority Over-sampling Technique
%
%   [X_new, y_new] = applySMOTE(X, y, 'k', 5)
%
%   Generates synthetic samples for the minority class to balance
%   the dataset. Reference: Chawla et al. [31] in paper.
%
%   Inputs:
%     X - [N x d] feature matrix
%     y - [N x 1] binary labels (0 or 1)
%     'k' - number of nearest neighbors (default: 5)
%
%   Outputs:
%     X_new - balanced feature matrix
%     y_new - balanced labels

    p = inputParser;
    addParameter(p, 'k', 5);
    parse(p, varargin{:});
    k = p.Results.k;

    classes = unique(y);
    counts = histcounts(y, [classes; max(classes)+1]);
    [maxCount, majIdx] = max(counts);
    [~, minIdx] = min(counts);
    majClass = classes(majIdx);
    minClass = classes(minIdx);

    X_min = X(y == minClass, :);
    X_maj = X(y == majClass, :);
    nMin = size(X_min, 1);
    nSynth = maxCount - nMin; % Number of synthetic samples needed

    if nSynth <= 0 || nMin < 2
        X_new = X;
        y_new = y;
        return;
    end

    % Adjust k if necessary
    k = min(k, nMin - 1);

    % Compute pairwise distances for minority class
    D = pdist2(X_min, X_min);
    synthSamples = zeros(nSynth, size(X, 2));

    for i = 1:nSynth
        % Pick random minority sample
        idx = randi(nMin);
        sample = X_min(idx, :);

        % Find k nearest neighbors
        [~, sortedIdx] = sort(D(idx, :));
        neighbors = sortedIdx(2:k+1); % Exclude self

        % Pick random neighbor
        nn = neighbors(randi(k));
        neighbor = X_min(nn, :);

        % Generate synthetic sample
        lambda = rand();
        synthSamples(i, :) = sample + lambda * (neighbor - sample);
    end

    % Combine original data with synthetic samples
    X_new = [X; synthSamples];
    y_new = [y; repmat(minClass, nSynth, 1)];

    % Shuffle
    perm = randperm(length(y_new));
    X_new = X_new(perm, :);
    y_new = y_new(perm);
end
