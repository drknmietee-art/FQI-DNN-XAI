function shapValues = computeKernelSHAP(net, X_test, X_background, nSamples)
% COMPUTEKERNELSHAP Fast Kernel SHAP approximation
%
%   Vectorized implementation - processes feature perturbations in batches.
%
%   Inputs:
%     net          - trained network
%     X_test       - [N x d] test samples
%     X_background - [M x d] background dataset
%     nSamples     - coalitions per feature (reduced for speed)
%
%   Output:
%     shapValues   - [N x d] SHAP values

    [N, d] = size(X_test);
    shapValues = zeros(N, d);

    % Use small background subset
    nBg = min(20, size(X_background, 1));
    bgIdx = randperm(size(X_background, 1), nBg);
    X_bg = X_background(bgIdx, :);
    nSamples = min(nSamples, 30); % Cap for speed

    % Precompute background mean for faster masking
    bg_mean = mean(X_bg, 1);

    for i = 1:N
        x = X_test(i, :);

        for j = 1:d
            % Create batch of perturbed inputs
            X_with = repmat(bg_mean, nSamples, 1);
            X_without = repmat(bg_mean, nSamples, 1);

            for s = 1:nSamples
                mask = rand(1, d) > 0.5;
                mask(j) = false;
                bg = X_bg(randi(nBg), :);

                x_w = bg; x_w(mask) = x(mask); x_w(j) = x(j);
                x_wo = bg; x_wo(mask) = x(mask);

                X_with(s, :) = x_w;
                X_without(s, :) = x_wo;
            end

            % Batch predict (much faster than one-by-one)
            sc_w = predict(net, X_with);
            sc_wo = predict(net, X_without);

            if size(sc_w, 2) >= 2
                shapValues(i, j) = mean(sc_w(:,2) - sc_wo(:,2));
            else
                shapValues(i, j) = mean(sc_w(:,1) - sc_wo(:,1));
            end
        end
    end
end
