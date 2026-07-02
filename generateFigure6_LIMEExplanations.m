function generateFigure6_LIMEExplanations(X, y, centers, spreads, numTerms, ...
    numQubits, numLayers, featureNames, outputDir)
% FIGURE 6: LIME local explanations
%   Subfigures: (a) IHD-positive, (b) IHD-negative, (c) Borderline case

    fig = figure('Position', [100 100 1200 400], 'Color', 'w');
    d = length(featureNames);

    % Train a simple model for LIME
    X_f = fuzzyMembershipEncoding(X, centers, spreads, numTerms);
    [X_q, ~] = quantumInspiredTransform(X_f, numQubits, numLayers);

    % Use linear model as surrogate
    mdl = fitglm(X, y, 'Distribution', 'binomial');
    coeffs = mdl.Coefficients.Estimate(2:end);

    sampleTypes = {'(a) IHD Positive', '(b) IHD Negative', '(c) Borderline'};
    posIdx = find(y == 1, 1);
    negIdx = find(y == 0, 1);

    % Find borderline case (prediction closest to 0.5)
    probs = predict(mdl, X);
    [~, borderIdx] = min(abs(probs - 0.5));

    sampleIndices = [posIdx, negIdx, borderIdx];

    for s = 1:3
        subplot(1, 3, s);
        idx = sampleIndices(s);
        x_sample = X(idx, :);

        % LIME: perturb around sample, fit local linear model (Eq. 19)
        nPerturb = 200;
        X_perturb = repmat(x_sample, nPerturb, 1) + ...
                    0.1 * randn(nPerturb, d) .* std(X);
        X_perturb = max(0, min(1, X_perturb));

        y_perturb = predict(mdl, X_perturb);

        % Proximity weights (exponential kernel)
        distances = sqrt(sum((X_perturb - x_sample).^2, 2));
        weights = exp(-distances.^2 / (0.75^2));

        % Fit weighted linear model
        W = diag(weights);
        localCoeffs = (X_perturb' * W * X_perturb + 0.01*eye(d)) \ ...
                      (X_perturb' * W * y_perturb);

        % Plot
        [sortedC, sortI] = sort(localCoeffs);
        colors = zeros(d, 3);
        for j = 1:d
            if sortedC(j) >= 0
                colors(j, :) = [0.2 0.7 0.3];
            else
                colors(j, :) = [0.8 0.3 0.2];
            end
        end
        barh(sortedC, 'FaceColor', 'flat', 'CData', colors);
        yticks(1:d);
        yticklabels(featureNames(sortI));
        xlabel('Local Feature Weight', 'FontSize', 10);
        title(sampleTypes{s}, 'FontSize', 12, 'FontWeight', 'bold');
        set(gca, 'FontSize', 8);
    end

    sgtitle('Figure 6. LIME Local Explanations', 'FontSize', 14, 'FontWeight', 'bold');
    saveas(fig, fullfile(outputDir, 'Figure6_LIMEExplanations.fig'));
    saveas(fig, fullfile(outputDir, 'Figure6_LIMEExplanations.png'));
    fprintf('  Figure 6: LIME explanations saved.\n');
    close(fig);
end
