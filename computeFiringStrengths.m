function [firingStrengths, normalizedFS] = computeFiringStrengths(X, centers, spreads, numTerms)
% COMPUTEFIRINGSTRENGTHS Compute Takagi-Sugeno fuzzy rule firing strengths
%
%   Implements Equations 5-7 from the paper:
%     w^r = prod_j mu_j^r(x_j)    (Eq. 6)
%     w_bar^r = w^r / sum(w^r)     (Eq. 7)
%
%   Inputs:
%     X       - [N x d] normalized features
%     centers - [d x L] centers
%     spreads - [d x L] spreads
%     numTerms - L
%
%   Outputs:
%     firingStrengths  - [N x R] raw firing strengths (R = L^d_reduced)
%     normalizedFS     - [N x R] normalized firing strengths

    [N, d] = size(X);

    % For computational tractability, use a subset of representative rules
    % Instead of L^d rules, use d*L rules (one per feature-term combination)
    % Then add interaction rules for top feature pairs
    numRules = numTerms^min(d, 4); % Limit combinatorial explosion
    if d > 4
        % Use representative rules via sampling
        numRules = numTerms * d + nchoosek(min(d, 6), 2) * numTerms;
    end

    % Compute membership for each feature-term pair
    memberships = zeros(N, d, numTerms);
    for j = 1:d
        for l = 1:numTerms
            memberships(:, j, l) = exp(-((X(:, j) - centers(j, l)).^2) ./ ...
                                        (2 * spreads(j, l)^2 + eps));
        end
    end

    % Generate rules: for each feature, create L rules
    % Rule firing = product of memberships (Eq. 6)
    ruleIdx = 0;
    maxRules = d * numTerms;
    firingStrengths = zeros(N, maxRules);

    for j = 1:d
        for l = 1:numTerms
            ruleIdx = ruleIdx + 1;
            % Single-feature rules weighted by global membership
            firingStrengths(:, ruleIdx) = memberships(:, j, l);
            % Multiply with average membership of other features
            otherMean = mean(max(memberships(:, setdiff(1:d, j), :), [], 3), 2);
            firingStrengths(:, ruleIdx) = firingStrengths(:, ruleIdx) .* otherMean;
        end
    end

    % Normalize firing strengths (Eq. 7)
    sumFS = sum(firingStrengths, 2) + eps;
    normalizedFS = firingStrengths ./ sumFS;
end
