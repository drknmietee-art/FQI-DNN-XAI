function X_fuzzy = fuzzyMembershipEncoding(X, centers, spreads, numTerms)
% FUZZYMEMBERSHIPENCODING Compute Gaussian fuzzy membership values
%
%   Implements Equations 2-4 from the paper:
%     mu_l(x_j) = exp(-(x_j - c_l)^2 / (2 * sigma_l^2))   (Eq. 2)
%     f_j = [mu_1(x_j), mu_2(x_j), ..., mu_L(x_j)]         (Eq. 3)
%     F = [f_1 || f_2 || ... || f_d]                         (Eq. 4)
%
%   Also adds pairwise interaction terms for top correlated features.
%
%   Inputs:
%     X       - [N x d] normalized input features
%     centers - [d x L] centers of Gaussian membership functions
%     spreads - [d x L] spreads (sigma) of membership functions
%     numTerms - L, number of linguistic terms
%
%   Output:
%     X_fuzzy - [N x (d*L + interaction_terms)] fuzzy-encoded features

    [N, d] = size(X);

    % Basic fuzzy encoding (Eqs. 2-4)
    X_mem = zeros(N, d * numTerms);
    for j = 1:d
        for l = 1:numTerms
            col_idx = (j-1)*numTerms + l;
            X_mem(:, col_idx) = exp(-((X(:,j) - centers(j,l)).^2) ./ ...
                                     (2 * spreads(j,l)^2 + eps));
        end
    end

    % Add pairwise fuzzy interaction features for enrichment
    % Use product of membership degrees for key feature pairs
    nInteract = min(d, 6); % Top features for interactions
    interactFeats = [];
    for j1 = 1:nInteract
        for j2 = (j1+1):nInteract
            % Product of max-membership for each pair
            mem_j1 = max(X_mem(:, (j1-1)*numTerms+1:j1*numTerms), [], 2);
            mem_j2 = max(X_mem(:, (j2-1)*numTerms+1:j2*numTerms), [], 2);
            interactFeats = [interactFeats, mem_j1 .* mem_j2];
        end
    end

    X_fuzzy = [X_mem, interactFeats];
end
