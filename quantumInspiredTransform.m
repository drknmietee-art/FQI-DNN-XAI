function [X_quantum, params] = quantumInspiredTransform(X, numQubits, numLayers)
% QUANTUMINSPIREDTRANSFORM Vectorized quantum-inspired feature transformation
%
%   FULLY VECTORIZED - processes all N samples simultaneously using matrix
%   operations. No per-sample loops.
%
%   Uses PCA-based projection (data-driven) instead of random projection
%   to ensure quantum features capture meaningful data variance.
%
%   Implements Equations 8-13 from the paper.
%
%   Inputs:
%     X         - [N x d_in] input features
%     numQubits - number of qubit-inspired units (n)
%     numLayers - number of variational layers (P)
%
%   Outputs:
%     X_quantum - [N x 2*numQubits] quantum-inspired features
%     params    - struct with parameters

    [N, d_in] = size(X);
    rng(42);

    % DATA-DRIVEN Projection: PCA-based (d_in -> numQubits)
    % Instead of random W_proj, use top PCA components so quantum
    % features capture actual data variance
    X_centered = X - mean(X, 1);
    if d_in >= numQubits
        [coeff, ~, ~] = pca(X_centered, 'NumComponents', numQubits);
        W_proj = coeff; % [d_in x numQubits] — top principal directions
    else
        [coeff, ~, ~] = pca(X_centered);
        % Pad with small random vectors if fewer PCs than qubits
        W_proj = [coeff, randn(d_in, numQubits - size(coeff,2)) * 0.01];
    end
    b_proj = zeros(1, numQubits);

    % Variational parameters — seeded for reproducibility
    % Use structured initialization (not purely random) for stability
    W_var = cell(numLayers, 1);
    b_var = cell(numLayers, 1);
    ent_w = cell(numLayers, 1);
    for p = 1:numLayers
        % Orthogonal-like initialization via QR decomposition
        [Q, ~] = qr(randn(numQubits));
        W_var{p} = Q(1:numQubits, 1:numQubits) * sqrt(2 / numQubits);
        b_var{p} = randn(1, numQubits) * 0.05; % Smaller bias init
        ent_w{p} = randn(1, numQubits) * 0.2;  % Moderate entanglement
    end

    % Step 1: Angle encoding (Eq. 8) — ALL samples at once
    % PCA projection ensures angles reflect true data structure
    theta = pi * tanh(X * W_proj + b_proj);  % [N x numQubits]

    % Step 2: Initial state (Eq. 9-10)
    cos_a = cos(theta / 2);  % [N x numQubits]
    sin_a = sin(theta / 2);  % [N x numQubits]

    % Step 3: Multi-layer variational circuit (Eq. 12)
    for p = 1:numLayers
        % Rotation layer — matrix multiply, all samples
        phi = theta * W_var{p} + b_var{p};  % [N x numQubits]
        cos_new = cos_a .* cos(phi/2) - sin_a .* sin(phi/2);
        sin_new = sin_a .* cos(phi/2) + cos_a .* sin(phi/2);

        % Entanglement layer (Eq. 11) — circular neighbor interaction
        sin_shifted = circshift(sin_new, [0, 1]);
        ent_angle = ent_w{p} .* sin_new .* sin_shifted;  % [N x numQubits]

        cos_a = cos_new .* cos(ent_angle) - sin_new .* sin(ent_angle);
        sin_a = sin_new .* cos(ent_angle) + cos_new .* sin(ent_angle);
    end

    % Step 4: Measurement (Eq. 13) — expectation values
    Z_exp = cos_a.^2 - sin_a.^2;        % <Z> [N x numQubits]
    Y_exp = 2 * cos_a .* sin_a;          % <Y> [N x numQubits]
    X_quantum = [Z_exp, Y_exp];           % [N x 2*numQubits]

    % Normalize
    mu = mean(X_quantum, 1);
    sg = std(X_quantum, 0, 1) + eps;
    X_quantum = (X_quantum - mu) ./ sg;

    params.W_proj = W_proj;
    params.b_proj = b_proj;
    params.W_var = W_var;
    params.b_var = b_var;
    params.ent_w = ent_w;
    params.theta = cat(1, b_var{:});
end
