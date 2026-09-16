function generateFigure9_AblationStudy(ablationNames, ablationResults, outputDir)
% FIGURE 9: Ablation study results
%   Subfigures: (a) Grouped bar chart, (b) Radar plot with rescaled axes
%
%   ablationResults: [nConfigs x 6] matrix with columns
%                    [Acc, Sen, Spe, Pre, F1, AUC] in [0,1] range

    nConfigs = size(ablationResults, 1);

    % ---- Color scheme (distinguishable, colorblind-friendly) ----
    cmap = [0.12 0.47 0.71;   % blue   - Full
            0.89 0.10 0.11;   % red    - w/o Fuzzy
            0.17 0.63 0.17;   % green  - w/o Quantum
            1.00 0.50 0.05;   % orange - w/o Attention
            0.55 0.34 0.29];  % brown  - DNN only
    if nConfigs > size(cmap,1)
        cmap = [cmap; lines(nConfigs - size(cmap,1))];
    end

    fig = figure('Position', [50 50 1300 550], 'Color', 'w');

    % ============================================================
    % (a) Grouped bar chart — all 6 metrics side by side
    % ============================================================
    subplot(1, 2, 1);
    barData = ablationResults * 100;  % [nConfigs x 6]
    metricNames = {'Acc', 'Sen', 'Spe', 'Pre', 'F1', 'AUC'};

    b = bar(barData, 'grouped');
    for m = 1:6
        b(m).FaceColor = 'flat';
        % Assign a shade per metric using a gray-to-blue gradient
        baseColor = [0.2+0.12*m, 0.35+0.08*m, 0.55+0.07*m];
        b(m).CData = repmat(min(baseColor, 1), nConfigs, 1);
    end

    xticks(1:nConfigs);
    xticklabels(ablationNames);
    xtickangle(25);
    ylabel('Score (%)', 'FontSize', 11, 'FontWeight', 'bold');
    title('(a) Ablation Study — All Metrics', 'FontSize', 12, 'FontWeight', 'bold');
    legend(metricNames, 'Location', 'southoutside', 'Orientation', 'horizontal', ...
           'FontSize', 8);
    grid on; box on;
    ylim([floor(min(barData(:))-3), ceil(max(barData(:))+3)]);
    set(gca, 'FontSize', 9);

    % ============================================================
    % (b) Radar / Spider plot — rescaled for visibility
    % ============================================================
    subplot(1, 2, 2);
    nMetrics = 6;
    metricLabels = {'Accuracy', 'Sensitivity', 'Specificity', ...
                    'Precision', 'F1-Score', 'AUC-ROC'};
    angles = linspace(0, 2*pi, nMetrics + 1);  % +1 to close

    % --- Rescale to [0,1] within visible data range ---
    % Find per-metric min/max across configs, then add padding
    vals_pct = ablationResults * 100;           % work in %
    col_min = min(vals_pct, [], 1);
    col_max = max(vals_pct, [], 1);
    ax_min  = floor(col_min - 2);              % lower bound with padding
    ax_max  = ceil(col_max + 2);               % upper bound with padding
    ax_range = ax_max - ax_min;
    ax_range(ax_range == 0) = 1;               % avoid /0

    hold on;

    % --- Grid rings (5 concentric) ---
    nRings = 5;
    ringFracs = linspace(0, 1, nRings + 1);    % 0, 0.2, 0.4, ... 1.0
    ringFracs = ringFracs(2:end);               % skip center
    theta_circle = linspace(0, 2*pi, 120);
    for ri = 1:length(ringFracs)
        r = ringFracs(ri);
        plot(r*cos(theta_circle), r*sin(theta_circle), '-', ...
             'Color', [0.85 0.85 0.85], 'LineWidth', 0.5);
    end

    % --- Spoke lines + metric labels ---
    for k = 1:nMetrics
        % Spoke
        plot([0, 1.08*cos(angles(k))], [0, 1.08*sin(angles(k))], '-', ...
             'Color', [0.75 0.75 0.75], 'LineWidth', 0.5);

        % Metric label (outer)
        lx = 1.22 * cos(angles(k));
        ly = 1.22 * sin(angles(k));
        text(lx, ly, metricLabels{k}, 'HorizontalAlignment', 'center', ...
             'FontSize', 9, 'FontWeight', 'bold');
    end

    % --- Ring value labels along first spoke ---
    for ri = 1:length(ringFracs)
        r = ringFracs(ri);
        % Show the actual % value this ring represents for Accuracy (metric 1)
        actual_val = ax_min(1) + r * ax_range(1);
        tx = r * cos(angles(1)) + 0.04;
        ty = r * sin(angles(1)) - 0.04;
        text(tx, ty, sprintf('%.0f%%', actual_val), ...
             'FontSize', 7, 'Color', [0.45 0.45 0.45]);
    end

    % --- Plot each configuration as filled polygon ---
    legendHandles = gobjects(nConfigs, 1);
    markers = {'o', 's', 'd', '^', 'v'};

    for a = 1:nConfigs
        % Normalize each metric to [0,1] within its axis range
        norm_vals = (vals_pct(a, :) - ax_min) ./ ax_range;
        norm_vals = max(norm_vals, 0.02);       % keep tiny minimum for visibility
        norm_vals = [norm_vals, norm_vals(1)];   % close polygon

        xp = norm_vals .* cos(angles);
        yp = norm_vals .* sin(angles);

        % Filled polygon with transparency
        fill(xp, yp, cmap(a,:), 'FaceAlpha', 0.08, 'EdgeColor', 'none');

        % Bold outline + markers
        mk = markers{min(a, length(markers))};
        legendHandles(a) = plot(xp, yp, ['-' mk], ...
            'Color', cmap(a,:), 'LineWidth', 2, ...
            'MarkerSize', 7, 'MarkerFaceColor', cmap(a,:), ...
            'MarkerEdgeColor', 'w');
    end

    axis equal; axis off;
    xlim([-1.5 1.5]); ylim([-1.5 1.5]);

    legend(legendHandles, ablationNames, ...
           'Location', 'southoutside', 'FontSize', 8, ...
           'Orientation', 'horizontal', 'NumColumns', 3);
    title('(b) Multi-Metric Radar (rescaled)', 'FontSize', 12, 'FontWeight', 'bold');

    % ---- Overall title and save ----
    sgtitle('Figure 9. Ablation Study Results', 'FontSize', 14, 'FontWeight', 'bold');

    saveas(fig, fullfile(outputDir, 'Figure9_AblationStudy.fig'));
    exportgraphics(fig, fullfile(outputDir, 'Figure9_AblationStudy.png'), ...
                   'Resolution', 300);
    fprintf('  Figure 9: Ablation study saved.\n');
    close(fig);
end
