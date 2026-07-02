function generateFigure11_Convergence(trainInfo, outputDir)
% FIGURE 11: Training convergence
%   Subfigures: (a) Loss curves, (b) Accuracy curves, (c) LR schedule

    fig = figure('Position', [100 100 1200 350], 'Color', 'w');

    epochs = 1:length(trainInfo.TrainingLoss);
    valEpochs = find(~isnan(trainInfo.ValidationLoss));

    % (a) Loss curves
    subplot(1, 3, 1);
    plot(epochs, trainInfo.TrainingLoss, 'b-', 'LineWidth', 1.5); hold on;
    plot(valEpochs, trainInfo.ValidationLoss(valEpochs), 'r-', 'LineWidth', 1.5);
    xlabel('Iteration', 'FontSize', 11);
    ylabel('Loss', 'FontSize', 11);
    title('(a) Training & Validation Loss', 'FontSize', 12, 'FontWeight', 'bold');
    legend({'Training', 'Validation'}, 'FontSize', 9, 'Location', 'northeast');
    grid on; set(gca, 'FontSize', 10);

    % (b) Accuracy curves
    subplot(1, 3, 2);
    plot(epochs, trainInfo.TrainingAccuracy, 'b-', 'LineWidth', 1.5); hold on;
    plot(valEpochs, trainInfo.ValidationAccuracy(valEpochs), 'r-', 'LineWidth', 1.5);
    xlabel('Iteration', 'FontSize', 11);
    ylabel('Accuracy (%)', 'FontSize', 11);
    title('(b) Training & Validation Accuracy', 'FontSize', 12, 'FontWeight', 'bold');
    legend({'Training', 'Validation'}, 'FontSize', 9, 'Location', 'southeast');
    grid on; set(gca, 'FontSize', 10);

    % (c) Learning rate schedule (piecewise decay)
    subplot(1, 3, 3);
    lr = 0.001;
    lrs = zeros(1, 200);
    for ep = 1:200
        if mod(ep, 50) == 0
            lr = lr * 0.5;
        end
        lrs(ep) = lr;
    end
    semilogy(1:200, lrs, 'g-', 'LineWidth', 2);
    xlabel('Epoch', 'FontSize', 11);
    ylabel('Learning Rate', 'FontSize', 11);
    title('(c) Learning Rate Schedule', 'FontSize', 12, 'FontWeight', 'bold');
    grid on; set(gca, 'FontSize', 10);

    sgtitle('Figure 11. Training Convergence', 'FontSize', 14, 'FontWeight', 'bold');
    saveas(fig, fullfile(outputDir, 'Figure11_Convergence.fig'));
    saveas(fig, fullfile(outputDir, 'Figure11_Convergence.png'));
    fprintf('  Figure 11: Convergence plots saved.\n');
    close(fig);
end
