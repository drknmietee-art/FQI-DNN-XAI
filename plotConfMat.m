function plotConfMat(cm, titleStr)
% Helper to plot a single confusion matrix
    imagesc(cm);
    colormap(gca, [1 1 1; 0.7 0.85 1; 0.3 0.5 0.8; 0.1 0.2 0.6]);
    for i = 1:size(cm, 1)
        for j = 1:size(cm, 2)
            textColor = 'k';
            if cm(i,j) > max(cm(:))*0.6, textColor = 'w'; end
            text(j, i, num2str(cm(i,j)), 'HorizontalAlignment', 'center', ...
                 'FontSize', 14, 'FontWeight', 'bold', 'Color', textColor);
        end
    end
    xlabel('Predicted'); ylabel('Actual');
    xticks([1 2]); yticks([1 2]);
    xticklabels({'Healthy', 'IHD'}); yticklabels({'Healthy', 'IHD'});
    title(titleStr, 'FontSize', 10, 'FontWeight', 'bold');
    set(gca, 'FontSize', 8);
end
