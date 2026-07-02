function s = softmax_custom(x)
% Custom softmax function
    e = exp(x - max(x));
    s = e / sum(e);
end
