function y_pred = K_means_predictor(X, C, labels, K)
    y_pred = step_assign_cluster(X, C);
    
    for i = 1:K
        cluster_idx = find(y_pred == i);
        y_pred(cluster_idx) = labels(i);
    end
    y_pred = y_pred';
end

function y = step_assign_cluster(X, C)
    y = [];
    for i = 1:length(X)
        d = fxdist(X(:,i),C);
        [~,idx] = min(d);
        y = [y idx];
    end
end

function d = fxdist(x,C)
    % CHANGE
    d = [];
    for i = 1:size(C,2)
        d = [d norm(x-C(:,i))];
    end
    % DO NOT CHANGE
end
