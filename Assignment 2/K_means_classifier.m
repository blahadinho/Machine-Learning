function [C, y_pred, labels] = K_means_classifier(X,y_true, K)

    [y_cluster, C] = K_means_clustering(X,K);
    labels = [];
    for i = 1:K
        labels = [labels mode(y_true(find(y_cluster == i)))]; 
    end
    y_pred = y_cluster;
    for i = 1:K
        cluster_idx = find(y_cluster == i);
        y_pred(cluster_idx) = labels(i);
    end
    y_pred = y_pred';    
end