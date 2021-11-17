function [y,C] = K_means_clustering(X,K)

% Calculating cluster centroids and cluster assignments for:
% Input:    X   DxN matrix of input data
%           K   Number of clusters
%
% Output:   y   Nx1 vector of cluster assignments
%           C   DxK matrix of cluster centroids

[D,N] = size(X);

intermax = 50;
conv_tol = 1e-6;
% Initialize
C = repmat(mean(X,2),1,K) + repmat(std(X,[],2),1,K).*randn(D,K);
y = zeros(N,1);
Cold = C;

for kiter = 1:intermax
    % CHANGE
    % Step 1: Assign to clusters
    y = step_assign_cluster(X,Cold);
    
    % Step 2: Assign new clusters
    C = step_compute_mean(X, y, K);
        
    if fcdist(C,Cold) < conv_tol
        return
    end
    %disp(['Current error: ', num2str(fcdist(C,Cold))])
    Cold = C;
    % DO NOT CHANGE
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

function d = fcdist(C1,C2)
    % CHANGE
    d = [];
    for i = 1:size(C1,2)
        d = [d norm(C2(:,i)-C1(:,i))];
    end
    % DO NOT CHANGE
end

function y = step_assign_cluster(X, C)
    y = [];
    for i = 1:length(X)
        d = fxdist(X(:,i),C);
        [~,idx] = min(d);
        y = [y idx];
    end
end

function C = step_compute_mean(X, y, K)
    C = [];
    for i = 1:K
        N = length(find(y == i));
        C = [C (1/N)*sum(X(:,find(y == i)),2)];
    end
end