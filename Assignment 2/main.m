%% Task T3
xp = [-2 2];
xn = [-1 1];
gx = -3:3;
gy = repelem(2.5, 7);

plot(xn,xn.^2,'b*')
hold on
plot(xp,xp.^2,'r*')
plot(gx, gy,'black')
legend('Class -1', 'Class +1', 'Decision boundary')
title('Data points in (x,x^2) space with decision boundary')
%% Task T4
xp = [-3 -2 2 4];
xn = [-1 0 1];
gx = -4:5;
gy = repelem(2.5, 10);
plot(xn,xn.^2,'b*')
hold on
plot(xp,xp.^2,'r*')
plot(gx, gy,'black')
legend('Class -1', 'Class +1', 'Decision boundary')
title('Data points in (x,x^2) space with decision boundary')
%% Task E1
X = train_data_01-mean(train_data_01,2);
[U,S,V] = svd(X);
%%
X0 = X(:,find(train_labels_01 == 0));
X1 = X(:,find(train_labels_01 == 1));
p1_0 = X0'*U(:,1);
p2_0 = X0'*U(:,2);
p1_1 = X1'*U(:,1);
p2_1 = X1'*U(:,2);
plot(p1_0,p2_0, 'r.')
hold on 
plot(p1_1,p2_1,'b.')
legend('Class 0','Class 1');
title('The two classes represented using PCA')
%% Some testing
figure(1)
histogram(p1_0)
hold on
histogram(p1_1)
figure(2)
histogram(p2_0)
hold on
histogram(p2_1)
%% Task E2 K = 2
[y,C] = K_means_clustering(train_data_01, 2);

X0 = X(:,find(y == 1));
X1 = X(:,find(y == 2));
p1_0 = X0'*U(:,1);
p2_0 = X0'*U(:,2);
p1_1 = X1'*U(:,1);
p2_1 = X1'*U(:,2);
C_mean = C - mean(train_data_01,2);
pc = C_mean'*U(:,1);
pc2 = C_mean'*U(:,2);
plot(p1_0,p2_0, 'r.')
hold on 
plot(p1_1,p2_1,'b.')
plot(pc,pc2,'g.','MarkerSize',20)
legend('Cluster 1','Cluster 2', 'Centroids')
title('K-means clustering with K = 2')
%% Task E2 K = 5
[y,C] = K_means_clustering(train_data_01, 5);

X0 = X(:,find(y == 1));
X1 = X(:,find(y == 2));
X2 = X(:,find(y == 3));
X3 = X(:,find(y == 4));
X4 = X(:,find(y == 5)); %#ok<*FNDSB>

p1_0 = X0'*U(:,1);
p2_0 = X0'*U(:,2);
p1_1 = X1'*U(:,1);
p2_1 = X1'*U(:,2);
p1_2 = X2'*U(:,1);
p2_2 = X2'*U(:,2);
p1_3 = X3'*U(:,1);
p2_3 = X3'*U(:,2);
p1_4 = X4'*U(:,1);
p2_4 = X4'*U(:,2);
C_mean = C - mean(train_data_01,2);
pc = C_mean'*U(:,1);
pc2 = C_mean'*U(:,2);
plot(p1_0,p2_0,'r.')
hold on 
plot(p1_1,p2_1,'b.')
plot(p1_2,p2_2,'y.')
plot(p1_3,p2_3,'c.')
plot(p1_4,p2_4,'m.')
plot(pc,pc2,'g.','MarkerSize',20)
legend('Cluster 1','Cluster 2','Cluster 3','Cluster 4','Cluster 5', 'Centriods')
title('K-means clustering with K = 5')
%% Task E3 K = 2
figure
sgtitle('Images of the centroids after K-means clustering')
subplot(121)
imshow(reshape(C(:,1),28,28),'InitialMagnification','fit')
title('First centroid')
subplot(122)
imshow(reshape(C(:,2),28,28),'InitialMagnification','fit')
title('Second centroid')
%% Task E3 K = 2
figure
sgtitle('Images of the centroids after K-means clustering')
subplot(151)
imshow(reshape(C(:,1),28,28),'InitialMagnification','fit')
title('First centroid')
subplot(152)
imshow(reshape(C(:,2),28,28),'InitialMagnification','fit')
title('Second centroid')
subplot(153)
imshow(reshape(C(:,3),28,28),'InitialMagnification','fit')
title('Third centroid')
subplot(154)
imshow(reshape(C(:,4),28,28),'InitialMagnification','fit')
title('Fourth centroid')
subplot(155)
imshow(reshape(C(:,5),28,28),'InitialMagnification','fit')
title('Fifth centroid')
%% Task E4
K = 2;
[C, y_cluster, labels] = K_means_classifier(train_data_01, train_labels_01, K);
y_pred_test = K_means_predictor(test_data_01, C, labels, K);

X_test = test_data_01-mean(test_data_01,2);
[U2,S2,V2] = svd(X_test);

X0 = X_test(:,find(y_pred_test == 0));
X1 = X_test(:,find(y_pred_test == 1));
p1_0 = X0'*U2(:,1);
p2_0 = X0'*U2(:,2);
p1_1 = X1'*U2(:,1);
p2_1 = X1'*U2(:,2);
C_mean = C - mean(C,2);
pc = C_mean'*U(:,1);
pc2 = C_mean'*U(:,2);
plot(p1_0,p2_0, 'r.')
hold on 
plot(p1_1,p2_1,'b.')
plot(pc,pc2,'g.','MarkerSize',20)
legend('Predicted class 0','Predicted class 1', 'Centroids')
title('K-means classification with K = 2')

misclass_train = nnz(y_cluster - train_labels_01)
misclass_test = nnz(y_pred_test - test_labels_01)
%%
length(test_labels_01);
length(find(test_labels_01(find(y_pred_test == 1))==0))
%% Task E5 different values of K
K = 12;
[C, y_cluster, labels] = K_means_classifier(X, train_labels_01, K);
y_pred_test = K_means_predictor(test_data_01, C, labels, K);

misclass_train = nnz(y_cluster - train_labels_01);
misclass_test = nnz(y_pred_test - test_labels_01);
disp(['K: ',num2str(K),' Test: ', num2str(misclass_test),' Train: ', num2str(misclass_train)])
disp(['Test %: ', num2str(misclass_test/length(test_labels_01))])
disp(['Train %: ', num2str(misclass_train/length(train_labels_01))])

%% Task E6
model = fitcsvm(train_data_01', train_labels_01);
y_pred_train = predict(model, train_data_01');
y_pred_test = predict(model, test_data_01');

misclass_train = nnz(y_pred_train - train_labels_01);
misclass_test = nnz(y_pred_test - test_labels_01);
disp(['Test: ', num2str(misclass_test),' Train: ', num2str(misclass_train)])
disp(['Test %: ', num2str(misclass_test/length(test_labels_01))])
disp(['Train %: ', num2str(misclass_train/length(train_labels_01))])
%%
length(find(test_labels_01(find(y_pred_test == 1))==1))
%% Task E7
beta = 1;
model_gauss = fitcsvm(train_data_01', train_labels_01, ...
                        'KernelFunction', 'gaussian', 'KernelScale', beta);

y_pred_train = predict(model_gauss, train_data_01');
y_pred_test = predict(model_gauss, test_data_01');

misclass_train = nnz(y_pred_train - train_labels_01);
misclass_test = nnz(y_pred_test - test_labels_01);
disp(['Test: ', num2str(misclass_test),' Train: ', num2str(misclass_train)])
disp(['Test %: ', num2str(misclass_test/length(test_labels_01))])
disp(['Train %: ', num2str(misclass_train/length(train_labels_01))])
%%
length(find(test_labels_01(find(y_pred_test == 0))==0))