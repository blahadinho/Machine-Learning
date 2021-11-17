%% Task 4
lambda = 10;
what = skeleton_lasso_ccd(t, X, lambda);

plot(n, t, 'x')
hold on
plot(n, X*what, 'o')
plot(ninterp, Xinterp*what)
title('LASSO optimization problem with {\lambda} = 10')
xlabel('n')
ylabel('t')
legend('Data points', 'Estimated point', 'Interpolated line')
disp(['Number of weights in what: ' num2str(length(find(what))) ' lambda = ' num2str(lambda) ])
%% Task 5
lambdavec = exp(linspace(log(0.1), log(10), 100));
[wopt,lambdaopt,RMSEval,RMSEest] = skeleton_lasso_cv(t, X, lambdavec, 5);

figure(1)
plot(lambdavec, RMSEval, '-x')
hold on
plot(lambdavec, RMSEest, '-*')
xline(lambdaopt, '--r');
xlabel('{\lambda}')
ylabel('RMSE')
title('RMSE for different lambdas')
legend('Valitation data', 'Estimation data', 'Optimal {\lambda}')

what = skeleton_lasso_ccd(t, X, lambdaopt);
%% Task 5
figure(2)
plot(n, t, 'x')
hold on
plot(n, X*what, 'o')
plot(ninterp, Xinterp*what)
title('LASSO optimization problem with {\lambda} = 1.8738')
xlabel('n')
ylabel('t')
legend('Data points', 'Estimated point', 'Interpolated line')

%% Task 6
lambdavec = exp(linspace(log(0.0001), log(0.1), 100));
[Wopt,lambdaopt,RMSEval,RMSEest] = skeleton_multiframe_lasso_cv(Ttrain,Xaudio,lambdavec,5);
%% Task 6
plot(lambdavec, RMSEval, '-x')
hold on
plot(lambdavec, RMSEest, '-*')
xline(lambdaopt, '--r');
xlabel('{\lambda}')
ylabel('RMSE')
title('{\lambda_{opt}} = 0.0043')
suptitle('RMSE for different {\lambda}')
legend('Valitation data', 'Estimation data', 'Optimal {\lambda}')
%% Task 7
save('task5','Wopt','lambdaopt','RMSEval','RMSEest')
%% Testing for task 7
Tq = fft(Ttest);
f = (0:length(Tq)-1)*fs/length(Tq);
Yq = fft(Yclean);
subplot(211)
plot(f,abs(Tq))
subplot(212)
plot(f,abs(Yq))
%% Play audio
soundsc(Ttest, fs)
pause(5)
soundsc(Yclean, fs)
%% Denoise audio
[Yclean] = lasso_denoise(Ttest,X,0.0043);
save('denoised_audio','Yclean','fs')