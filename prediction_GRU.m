%% LSTM 기반 ABP 평균 예측 (전체 데이터)
% temp_segments : {cell} 각 원소는 1차원 신호 벡터
% -----------------------------------------------------
clc; clearvars -except temp_segments;

%% 0. 파라미터 설정 --------------------------------------------------
winLen  = 250;    % 윈도 길이 (2 s @125 Hz)
stride  = 125;    % 50 % overlap
gpuOn   = "gpu";  % 'gpu' | 'auto' | 'cpu'

%% 1. 입력·레이블 생성 -----------------------------------------------
Xall = {};  Yall = [];
for k = 1:numel(temp_segments)
    raw = single(temp_segments{k}(:));
    raw = (raw - mean(raw)) ./ std(raw);   % z-score (채널 1)

    d1  = [0; diff(raw)];                  % 1차 미분 (채널 2)
    d2  = [0; diff(d1)];                   % 2차 미분 (채널 3)

    sig3 = [raw.' ; d1.' ; d2.'];          % [3×T]  (행: 3 채널)

    T = size(sig3,2);
    for ii = 1:stride:(T-winLen+1)
        seg = sig3(:, ii:ii+winLen-1);     % [3×window] 윈도우
        Xall{end+1,1} = seg;               % 셀 배열에 저장
        Yall(end+1,1) = mean(seg(1,:));    % 채널 1(PPG) 평균 → 레이블
    end
end

Xall = Xall(:);
Yall = single(Yall(:));

%% 2. 학습/테스트 분할 (80/20) ---------------------------------------
rng(42);  % 재현성
N = numel(Xall);

if N == 0
    error("데이터가 비어 있습니다. temp_segments에 유효한 데이터가 들어 있는지 확인하세요.");
end

idx    = randperm(N);
Xall   = Xall(idx);
Yall   = Yall(idx);

nTrain = floor(0.8 * N);
XTrain = Xall(1:nTrain);
YTrain = Yall(1:nTrain);
XTest  = Xall(nTrain+1:end);
YTest  = Yall(nTrain+1:end);

fprintf("데이터: %d (train) / %d (test)\n", numel(XTrain), numel(XTest));

%% 3. LSTM 네트워크 정의 ---------------------------------------------
layers = [
    sequenceInputLayer(3, "Normalization", "none")
    gruLayer(128, "OutputMode", "last")   % ← 이 부분만 변경됨
    fullyConnectedLayer(64)
    reluLayer
    fullyConnectedLayer(1)
    regressionLayer
];

valFreq = max(1, floor(nTrain / 64));  % 최소 1 이상 보장

opts = trainingOptions("adam", ...
    "ExecutionEnvironment", gpuOn, ...
    "MaxEpochs",          5, ...
    "MiniBatchSize",      256, ...
    "InitialLearnRate",   1e-3, ...
    "GradientThreshold",  1, ...
    "Shuffle",            "every-epoch", ...
    "Plots",              "training-progress", ...
    "Verbose",            false, ...
    "ValidationData",     {XTest, YTest}, ...
    "ValidationFrequency",valFreq, ...
    "LearnRateSchedule",  "piecewise", ...
    "LearnRateDropFactor",0.3, ...
    "LearnRateDropPeriod",20);

%% 4. 학습 -----------------------------------------------------------
net = trainNetwork(XTrain, YTrain, layers, opts);

%% 5. 예측·평가 ------------------------------------------------------
YPred = predict(net, XTest, "MiniBatchSize", 1);

rmse    = sqrt(mean((YPred - YTest).^2));
corrVal = corr(YPred, YTest);

fprintf("\n전체 데이터 결과 (N=%d)\nRMSE : %.4f\nCorr : %.4f\n", N, rmse, corrVal);

figure;
subplot(2,1,1);
plot(YTest, 'k--'); hold on; plot(YPred, 'r'); grid on;
legend('True', 'Predicted');
xlabel('Sample'); ylabel('Norm ABP');
title(sprintf('Prediction (N=%d, RMSE=%.4f, Corr=%.4f)', N, rmse, corrVal));

subplot(2,1,2);
scatter(YTest, YPred, '.'); grid on; lsline;
xlabel('True'); ylabel('Predicted');
title('Parity Plot');

%% 6. 네트워크 저장 (선택) ------------------------------------------
% save('abp_lstm_net.mat', 'net');
