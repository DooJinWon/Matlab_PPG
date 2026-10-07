%% mmHg 복원 & 지표 계산  (YPred·YTest → mmHg)
% ─────────────── 사용 전 확인 ───────────────
% 1) raw_ABP_cell : {cell}  →  각 원소 = 원본 ABP 파형(mmHg) 벡터
% 2) YPred , YTest : z-score 스케일 예측·실측 (열벡터 or 행벡터)
%% ① Part_1~Part_4 불러와 ABP(mmHg) 모으기
raw_ABP_cell = {};            % 초기화

for p = 1:4
    fileName = sprintf('Part_%d.mat', p);
    if ~isfile(fileName), warning('%s 없음 – 건너뜀',fileName); continue; end

    S = load(fileName);                 % 구조체로 로드
    varName = sprintf('Part_%d', p);
    part    = S.(varName);              % 셀 배열

    for k = 1:numel(part)
        rec = part{k};
        if isempty(rec) || size(rec,1) < 2, continue; end
        abp_mmHg = rec(2,:);            % 2행 = ABP(mmHg)
        raw_ABP_cell{end+1} = single(abp_mmHg(:));  % 열벡터 저장
    end
end

fprintf("ABP 구간 수 : %d\n", numel(raw_ABP_cell));
if exist('raw_ABP_cell','var') ~= 1
    error('raw_ABP_cell 변수가 없습니다.  원본 ABP(mmHg) 파형을 셀 배열로 준비하세요.');
end
if ~exist('YPred','var') || ~exist('YTest','var')
    error('YPred · YTest 벡터가 없어 mmHg 복원을 수행할 수 없습니다.');
end

%% 1. μ·σ 계산 (NaN/Inf 제거)
abp_raw_all = single(vertcat(raw_ABP_cell{:}));
abp_raw_all = abp_raw_all(isfinite(abp_raw_all));   % NaN·Inf 제거

abp_mu  = mean(abp_raw_all);
abp_std = std( abp_raw_all);

if ~isfinite(abp_mu) || ~isfinite(abp_std) || abp_std==0
    error('원본 ABP 데이터의 μ/σ 계산에 실패했습니다. raw_ABP_cell 값을 확인하세요.');
end

%% 2. 복원 (z-score → mmHg)
YPred_mm = YPred(:) * abp_std + abp_mu;
YTest_mm = YTest(:) * abp_std + abp_mu;

%% 3. 지표
rmse_mm = sqrt(mean((YPred_mm - YTest_mm).^2));
mae_mm  = mean(abs(YPred_mm - YTest_mm));
corr_mm = corr(YPred_mm, YTest_mm);      % 스칼라

fprintf("\n=== mmHg 단위 성능 ===\n");
fprintf("RMSE : %.2f mmHg\n", rmse_mm);
fprintf("MAE  : %.2f mmHg\n", mae_mm);
fprintf("Corr : %.4f\n",     corr_mm);

%% 4. 그래프 (실측 vs 예측)
figure;
subplot(2,1,1);
plot(YTest_mm,'k--'); hold on; plot(YPred_mm,'r');
grid on; legend('True mmHg','Pred mmHg');
title(sprintf('ABP 예측 (RMSE=%.2f mmHg, Corr=%.4f)', rmse_mm, corr_mm));
xlabel('Sample'); ylabel('ABP (mmHg)');

subplot(2,1,2);
scatter(YTest_mm, YPred_mm, 6,'filled'); grid on; axis equal;
hold on; plot(xlim, xlim, 'k--');        % 45° 기준선
xlabel('True mmHg'); ylabel('Pred mmHg');
title('Parity Plot (mmHg)');
