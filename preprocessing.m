%% 초기 설정
fs = 125;
window_size = fs * 8;   % 1000
stride = fs * 2;        % 250

[b, a] = butter(4, [0.5 8]/(fs/2), 'bandpass');

%% Part_1 ~ Part_4 반복
for p = 1:4
    fprintf('\n Part_%d 전처리 시작...\n', p);

    % 초기화
    temp_segments = {};
    temp_labels = [];

    var_name = sprintf('Part_%d', p);
    if ~exist(var_name, 'var')
        warning('%s 변수 없음.', var_name);
        continue;
    end
    part = eval(var_name);

    for j = 1:length(part)
        raw = part{j};
        if isempty(raw) || size(raw,1) < 2, continue; end

        ppg_raw = raw(1, :);
        abp_raw = raw(2, :);

        if any(~isfinite(ppg_raw)) || length(ppg_raw) < window_size, continue; end
        if any(~isfinite(abp_raw)) || length(abp_raw) < window_size, continue; end

        % NaN/Inf 보정
        ppg_raw(~isfinite(ppg_raw)) = 0;
        abp_raw(~isfinite(abp_raw)) = 0;

        % 필터링
        ppg = filtfilt(b, a, ppg_raw);

        % std 0 제거
        if std(ppg) == 0 || std(abp_raw) == 0, continue; end

        % 정규화 + 단일 정밀도(single)
        ppg = single((ppg - mean(ppg)) / std(ppg));
        abp = single((abp_raw - mean(abp_raw)) / std(abp_raw));

        % 슬라이딩 윈도우
        for i = 1:stride:(length(ppg) - window_size)
            temp_segments{end+1} = ppg(i:i+window_size-1);
            temp_labels(end+1) = mean(abp(i:i+window_size-1));
        end
    end

    % ➜ 2D 배열로 변환: [window_size × N]
    N = length(temp_segments);
    X = zeros(window_size, N, 'single');
    for i = 1:N
        X(:, i) = temp_segments{i};
    end
    Y = single(temp_labels(:));

    % 저장 (압축 효율을 위해 변수명도 최소화)
    filename = sprintf('ppg_abp_compact_p%d.mat', p);
    save(filename, 'X', 'Y', '-v7.3');
    fprintf('저장 완료: %s (샘플 수: %d)\n', filename, N);
end


