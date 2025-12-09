tic
startTime = tic;
numMC = 1000;
% outputs = zeros(numMC, length(selectedIndexes_output), length_ecg);% Output: (n, number of predicted leads, signal length)
inputList = cell(length(testDataIN), numSelected + length(selectedIndexes_input));
numTestdata_plot = length(testDataIN);
numPredicted = size(testDataOUT{1}, 1);
length_ecg = size(testDataOUT{1}, 2);
eps = 1e-3;
for idx = 1 : numTestdata_plot
    ecg = testDataIN{idx};  % [1×512]
    dl_ecg = dlarray(ecg, 'CTB');
    inputList{idx, 1} = dl_ecg;
    for j = 1 : numSelected
        currentFolderName = selectedFolders{j};
        dl_feat = dlarray(testData_one.(currentFolderName){idx}, 'CTB');
        inputList{idx, j + length(selectedIndexes_input)} = dl_feat;
    end
end

std_all = zeros(numTestdata_plot, length(selectedIndexes_output), length_ecg);
mean_all = zeros(numTestdata_plot, length(selectedIndexes_output), length_ecg);
cv_all = zeros(numTestdata_plot, length(selectedIndexes_output), length_ecg);
RMS_all = zeros(numTestdata_plot, length(selectedIndexes_output), length_ecg);
for n = 1 : numTestdata_plot

    input1 = inputList{n, 1};
    input2 = inputList{n, 2};
    input3 = inputList{n, 3};
    input4 = inputList{n, 4};
    input5 = inputList{n, 5};
    input6 = inputList{n, 6};
    outputs = zeros(numMC, length(selectedIndexes_output), length_ecg);% Output: (n, number of predicted leads, signal length)

    parfor m = 1 : numMC
        dlY = forward(net, input1, input2, input3, input4, input5, input6);
        outputs(m, :, :) = extractdata(dlY);  % [11×512]
    end

    % Compute mean prediction and standard deviation
    outputs_mean = squeeze(mean(outputs, 1));  % [leads × 512]
    outputs_std  = squeeze(std(outputs, 0, 1));% [leads × 512]

    % % Compute local RMS
    % w = 15; 
    % localRMS = sqrt(movmean(outputs_mean.^2, w, 2)); % [leads × 512]

    % % CV = sigma / RMS
    % CV = outputs_std ./ (localRMS + eps);

    % % Save
    % RMS_all(n,:,:) = localRMS;
    mean_all(n,:,:) = outputs_mean;
    std_all(n,:,:)  = outputs_std;
    % cv_all(n,:,:)   = CV;

    if mod(n, 10) == 0
        elapsed = toc(startTime);
        fprintf('現在の進捗状況：%d / %d、経過時間：%.2f 秒\n', n, numTestdata_plot, elapsed);
    end
end
% path_CP_save = 'your_directory_here';
% if ~exist(path_CP_save, 'dir')
% % Create folder if it does not exist
% mkdir(path_CP_save);
% end
% save([path_CP_save, '\std_all.mat'],"std_all");
%%
featInput = struct();
rel_error_map = zeros(numTestdata_plot, numPredicted, length_ecg);
abs_error_map = zeros(numTestdata_plot, numPredicted, length_ecg);
for ii =1 : numTestdata_plot
    for j = 1 : numSelected
        currentFolderName = selectedFolders{j};
        featInput.(currentFolderName) = dlarray(testData_one.(currentFolderName){ii}, 'CT');
        featInput_extract = struct2cell(featInput);
    end
end
% Create data for true values
true_vals = zeros(numTestdata_plot, numPredicted, length_ecg);
for ii = 1 :numTestdata_plot
    true_vals(ii, :, :) = testDataOUT{ii}(:, :);
end

predicted_values = zeros(numTestdata_plot, numPredicted, length_ecg);
for ii = 1 :numTestdata_plot
    ecgInput = dlarray(testDataIN{ii}, 'CT');
    predicted_values(ii, :, :) = extractdata(predict(net, ecgInput, featInput_extract{:}));
    for k = 1:numPredicted
        % Get true and predicted values, then compute errors
         rel_error_map(ii, k, :) = abs(squeeze(predicted_values(ii, k, :))' - squeeze(true_vals(ii, k, :))'./ ...
                                                                            (squeeze(true_vals(ii, k, :))' + eps));
        abs_error_map(ii, k, :) = abs(squeeze(predicted_values(ii, k, :))' - squeeze(true_vals(ii, k, :))');
    end
end
%% Metric computation and visualization (Spread-Error Plot)

% 1. Vectorize data (3D -> 1D)
% Reshape [N x C x T] data into a single vector
vec_errors_MC = abs_error_map(:);
vec_stds_MC = std_all(:);
% Alpha scaling
alpha_opt_MC = sqrt(mean((vec_errors_MC ./ vec_stds_MC).^2));
vec_stds_scaled_MC = vec_stds_MC * alpha_opt_MC;

% Binning
num_bins = 60;
[sorted_std_MC, sort_idx] = sort(vec_stds_scaled_MC); % Sort as vectors
sorted_err_MC = vec_errors_MC(sort_idx);

% Equal-count binning
samples_per_bin = floor(length(sorted_std_MC) / num_bins);
bin_spread_MC = zeros(num_bins, 1);
bin_rmse_MC = zeros(num_bins, 1);

for b = 1:num_bins
    idx_s = (b-1)*samples_per_bin + 1;
    idx_e = b*samples_per_bin;
    if b == num_bins, idx_e = length(sorted_std_MC); end
    
    % Compute mean predictive uncertainty and RMS error within each bin
    bin_spread_MC(b) = mean(sorted_std_MC(idx_s:idx_e));
    bin_rmse_MC(b) = sqrt(mean(sorted_err_MC(idx_s:idx_e).^2));
end

% --- Plot spread–error curve ---
plot(bin_spread_MC, bin_rmse_MC, 'o-', 'LineWidth', 2);
hold on;

% Plot ideal line (y = x)
max_val = max([max(bin_spread_MC), max(bin_rmse_MC)]);
plot([0, max_val], [0, max_val], 'r--', 'DisplayName', 'Ideal');

xlabel('Predictive Uncertainty (Std)');
% Change the y-axis label depending on which error metric you use
ylabel('Empirical Error (RMSE)'); 
title({'Spread-Error Plot', ['Spearman Corr: ' num2str(spearman_corr, '%.3f')]});
legend('Model Performance', 'Ideal (y=x)', 'Location', 'best');
% 
% % --- Histogram of the distribution ---
hold off;

figure;
histogram(vec_stds_MC, 50); % Use the vectorized data for the histogram as well
title('Distribution of Predicted Std');
xlabel('Standard Deviation');
ylabel('Count');
%% Scatter plot (Absolute Error vs Uncertainty)

std_all2_norm = zeros(numTestdata_plot, length_ecg);
numPlots = 1;
R = zeros(1, numPlots);

tf_ci = false;
ci = 99.999; % Confidence interval [%]
alpha = 1 - ci/ 100;
z_score = norminv(1 - alpha/2);

all_stds_DE_scaled = std_all * alpha_opt_MC;

for i = 1 : numPlots
    figure;

    std_all_max = max(all_stds_DE_scaled(:, i, :), [], 3);
    for j = 1 : numTestdata_plot
        std_all2_norm(j, :) = all_stds_DE_scaled(j, i, :) / std_all_max(j);
    end
    % Convert to vectors
    x = std_all2_norm;

    y = abs_error_map(:, i, :) ;
    y = squeeze(y);

    mu_y = mean(y, 'omitnan');
    sigma_y = std(y, 'omitnan');
    lower_bound = mu_y - z_score * sigma_y;
    upper_bound = mu_y + z_score * sigma_y;

    if tf_ci
        valid_idx = (y >= lower_bound) & (y <= upper_bound) & isfinite(y) & isfinite(x);
        x = x(valid_idx);
        y = y(valid_idx);
    end

    % Vectorize multi-dimensional data
    x = x(:);
    y = y(:);

    % Power-law fitting: y = a * x^b
    ftype = fittype('a*x^b', 'independent','x','coefficients',{'a','b'});
    opts = fitoptions('Method','NonlinearLeastSquares'); % Initial values a=1, b=1
    [powerFit, gof] = fit(x, y, ftype, opts);
    
    % Generate data for fitted curve
    x_fit_line = linspace(min(x), max(x), 200);
    y_fit_line = powerFit.a * (x_fit_line.^powerFit.b);

    % Plot
    scatter(downsample(x, 5), downsample(y, 5), 7,'filled', 'MarkerFaceAlpha', 0.7); hold on;
    plot(x_fit_line, y_fit_line, 'r-', 'LineWidth', 2)

    % Legend
    lgd = legend('data points', ...
        sprintf('Power Approximation (R = %.3f)', ...
        sqrt(gof.rsquare)));
    lgd.FontName = 'Times New Roman';
    lgd.FontSize = 12;
    lgd.Color = 'w';
    lgd.Location = 'northeast';

    ax = gca;
    ax.FontName =  'Times New Roman';
    ax.FontSize = 12;
    xlim([0 1]);
    ymax = max(downsample(y, 20), [], 'all') * 1.05;
    ylim([0 ymax]);

    xlabel('Normalized STD', 'FontName','Times New Roman', 'FontSize', 14);
    ylabel('Absolute Error', 'FontName','Times New Roman', 'FontSize', 14);
    % title(sprintf('Lead %s', names{1, i + 1}), 'FontName','Times New Roman', 'FontSize', 14);
    box on

    filename = 'your_directory_here/scatter_MC.emf';
    exportgraphics(gcf, filename, 'Resolution', 300);
    R(1, i) = sqrt(gof.rsquare);
    fprintf('Lead %s, R = %.3f', names{1, i + 1}, sqrt(gof.rsquare));
end
%% Scatter plot (Relative Error vs Uncertainty)

std_all2_norm = zeros(numTestdata_plot, length_ecg);
R = zeros(1, numPlots);
numPlots = 11;
for i = 1 : numPlots
    figure;

    std_all_max = max(std_all(:, i, :), [], 3);
    for j = 1 : numTestdata_plot
        std_all2_norm(j, :) = std_all(j, i, :) / std_all_max(j);
    end
    % Convert to vector
    x = std_all2_norm;

    y = rel_error_map(:, i, :) ;
    y = squeeze(y);

    % Vectorize (because of multi-dimension)
    x = x(:);
    y = y(:);

    % Power approximation: y = a * x^b
    ftype = fittype('a*x^b', 'independent','x','coefficients',{'a','b'});
    opts = fitoptions('Method','NonlinearLeastSquares',...
              'StartPoint',[1 1]); % Initial values: a = 1, b = 1
    [powerFit, gof] = fit(x, y, ftype, opts);
    
    % Data for fitted curve
    x_fit_line = linspace(min(x), max(x), 200);
    y_fit_line = powerFit.a * (x_fit_line.^powerFit.b);

    % Plot
    scatter(downsample(x, 15), downsample(y, 15), 7,'filled', 'MarkerFaceAlpha', 0.7); hold on;
    plot(x_fit_line, y_fit_line, 'r-', 'LineWidth', 2)

    % Display
    lgd = legend('data points', ...
        sprintf('Power Approximation (R = %.3f)', ...
        sqrt(gof.rsquare)));
    lgd.FontName = 'Times New Roman';
    lgd.FontSize = 12;
    lgd.Color = 'w';
    lgd.Location = 'northeast';

    ax = gca;
    ax.FontName =  'Times New Roman';
    ax.FontSize = 12;
    xlim([0 1]);
    ymax = max(downsample(y, 20), [], 'all') * 1.05;
    ylim([0 2500]);

    xlabel('Normalized STD', 'FontName','Times New Roman', 'FontSize', 14);
    ylabel('Relative Error', 'FontName','Times New Roman', 'FontSize', 14);
    % title(sprintf('Lead %s', names{1, i + 1}), 'FontName','Times New Roman', 'FontSize', 14);
    box on

    filename = sprintf('your_path/figure3000_%02d_progress.emf', i); % path generalized
    exportgraphics(gcf, filename, 'Resolution', 300);
    R(1, i) = sqrt(gof.rsquare);
    fprintf('Lead %s, R = %.3f', names{1, i + 1}, sqrt(gof.rsquare));
end
%%
% Setting time axis
% t = 1:length(testDataIN{1});
max_view = length(testDataIN{1});
max_view = 120;
fs = 100;                % Sampling frequency [Hz]
dt = 1/fs;
t = (0:max_view-1) * dt; % Convert to seconds

alphaV = 0.50;              % Opacity of heatmap (0–1)
% Background color of axes
bg_color = [0.95 0.95 0.95];


% -------- Create colormap (0–0.2: background color, 0.2–1.0: blue→green→red) --------
n = 256; % Resolution of colormap
rgb_map = zeros(n,3);

rate_threshold = 0.2;

for i = 1:n
    rate = (i-1)/(n-1);  % Between 0 and 1

    if rate <= rate_threshold
        % Set background color for 0–0.2 range (makes it look transparent)
        rgb_map(i,:) = bg_color;
    else
        % Remap 0.2–1.0 range to 0–1 and apply gradient
        rate_norm = (rate - rate_threshold) / (1 - rate_threshold); % 0.2 -> 0, 1.0 -> 1

        if rate_norm < 0.5
            % Blue → Green
            ratio = rate_norm * 2;
            rgb_map(i,:) = (1 - ratio) * [0, 0, 1] + ratio * [0, 1, 0];
        else
            % Green → Red
            ratio = (rate_norm - 0.5) * 2;
            rgb_map(i,:) = (1 - ratio) * [0, 1, 0] + ratio * [1, 0, 0];
        end
    end
end
% --------------------------------------------------------------------------


for ind = [5,1,11,17]
    for i = 1:numPlots
        % Get indices for YTest and YPred
        % Retrieve ground truth
        YTest = testDataOUT(ind);
        y_true  = YTest{1};
        y_pred  = predicted_values(ind, i, :);
        y_pred = squeeze(y_pred)';

        for j = 1 : numTestdata_plot
            std_all2_norm(j, :) = std_all(j, i, :) / std_all_max(j);
        end

        cm = std_all2_norm(ind, :);
    
        % % % Display range (slightly expanded for clear waveform visualization)
        ymin = min([y_pred(:, 1:max_view), y_true(i, 1:max_view)]) * 1.05;
        ymax = max([y_pred(:, 1:max_view), y_true(i, 1:max_view)]) * 1.05;
        % 
        % for ii = 1:length_ecg
        %     mask = abs(y_true(i,:)) <= 20;
        %     cv(mask) = NaN;
        % end
    
        C  = [cm(1:max_view); cm(1:max_view)];          % 2 x |idx|

        A_base = double(~isnan(C)); % NaN → transparent (0)
        
        % Mask values ≤ 0.2 and set alpha to 0 for those regions
        mask_opaque = C > 0.2; 
        A = A_base .* mask_opaque * alphaV;
    
        % ------- Plot -------
        figure; hold on;
        set(gcf, 'Position', [100, 100, 800, 400]); 

        hImg = imagesc(t, [ymin ymax], C);
        set(hImg, 'AlphaData', A);            % Low amplitude = transparent
        set(gca,'Color',[0.95 0.95 0.95]);    
        colormap(rgb_map);
        cb = colorbar; cb.Label.String = 'Normalized STD'; cb.FontSize = 12; cb.FontName = 'Times New Roman';
        clim([0 1]);
    
        % Overlay predicted and reference waveforms
        plot(t, y_pred(:, 1:max_view), 'k-', 'LineWidth', 1.4);                 
        plot(t, y_true(i, 1:max_view), 'Color', [0.3 0.3 0.3], 'LineWidth', 1);
    
        xlim([0 max(t)]);
        ylim([ymin ymax]);
        xlabel('Time (sec)');
        ylabel('Amplitude [µV]');
    
        ax = gca; ax.FontName =  'Times New Roman'; ax.FontSize = 12;
    
        lgd = legend('Predicted ECG','Reference ECG','Location','best');
        lgd.Color = 'w';
        lgd.FontSize = 12;
        lgd.FontName = 'Times new roman';
        lgd.Location = 'northeast';
    
        % title(sprintf('Lead %s', names{1, i + 1}), 'FontName','Times New Roman', 'FontSize', 14);
        box on; hold off;
        fprintf('ind:%d',ind)
        % filename = sprintf('your_path/HeatMap_Lead2_120_sample%02d.emf', ind); % generalized
        % exportgraphics(gcf, filename, 'Resolution', 300);
    end
end
