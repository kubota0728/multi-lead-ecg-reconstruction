numPredicted = size(testDataOUT{1});
numPredicted = numPredicted(1);
correlation_coefficients = zeros(length(testDataIN), numPredicted);
rmse_values = zeros(length(testDataIN), numPredicted);
mse_values = zeros(length(testDataIN), numPredicted);

featInput = struct();
featInput_extract = cell(numTestdata, 1);
for i = 1:length(testDataIN)
    for j = 1 : numSelected
        currentFolderName = selectedFolders{j};
        featInput.(currentFolderName) = dlarray(testData_one.(currentFolderName){i}, 'CT');
        featInput_extract{ii} = struct2cell(featInput);
    end
end

for i = 1:length(testDataIN)
    ecgInput = dlarray(testDataIN{i}, 'CT');
    predicted_values = extractdata(predict(net, ecgInput, featInput_extract{:}));
    for k = 1:numPredicted

        % Obtain the ground-truth and predicted values
        true_values_n = testDataOUT{i}(k,:);
         
        % Compute the correlation coefficient
        correlation_coefficients_2 = corrcoef(true_values_n, predicted_values(k,:));
        correlation_coefficients(i, k) = correlation_coefficients_2(1,2);
         
        % Compute the RMSE
        rmse_values(i, k) = sqrt(mean((true_values_n - predicted_values(k,:)).^2))*10^(-3);

        % % Compute the MSE
        % mse_values(i, j) = mean((true_values_n - predicted_values(j,:)).^2);
    end
end



% Compute the average correlation coefficient, RMSE, and MSE
aveCorr = mean(correlation_coefficients, 1);
aveRMSE = mean(rmse_values, 1);
% aveMSE = mean(mse_values, 1);

disp(aveCorr)
disp(mean(aveCorr))
disp(mean(aveRMSE))
% disp(mean(aveMSE))

% nanCol = NaN(size(testDataOUT,1),length(selectedIndexes_input));
% insertCol = selectedIndexes_input;
% correlation_coefficients_new = [correlation_coefficients(:, 1:insertCol-1), nanCol, correlation_coefficients(:, insertCol:end)];
% [nSamples, nSignals] = size(correlation_coefficients_new);
% 
% meanCorr = mean(correlation_coefficients_new, 1);
% z = atanh(correlation_coefficients_new);  % Fisher Z-transformation
% z_mean = mean(z, 1);
% z_std = std(z, 0, 1);  % Standard deviation in Z-score space
% 
% % Standard error (SE) and the 95% confidence interval (in Z-score space)
% se = z_std / sqrt(nSamples);  
% z_ci_low = z_mean - 1.96 * se;
% z_ci_high = z_mean + 1.96 * se;
% 
% % Inverse transform back to the correlation coefficient (r)
% r_ci_low = tanh(z_ci_low);
% r_ci_high = tanh(z_ci_high);
% r_error = [meanCorr - r_ci_low; r_ci_high - meanCorr];
% 
% % Plot
% figure;
% plot(1:nSignals, meanCorr, r_error(1,:), r_error(2,:), 'o-', 'LineWidth', 1.5);
% xlim([1 12])
% % ylim([0 1]);
% xlabel('Signal Index');
% ylabel('Mean Correlation Coefficient');
% grid on;
% 
% path_out = ['your_directory_here\' ...
%             'input_',num2str(names{selectedIndexes_input})];
% 
% if ~exist(path_out, 'dir')
%     % Create the folder if it does not exist
%     mkdir(path_out);
% end
% saveas(gcf, [path_out,'\correlation_plot.fig']); % Save as MATLAB .fig file
% save([path_out, '\aveCorr.mat'], "aveCorr"); % Save the correlation coefficients

% % Standard deviation
% stdCorr = std(correlation_coefficients, 0, 1);
% 
% % Plot with error bars
% figure;
% errorbar(1:numPredicted, aveCorr, stdCorr, 'o-', 'LineWidth', 1.5, 'CapSize', 8);
% grid on;
% xlabel('Prediction Index');
% ylabel('Correlation Coefficient');
% % ylim([0 1]);  % For visualization clarity when coefficients range from 0–1

% % Plot boxplot of correlation coefficients for each column
% figure;
% boxplot(correlation_coefficients);
% title('Correlation Coefficients Boxplot');
% xlabel('Prediction Number');
% ylabel('Correlation Coefficient');

% % Plot boxplot of RMSE for each column
% figure;
% disp(std(correlation_coefficients));
% boxplot(rmse_values);
% title('RMSE Boxplot');
% xlabel('Prediction Number');
% ylabel('RMSE');

% % Plot boxplot of MSE for each column
% figure;
% boxplot(mse_values);
% title('MSE Boxplot');
% xlabel('Prediction Number');
% ylabel('MSE');

% correlation_coefficients: assumed to be a matrix of [subjects × signals]
