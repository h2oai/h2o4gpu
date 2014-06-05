function run_bench

% Add ../pogs
addpath('..')

% Setup data.
rho = exp(linspace(log(0.01), log(1000), 20));
dim_small = [200, 200,  200,  200,  200]';
dim_large = [300, 500, 1000, 2000, 5000]'; 

params.quiet = true;
params.MAXITR = 2000;
params.RELTOL = 1e-3;
params.cmp_cvx = true;

% Setup Figures
h_iter = figure;
h_err = figure;
h_viol = figure;
set(h_iter, 'Visible', 'off');
set(h_err, 'Visible', 'off');
set(h_viol, 'Visible', 'off');

% Run LP-Eq
disp('Running Equality LP')
dims = [dim_small, dim_large];
[n_iter, err, max_viol] = bench_lp_eq(dims, rho, params);
do_plot(231, h_iter, h_err, h_viol, n_iter, err, max_viol, dim_large, ...
        rho, 'n', 'Equality LP')
fprintf('NumIter: %d, Error: %e\n', n_iter, err);

% Run LP-InEq
disp('Running Inequality LP')
dims = [dim_large, dim_small];
[n_iter, err, max_viol] = bench_lp_ineq(dims, rho, params);
do_plot(232, h_iter, h_err, h_viol, n_iter, err, max_viol, dim_large, ...
        rho, 'm', 'Inequality LP')
fprintf('NumIter: %d, Error: %e\n', n_iter, err);

% Run NonNeg-l2
disp('Running Non-Negative L2')
dims = [dim_large, dim_small];
[n_iter, err, max_viol] = bench_nonneg_l2(dims, rho, params);
do_plot(233, h_iter, h_err, h_viol, n_iter, err, max_viol, dim_large, ...
        rho, 'm', 'Non-Negative Least Squares')
fprintf('NumIter: %d, Error: %e\n', n_iter, err)

% Run SVM
disp('Running SVM')
dims = [dim_large, dim_small];
[n_iter, err, max_viol] = bench_svm(dims, rho, params);
do_plot(234, h_iter, h_err, h_viol, n_iter, err, max_viol, dim_large, ...
        rho, 'm', 'SVM')
fprintf('NumIter: %d, Error: %e\n', n_iter, err)

% Run Lasso
disp('Running Lasso')
dims = [dim_large, dim_small];
[n_iter, err, max_viol] = bench_lasso(dims, rho, params);
do_plot(235, h_iter, h_err, h_viol, n_iter, err, max_viol, dim_large, ...
        rho, 'm', 'Lasso')
fprintf('NumIter: %d, Error: %e\n', n_iter, err)

% Save to PDF
datenow = datestr(now, 'dd-mm-hhMM');
set(h_iter, 'PaperPosition', [0 0 15 10]);
set(h_iter, 'PaperSize', [15 10]);
saveas(h_iter, [datenow '-NumIter'], 'pdf')

set(h_err, 'PaperPosition', [0 0 15 10]);
set(h_err, 'PaperSize', [15 10]);
saveas(h_err, [datenow '-Error'], 'pdf')

set(h_viol, 'PaperPosition', [0 0 15 10]);
set(h_viol, 'PaperSize', [15 10]);
saveas(h_viol, [datenow '-MaxViolation'], 'pdf')

end

function do_plot(subplot_id, h_iter, h_err, h_viol, n_iter, err, ...
                 max_violation, dim_large, rho, dim_text, title_text)

plot_cols = ggplot_colors(size(n_iter, 2));

legend_text = cell(size(n_iter, 2), 1);
for i = 1:size(n_iter, 2)
  legend_text{i} = sprintf('%s = %d', dim_text, dim_large(i));
end

figure(h_iter)
subplot(subplot_id)
set(gca, 'ColorOrder', plot_cols);
set(gca, 'NextPlot',' replacechildren')
plot(rho, n_iter)
set(gca, 'XScale', 'log')
xlabel('rho')
ylabel('Num Iter')
title(title_text)
h_legend = legend(legend_text);
set(h_legend, 'FontSize', 10);

figure(h_err)
subplot(subplot_id)
set(gca, 'ColorOrder', plot_cols);
set(gca, 'NextPlot',' replacechildren')
plot(rho, err)
set(gca, 'XScale', 'log', 'YScale', 'log')
xlabel('rho')
ylabel('Error')
title(title_text)
h_legend = legend(legend_text);
set(h_legend, 'FontSize', 10);

figure(h_viol)
subplot(subplot_id)
set(gca, 'ColorOrder', plot_cols);
set(gca, 'NextPlot',' replacechildren')
plot(rho, max_violation)
set(gca, 'XScale', 'log', 'YScale', 'log')
xlabel('rho')
ylabel('Max Violation')
title(title_text)
h_legend = legend(legend_text);
set(h_legend, 'FontSize', 10);

end

