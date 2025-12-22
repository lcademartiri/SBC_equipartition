function [p,pgp,sgd_correction,sgd_edges,history] = sbc_setup_sgd_v10(S,PDF,opts,data_folder)
% SBC_SETUP_SGD_V9 (The "Sawtooth" Annealer)
%
% Logic:
% 1. Search Phase: Starts at base_batch_size (physically derived).
% 2. Convergence: Requires RMS < 1.2 * NoiseFloor for 3 consecutive batches.
% 3. Annealing: On success, Batch x 4, Gain x 0.5.
% 4. Production: Once Batch >= 1e5, freezes Gain to 0 and runs 1e6 steps.
%
% Key Features: Quadratic Governor, Robust SNR, Free Tangential Ghosts.

if nargin < 3, opts = struct(); end

% -------------------- 1. Option Parsing -----------------------
% Dynamics & Control
if isfield(opts,'base_gain'),            sgd_base_gain_default = opts.base_gain; else sgd_base_gain_default = 0.5; end
if isfield(opts,'sgd_smooth_win'),       sgd_smooth_win = opts.sgd_smooth_win; else sgd_smooth_win = 5; end
if isfield(opts,'sgd_cap'),              sgd_cap = opts.sgd_cap; else sgd_cap = 0.003 * S.rp; end 
if isfield(opts,'clip_frac'),            clip_frac = opts.clip_frac; else clip_frac = 0.3; end 
if isfield(opts,'abs_cap_frac'),         abs_cap_frac = opts.abs_cap_frac; else abs_cap_frac = 0.005; end

% Annealing / Convergence criteria
if isfield(opts,'rms_tolerance'),        rms_tolerance = opts.rms_tolerance; else rms_tolerance = 1.2; end 
if isfield(opts,'consecutive_passes'),   required_passes = opts.consecutive_passes; else required_passes = 3; end 
if isfield(opts,'stage1_grace_batches'), stage1_grace_batches = opts.stage1_grace_batches; else stage1_grace_batches = 25; end
if isfield(opts,'stage_grace_batches'),  stage_grace_batches = opts.stage_grace_batches; else stage_grace_batches = 3; end
if isfield(opts,'max_batch_size'),       max_batch_size_limit = opts.max_batch_size; else max_batch_size_limit = 100000; end
if isfield(opts,'max_stage_batches'), max_stage_batches = opts.max_stage_batches; else max_stage_batches = 15; end

% SNR & Gating
if isfield(opts,'snr_target'),           snr_target = opts.snr_target; else snr_target = 5.0; end
if isfield(opts,'n_min'),                n_min = opts.n_min; else n_min = 20; end
if isfield(opts,'use_soft_snr'),         use_soft_snr = opts.use_soft_snr; else use_soft_snr = true; end

% System / IO
if isfield(opts,'debugging'),            debugging = opts.debugging; else debugging = false; end
if isfield(opts,'graphing'),             graphing = opts.graphing; else graphing = true; end
if isfield(opts,'enable_io'),            enable_io = opts.enable_io; else enable_io = true; end
if isfield(opts,'metric_smoothing_param'), metric_smoothing_param = opts.metric_smoothing_param; else metric_smoothing_param = 0.8; end

% -------------------- 2. Derived Physical Params -----------------------
gCS = (1 - S.phi/2) / (1 - S.phi)^3;
diffE = S.esdiff * S.alpha / gCS;
tau_alpha = (S.rp^2) / (6 * diffE);
relaxsteps = ceil(tau_alpha / S.timestep);

% Noise prefactor for phi=0.4 (Crowded system correction)
noise_prefactor = 2; 
min_stage_rms = inf; % Track best performance in current stage

% Base Batch: Tied to relaxation time. 
% Minimum data required to see a statistically independent sample.
base_batch_size = max( ceil(10 * relaxsteps), 2000 );

% -------------------- 3. Filenames & Resume -----------------------
if S.potential==1, potname='lj'; elseif S.potential==2, potname='wca'; else potname='hs'; end
seriesname = 'sgd_v8_anneal';
filenamecorrection = sprintf(['ASYMCORR_',seriesname,'_%s_%.0e_%.0e_%.0f_%.1f_%.1e.mat'],...
    potname,S.rp,S.phi,S.N,S.pot_epsilon/S.kbT,S.pot_sigma);
filestartingconfiguration = sprintf(['START_SBC_%s_%.0e_%.0e_%.0f_%.1f_%.1e.mat'],...
    potname,S.rp,S.phi,S.N,S.pot_epsilon/S.kbT,S.pot_sigma);
if S.bc==1
    filepdfdenom = sprintf('PDFdenom_SBC_%.0e_%.0e_%.0f.mat',S.rp,S.phi,S.N);
elseif S.bc==2
    filepdfdenom = sprintf('PDFdenom_PBCc_%.0e_%.0e_%.0f.mat',S.rp,S.phi,S.N);
elseif S.bc==3
    filepdfdenom = sprintf('PDFdenom_PBCFCC_%.0e_%.0e_%.0f.mat',S.rp,S.phi,S.N);
end

if enable_io && exist(filenamecorrection,'file') && exist(filestartingconfiguration,'file')
    load(filenamecorrection); load(filestartingconfiguration);
    history = [];
    return
end

% -------------------- 4. Helpers -----------------------------------

if S.potential~=0
    H = pot_force(S.potential,S.rc,30000,S.pot_sigma,S.pot_epsilon);
    % clamp = mcdClamp(1e6,normrnd(0,S.stdx,1e6,3),S.esdiff,S.timestep,H(H(:,1) >= 0.8 * S.pot_sigma, :),S.kbT);
    H_interpolant = griddedInterpolant(H(:,1), H(:,2), 'linear', 'nearest');
end

% -------------------- 5. Initial Placement -----------------------
disp('Creating initial FCC-like lattice...');
flag = 1;
if debugging, rng(100); end
while flag==1
    basis=[0,0.7071,0.7071;0.7071,0,0.7071;0.7071,0.7071,0].*(2.01*S.rp);
    maxsteps=2*ceil(((S.br*2)*sqrt(3))/(2*S.rp));
    templist=double(linspace(-maxsteps,maxsteps,2*maxsteps+1)');
    [x1,x2,x3] = meshgrid(templist,templist,templist);
    possiblepositions = x1(:).*basis(1,:)+x2(:).*basis(2,:)+x3(:).*basis(3,:);
    possiblepositions = bsxfun(@plus,possiblepositions,-S.rp*[1,1,1]./vecnorm([1,1,1],2));
    tempnorms = vecnorm(possiblepositions,2,2);
    possiblepositions(tempnorms > (S.br - S.rp), :) = [];
    possiblepositions = possiblepositions(randperm(size(possiblepositions,1)), :);
    p = possiblepositions(1:S.N, :);
    D = pdist(p)';
    if sum(D < (2*S.rp)) == 0, flag = 0; end
end

% -------------------- 6. SGD State Init ---------------------------------
potdepth = 2 * S.rc;
if 2*S.br - 2*potdepth < (10 * S.rp), potdepth = 2*S.br - 10*S.rp; end

sgd_edges = sort((S.br:-0.02*S.rp:S.br - potdepth)');
sgd_bins = numel(sgd_edges) - 1;
sgd_centers = sgd_edges(1:end-1) + diff(sgd_edges)/2;
% --- SPATIAL TAPER MASK (Insert here) ---
% Force correction to 0 at the inner edge to prevent impulsive forces.
% We calculate distance from the Inner Edge (Smallest Radius).
r_inner_edge = min(sgd_centers) - (diff(sgd_edges(1:2))/2); 
taper_width = 1.0 * S.rp; % Ramp over 1 particle radius

taper_mask = zeros(sgd_bins, 1);
for b = 1:sgd_bins
    dist_from_inner = sgd_centers(b) - r_inner_edge;
    % Linear Ramp 0 -> 1
    taper_mask(b) = min(1, max(0, dist_from_inner / taper_width));
end
sgd_correction = zeros(sgd_bins, 1);
F_corr_interp = griddedInterpolant(sgd_centers, sgd_correction, 'linear', 'nearest');

batch_sum_drift = zeros(sgd_bins,1);
batch_sum_drift_sq = zeros(sgd_bins,1);
batch_counts = zeros(sgd_bins,1);
steps_in_batch = 0;

% --- Annealing Variables ---
current_batch_mult = 1;
sgd_batch_size = base_batch_size * current_batch_mult;
sgd_base_gain = sgd_base_gain_default;
sgd_gain = sgd_base_gain;
rms_pass_counter = 0;
batches_in_stage = 0; 
stage_index = 1;
is_frozen_production = false;

% Thermalization check
% tightened thermalization: require 10 * relaxsteps or minimum 2000
min_therm_steps = max( ceil(10 * relaxsteps), 2000 );

% Diagnostics
dens_metric = 0;

% Plotting Init
history = struct('steps',[],'dens_dev',[],'dens_smooth',[],'max_corr',[],'gain',[],...
    'batch_size',[],'fraction_updated',[],'median_snr',[]);

if graphing && S.pot_corr
    ndens.edges = sort((S.br:-0.02*S.rp:(S.br - potdepth))');
    ndens.centers = ndens.edges(1:end-1) + diff(ndens.edges)/2;
    ndens.counts = zeros(numel(ndens.centers),1);
    ndens.vols = (4/3)*pi*(ndens.edges(2:end).^3 - ndens.edges(1:end-1).^3);
    ndens.ndens0 = (S.N / S.bv);

    f_fig = figure('Units','normalized','Position',[0.05 0.05 0.85 0.85]);
    set(gcf,'Color','k');
    ax_dens = subplot(3,2,1);
    ax_conv = subplot(3,2,2);
    ax_ctrl = subplot(3,2,3);
    ax_diag = subplot(3,2,4);
    ax_snr  = subplot(3,2,5);
    axs = [ax_dens, ax_conv, ax_ctrl, ax_diag, ax_snr];
    for a = axs
        set(a,'Color','k','XColor','w','YColor','w','LineWidth',3);
        set(get(a,'Title'),'FontWeight','bold','Color','w');
        set(get(a,'XLabel'),'FontWeight','bold','Color','w');
        set(get(a,'YLabel'),'FontWeight','bold','Color','w');
        set(a,'FontWeight','bold','FontSize',10);
    end
end

% -------------------- 7. Main Loop -----------------------------------------
qs = 0;
thermflag = 0;
r2_uniform = 3/5 * S.br^2;
pgp = p - (2*S.br).*(p ./ (vecnorm(p,2,2) + eps));
reverseStr = '';

if S.pot_corr
    fprintf('Starting SGD V9 (Annealing). Base Batch: %d. Gain: %.2f. Cap: %.2e\n', sgd_batch_size, sgd_base_gain, sgd_cap);
else
    fprintf('Thermalizing structure')
end
% ---- CALCULATION OF DISPLACEMENT LIBRARIES ----
DISP=build_noise_library(S.stdx,1e6);
% ----
while true
    qs = qs + 1;

    prho = vecnorm(p, 2, 2);
    pvers = p ./ (prho + eps);
    idxgp = prho > (S.br - S.rc);

    % --- Thermalization Check ---
    if thermflag == 0
        spread_ratio = mean(prho.^2) / r2_uniform;
        is_expanded = spread_ratio > 0.95;
        is_relaxed = qs > min_therm_steps;
        if is_relaxed && is_expanded
            thermflag = 1; qs = 1;
            disp('--- Thermalization complete ---');
            if ~S.pot_corr
                if enable_io,save([data_folder,'\',filestartingconfiguration], 'p', 'pgp', 'S');end
                return
            end
        end
    end

    % --- Physics: Potentials & Displacement ---
    if S.potential ~= 0
        ptemp = [p; pgp(idxgp,:)];
        all_potdisps = potential_displacements_v2(ptemp, S, H, H_interpolant, 0);
        potdisps = all_potdisps(1:S.N, :);
        potdispsgp = all_potdisps(S.N+1:end, :);
        draw=randi(1e6,[S.N 1]);
        v_rand = DISP(draw, :);
        base_disp = v_rand + potdisps;
    else
        draw=randi(1e6,[S.N 1]);
        v_rand = DISP(draw, :);
        base_disp = v_rand;
        potdispsgp = zeros(sum(idxgp), 3);
    end

    % --- Accumulation (Only after thermalization) ---
    if thermflag == 1
        dr_raw = sum(base_disp .* pvers, 2);
        bin_idx = discretize(prho, sgd_edges);
        valid_mask = bin_idx > 0 & bin_idx <= sgd_bins;
        if any(valid_mask)
            idxs = bin_idx(valid_mask);
            values = dr_raw(valid_mask);
            new_sums = accumarray(idxs, values, [sgd_bins, 1]);
            new_sums_sq = accumarray(idxs, values.^2, [sgd_bins, 1]);
            new_counts = accumarray(idxs, 1, [sgd_bins, 1]);
            batch_sum_drift = batch_sum_drift + new_sums;
            batch_sum_drift_sq = batch_sum_drift_sq + new_sums_sq;
            batch_counts = batch_counts + new_counts;
        end
        if graphing
            [hc, ~] = histcounts(vecnorm(p,2,2), ndens.edges);
            ndens.counts = ndens.counts + hc';
        end

        steps_in_batch = steps_in_batch + 1;

        % --- BATCH PROCESSING TRIGGER ---
        if steps_in_batch >= sgd_batch_size
            
            % A. Stats Processing
            has_data = batch_counts > 0;
            bin_mean = zeros(sgd_bins,1);
            snr = zeros(sgd_bins,1);
            n_i = batch_counts;
            idxs = find(has_data);
            
            for ii = idxs'
                ni = n_i(ii);
                mu = batch_sum_drift(ii) / ni;
                ss = batch_sum_drift_sq(ii);
                
                % Robust SNR Calculation (Prevent Infinite SNR)
                if ni > 1
                    v = max(0, (ss - ni * mu^2) / (ni - 1));
                    se = sqrt(v) / sqrt(ni);
                    if se > 1e-15, s = abs(mu) / se; else, s = 0; end
                else
                    s = 0; 
                end
                bin_mean(ii) = mu;
                snr(ii) = s;
            end
            
            % ===============================
            % B. DENSITY METRIC (v10)
            % ===============================
            
            w_count = steps_in_batch;
            
            curr_ndens = (ndens.counts / max(1,w_count)) ./ ndens.vols;
            dens_residuals = curr_ndens - ndens.ndens0;
            
            dens_metric_raw = sqrt( ...
                sum(ndens.vols .* dens_residuals.^2) / ...
                sum(ndens.vols) ) / ndens.ndens0;

            
            % Smooth metric (same logic as before)
            if dens_metric == 0
                dens_metric = dens_metric_raw;
            else
                dens_metric = (1 - metric_smoothing_param) * dens_metric + ...
                               metric_smoothing_param * dens_metric_raw;
            end
            
            min_stage_rms = min(min_stage_rms, dens_metric);

            
            % ===============================
            % DENSITY NOISE FLOOR (Poisson)
            % ===============================
            
            expected_counts = ndens.ndens0 .* ndens.vols * w_count;
            expected_counts(expected_counts == 0) = inf;
            
            dens_noise_floor = noise_prefactor * ...
                sqrt(mean(1 ./ expected_counts));

            
            % C. Gain Calculation (Quadratic Governor)
            current_max_corr = max(abs(sgd_correction));
            cap_proximity = min(1, current_max_corr / sgd_cap);
            governor_factor = max(0.01, 1 - cap_proximity^2);
            sgd_gain = sgd_base_gain * governor_factor; 
            
            % 5. Record History & Plot BEFORE Reset
            
            % --- A. UPDATE HISTORY STRUCTURE ---
            history.steps(end+1) = qs;
            history.dens_dev(end+1) = dens_metric_raw;
            history.dens_smooth(end+1) = dens_metric;
            history.max_corr(end+1) = max(abs(sgd_correction));
            history.gain(end+1) = sgd_gain;
            history.batch_size(end+1) = sgd_batch_size;
            history.fraction_updated(end+1) = numel(find(has_data)) / max(1, sgd_bins);
            
            % Robust Median SNR (Handle empty case)
            if any(has_data)
                history.median_snr(end+1) = median(snr(has_data));
            else
                history.median_snr(end+1) = 0;
            end          
            
            % E. Correction Update (Only if not frozen)
            bins_to_update = false(sgd_bins,1);
            if ~is_frozen_production
                delta = zeros(sgd_bins,1);
                for ii = 1:sgd_bins
                    if n_i(ii) < n_min
                        lambda = 100;
                        mu_eff = (n_i(ii)/(n_i(ii)+lambda)) * bin_mean(ii);
                        s_eff = snr(ii) * (n_i(ii)/(n_i(ii)+lambda));
                    else
                        mu_eff = bin_mean(ii);
                        s_eff = snr(ii);
                    end
                    if use_soft_snr, per_bin_factor = min(1, s_eff / snr_target);
                    else, per_bin_factor = double(s_eff >= snr_target); end
                    
                    if per_bin_factor > 0
                        eta_i = sgd_gain * per_bin_factor; 
                        % Apply Taper: Forces delta to 0 at inner boundary
                        delta(ii) = -eta_i * mu_eff * taper_mask(ii);
                        bins_to_update(ii) = true;
                    end
                end
                
                min_clip_floor = 5e-2 * sgd_cap; 
                max_current_corr = max(max(abs(sgd_correction)), min_clip_floor);
                max_delta_rel = clip_frac * max_current_corr;
                abs_cap = abs_cap_frac * S.br;
                
                bins_updated = find(bins_to_update);
                for ii = bins_updated'
                    d = delta(ii);
                    d = sign(d) * min(abs(d), max_delta_rel);
                    d = sign(d) * min(abs(d), abs_cap);
                    sgd_correction(ii) = sgd_correction(ii) + d;
                end
                
                sgd_correction = max(min(sgd_correction, sgd_cap), -sgd_cap);
                sgd_correction = smoothdata(sgd_correction, 'movmean', sgd_smooth_win);
                F_corr_interp.Values = sgd_correction;
            end
            
            % F. Adaptive Annealing Logic
            batches_in_stage = batches_in_stage + 1;

            % Select the appropriate grace period
            if stage_index == 1
                current_grace_limit = stage1_grace_batches; % Use the long warmup for construction
            else
                current_grace_limit = stage_grace_batches;  % Use short warmup for refinement
            end
            
            % 2. Standard RMS Check (FIXED: Uses current_grace_limit)
            if batches_in_stage > current_grace_limit
                if dens_metric < (rms_tolerance * dens_noise_floor)
                    rms_pass_counter = rms_pass_counter + 1;
                else
                    rms_pass_counter = 0;
                end
            else
                 rms_pass_counter = 0; 
            end
            
            % 2. Adaptive Learning Trigger (The User's Idea)
            learning_triggered = false;
            
            % DEFINE TIMEOUT: Give Stage 1 extra time (4x) to climb the hill
            if stage_index == 1
                current_timeout = max_stage_batches * 4; % e.g. 60 batches
            else
                current_timeout = max_stage_batches;     % e.g. 15 batches
            end
            
            % CHECK TIMEOUT (Now applies to ALL stages, including Stage 1)
            if batches_in_stage >= current_timeout
                % We are stuck. The current noise floor prediction is wrong.
                % LEARN from the minimum RMS we achieved.
                
                theoretical_sigma = sqrt(mean(bin_noise_sigma.^2));
                observed_floor = min_stage_rms;
                
                % Calculate New Prefactor
                new_factor = observed_floor / theoretical_sigma;
                
                fprintf('\n(!) ADAPTIVE LEARNING (Stage %d Timeout at %d batches).\n', stage_index, batches_in_stage);
                fprintf('    Theoretical Noise: %.4f | Observed Floor: %.4f\n', theoretical_sigma, observed_floor);
                fprintf('    Updating Noise Prefactor: %.2f -> %.2f\n', noise_prefactor, new_factor);
                
                % Update Global Physics Model
                noise_prefactor = max(noise_prefactor, new_factor);
                
                % Recalculate current floor
                dens_noise_floor = noise_prefactor * theoretical_sigma;
                
                % Force transition
                learning_triggered = true;
            end
            
            transition_triggered = (rms_pass_counter >= required_passes) || learning_triggered;

            % --- B. PLOT (Now that history exists) ---
             if graphing
                set(0, 'CurrentFigure', f_fig);
                
                % 1. Density (Top Left)
                curr_ndens_norm = 100 * (((ndens.counts / max(1,w_count))./ndens.vols) ./ ndens.ndens0);
                subplot(ax_dens); 
                plot(ndens.centers, curr_ndens_norm, 'w', 'LineWidth', 2);
                xline(S.br, '-r'); ylim([80 120]); xlim([0 S.br]);
                title('Density [%]','Color','w'); 
                set(gca,'Color','k','XColor','w','YColor','w','LineWidth',2);

                % 2. Convergence Metrics (Top Right)
                subplot(ax_conv);
                yyaxis left; ax=gca; ax.YColor=[1 1 0];
                % '.-' ensures points are visible even if it's the very first batch
                plot(history.steps, history.dens_smooth, '.-', 'Color',[1 1 0], 'LineWidth', 2); 
                ylabel('Density RMS');
                yyaxis right; ax=gca; ax.YColor=[0 1 1];
                plot(history.steps, history.max_corr, '.-', 'Color',[0 1 1], 'LineWidth', 2); 
                ylabel('MaxCorr');
                title(sprintf('Stage %d | Batch %d', stage_index, sgd_batch_size),'Color','w');
                set(gca,'Color','k','XColor','w','LineWidth',2);

                % 3. Control State (Mid Left)
                subplot(ax_ctrl);
                plot(history.steps, history.gain, 'm.-', 'LineWidth', 2); 
                title('Gain','Color','w');
                set(gca,'Color','k','XColor','w','YColor','w','LineWidth',2);
                yscale log
                
                % 4. Diagnostics (Mid Right)
                subplot(ax_diag);
                plot(history.steps, history.fraction_updated, 'w.-', 'LineWidth', 2);
                ylim([0 1]);
                title('Fraction Updated','Color','w');
                set(gca,'Color','k','XColor','w','YColor','w','LineWidth',2);

                % 5. SNR (Bottom Right)
                subplot(ax_snr);
                plot(history.steps, history.median_snr, 'g.-', 'LineWidth', 2);
                title('Median SNR','Color','w');
                set(gca,'Color','k','XColor','w','YColor','w','LineWidth',2);

                drawnow;
                
                msg = sprintf('Step %d | RMS: %.4f (Target: %.4f) | Pass: %d/%d | Gain: %.1e', ...
                    qs, dens_metric, rms_tolerance*dens_noise_floor, rms_pass_counter, required_passes, sgd_gain);
                fprintf([reverseStr, msg]);
                reverseStr = repmat('\b', 1, length(msg));
            end
            
            if transition_triggered && ~is_frozen_production
                fprintf('\n=== STAGE %d COMPLETE (RMS %.4f vs Target %.4f) ===\n', ...
                    stage_index, dens_metric, rms_tolerance*dens_noise_floor);
                
                % ... [Standard Snapshot Save Logic] ...
                if enable_io
                    sname = sprintf('SNAPSHOT_Stage%d_%s.mat', stage_index, seriesname);
                    save(sname, 'sgd_correction', 'p', 'pgp', 'S');
                end

                % --- FIX: Lookahead Logic ---
                % Calculate what the NEXT batch size would be
                next_batch_mult = current_batch_mult * 4;
                next_batch_size = base_batch_size * next_batch_mult;

                % Check if the NEXT stage is too big BEFORE we enter it
                if next_batch_size > max_batch_size_limit
                    % Enter Frozen Production NOW instead of starting a huge stage
                    is_frozen_production = true;
                    sgd_base_gain = 0; sgd_gain = 0; sgd_batch_size = 1e6;
                    fprintf('*** ANNEALING COMPLETE (Next batch > Limit). ENTERING FROZEN PRODUCTION ***\n');
                else
                    % Safe to proceed with SAWTOOTH Transition
                    current_batch_mult = next_batch_mult;
                    sgd_batch_size = next_batch_size;
                    sgd_base_gain = sgd_base_gain * 0.5;
                    
                    % Update Target (Scaling by sqrt(4)=2)
                    % This replaces the 'new_floor_estimate' line you had before
                    theoretical_sigma_next = sqrt(mean(bin_noise_sigma.^2)) * 0.5; 
                    
                    fprintf('>>> ADVANCING to Stage %d.\n', stage_index + 1);
                    fprintf('    New Batch Size: %d\n', sgd_batch_size);
                    fprintf('    New Base Gain:  %.3f\n', sgd_base_gain);
                    
                    stage_index = stage_index + 1;
                    rms_pass_counter = 0;
                    batches_in_stage = 0; 
                    min_stage_rms = inf; % Reset tracker for new stage
                end
            elseif steps_in_batch >= sgd_batch_size && is_frozen_production
                disp('--- PRODUCTION RUN COMPLETE ---');
                break;
            end
            
            % G. Reset Accumulators
            batch_sum_drift(:) = 0; batch_sum_drift_sq(:) = 0; batch_counts(:) = 0;
            steps_in_batch = 0;
            ndens.counts(:) = 0; dens.pre.counts(:) = 0; 
            if transition_triggered, dens_metric = 0; end
        end
    end

    % --- Update & Apply (Every Step) ---
    dr_corr_mag = F_corr_interp(prho);
    core_mask = prho < (S.br - potdepth); dr_corr_mag(core_mask) = 0;
    total_disp = base_disp + (dr_corr_mag .* pvers);
    p2 = p + total_disp;

    % Ghost Update (Free Tangential)
    draw=randi(1e6,[S.N 1]);
    v_rand_gp = DISP(draw, :);
    if S.potential ~= 0, v_rand_gp(idxgp,:) = v_rand_gp(idxgp,:) + potdispsgp; end
    pgp_norm = vecnorm(pgp, 2, 2) + eps;
    pgp_dir  = pgp ./ pgp_norm;
    v_rad_comp = sum(v_rand_gp .* pgp_dir, 2);
    v_tan_gp   = v_rand_gp - (v_rad_comp .* pgp_dir);
    pgp_temp = pgp + v_tan_gp;
    pgp_next_dir  = pgp_temp ./ (vecnorm(pgp_temp, 2, 2) + eps);
    pgp2  = pgp_next_dir .* (2*S.br - vecnorm(p2, 2, 2));

    % Hard Sphere Reset logic
    if S.potential == 0
        distpp = pdist2(p2, [p2; pgp2], 'squaredeuclidean');
        idxd = distpp > 0 & distpp < (2*S.rp)^2;
        [r_idx, c_idx] = find(idxd);
        c_idx(c_idx > S.N) = c_idx(c_idx > S.N) - S.N;
        resetters = unique([r_idx; c_idx]);
        if ~isempty(resetters)
            p2(resetters, :) = p(resetters, :);
            pgp2(resetters, :) = pgp(resetters, :);
        end
    end

    p2rho = vecnorm(p2,2,2); pgp2rho = vecnorm(pgp2,2,2);
    idx_swap = p2rho > pgp2rho;
    p(idx_swap,:) = pgp2(idx_swap,:); pgp(idx_swap,:) = p2(idx_swap,:);
    p(~idx_swap,:) = p2(~idx_swap,:); pgp(~idx_swap,:) = pgp2(~idx_swap,:);

    if qs > 5e7, fprintf('Internal safety limit (5e7) reached.\n'); break; end
end

ASYMCORR.correction = [sgd_centers, sgd_correction];
if enable_io
    save(filenamecorrection, 'ASYMCORR', 'sgd_edges');
    save(filestartingconfiguration, 'p', 'pgp', 'S');
end
disp('SGD V8 Complete.');
end