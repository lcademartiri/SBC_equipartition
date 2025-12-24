function [p,pgp,ASYMCORR] = scf_v1(S,H,H_interpolant,opts,data_folder)
% SGD_PDF_METRIC (The "Sawtooth" Annealer)
%
% Logic:
% 1. Search Phase: Starts at base_batch_size (physically derived).
% 2. Convergence: Requires RMS < 1.2 * NoiseFloor for 3 consecutive batches.
% 3. Annealing: On success, Batch x 4, Gain x 0.5.
% 4. Production: Once Batch >= 1e5, freezes Gain to 0 and runs 1e6 steps.
%
% Key Features: Quadratic Governor, Robust SNR, Free Tangential Ghosts.

if nargin < 3, opts = struct(); end

% #################################################################################
% -------------------- 1. PARSING OPTIONS ================-------------------------
% #################################################################################

% Dynamics & Control
if isfield(opts,'base_gain'),            sgd_base_gain_default = opts.base_gain; else sgd_base_gain_default = 0.5; end
if isfield(opts,'sgd_smooth_win'),       sgd_smooth_win = opts.sgd_smooth_win; else sgd_smooth_win = 5; end
if isfield(opts,'sgd_cap'),              sgd_cap = opts.sgd_cap; else sgd_cap = 0.003 * S.rp; end  % should be set based on stdx . right now it is redundant
if isfield(opts,'clip_frac'),            clip_frac = opts.clip_frac; else clip_frac = 0.3; end 
if isfield(opts,'abs_cap_frac'),         abs_cap_frac = opts.abs_cap_frac; else abs_cap_frac = 0.005; end
if isfield(opts,'flatness_lambda'),      flatness_lambda = opts.flatness_lambda; else flatness_lambda = 2; end % conservative default

% Annealing / Convergence criteria
if isfield(opts,'rms_tolerance'),        rms_tolerance = opts.rms_tolerance; else rms_tolerance = 1.2; end 
if isfield(opts,'consecutive_passes'),   required_passes = opts.consecutive_passes; else required_passes = 3; end 
if isfield(opts,'stage1_grace_batches'), stage1_grace_batches = opts.stage1_grace_batches; else stage1_grace_batches = 25; end
if isfield(opts,'stage_grace_batches'),  stage_grace_batches = opts.stage_grace_batches; else stage_grace_batches = 3; end
if isfield(opts,'max_batch_size'),       max_batch_size_limit = opts.max_batch_size; else max_batch_size_limit = 100000; end
if isfield(opts,'max_stage_batches'),    max_stage_batches = opts.max_stage_batches; else max_stage_batches = 15; end

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
gCS = (1 - S.phi/2) / (1 - S.phi)^3; % Carnahan-Starling Contact Value g(sigma) - probability of finding two particles touching each other compared to an ideal gas
diffE = S.esdiff * S.alpha / gCS; % Enskog Effective Diffusivity - how fast a particle diffuses inside a dense crowd - corrected by alpha
tau_alpha = (S.rp^2) / (6 * diffE); % Structural relaxation time - characteristic time it takes for a particle to diffuse a distance equal to its own radius
relaxsteps = ceil(tau_alpha / S.timestep); % Structural relaxation STEPS
steps_for_decorrelation = ceil(5 * relaxsteps); % times for configurational memory to decay to <1% (exp(-5) = 0.006)

% Noise prefactor for phi=0.4 (Crowded system correction)
noise_prefactor = 0;
abs_cap = S.stdx;
min_clip_floor = 5e-2 * sgd_cap; % minimum change calculated so that 0 correction bins can start changing
% Depth of the correction 
taper_width = 1.0 * S.rp; % width of the taper in radius units
potdepth = S.rc+taper_width;
if 2*S.br - 2*potdepth < 0, potdepth = S.br; end % if correction depth is larger than the domain radius, just use the domain radius 

min_stage_rms = inf; % Track best performance in current stage

% #################################################################################
% -------------------- 2. FILENAMES -------------------------
% #################################################################################
% name the potentials
if S.potential==1, potname='lj'; elseif S.potential==2, potname='wca'; else potname='hs'; end

% name the series with the specific name passed on from opts
if isfield(opts, 'series_name')
    seriesname = opts.series_name; % Uses the Unique ID from barebones (e.g., Rep1, Rep2)
else
    seriesname = 'sgd_anneal';
end

% define specific names of the correction and the starting config
filenamecorrection = sprintf(['ASYMCORR_',seriesname,'_%s_%.0e_%.0e_%.0f_%.1f_%.1e.mat'],...
    potname,S.rp,S.phi,S.N,S.pot_epsilon/S.kbT,S.pot_sigma);
filestartingconfiguration = sprintf(['START_SBC_',seriesname,'_%s_%.0e_%.0e_%.0f_%.1f_%.1e.mat'],...
    potname,S.rp,S.phi,S.N,S.pot_epsilon/S.kbT,S.pot_sigma);
	
% load pdf denominator if available in database
if S.bc==1
    filepdfdenom = sprintf('PDFdenom_SBC_%.0e_%.0e_%.0f.mat',S.rp,S.phi,S.N);
elseif S.bc==2
    filepdfdenom = sprintf('PDFdenom_PBCc_%.0e_%.0e_%.0f.mat',S.rp,S.phi,S.N);
elseif S.bc==3
    filepdfdenom = sprintf('PDFdenom_PBCFCC_%.0e_%.0e_%.0f.mat',S.rp,S.phi,S.N);
end

% if all files are already available, just load them and exit
if enable_io && exist(filenamecorrection,'file') && exist(filestartingconfiguration,'file')
    load(filenamecorrection); load(filestartingconfiguration);
    history = [];
    return
end

% #################################################################################
% -------------------- 5. STARTING PARTICLE CONFIGURATION -------------------------
% #################################################################################

disp('Creating initial FCC-like lattice...');

if debugging, rng(100); end
p=startingPositions_lj(S);
% --- INSTANT THERMALIZATION (MC SHAKER) ---
fprintf('Running Monte Carlo Shaker to erase memory...\n');
for k = 1:1e5 
    idx = randi(S.N);
    trial_pos = p(idx,:) + (rand(1,3)-0.5) * S.br; 
    % Simple Hard Sphere & Boundary Check
    if norm(trial_pos) < (S.br)
        % Only check neighbors if inside box
        dists = pdist2(trial_pos, p, 'squaredeuclidean');
        dists(idx) = inf; % Ignore self
        if min(dists) > (2*S.rp)^2
            p(idx,:) = trial_pos; % Accept Move
        end
    end
end
pgp = p - (2*S.br).*(p ./ (vecnorm(p,2,2) + eps)); % Update ghosts
% ------------------------------------------


% #################################################################################
% -------------------- 6. SGD STATE INITALIZATION ---------------------------------
% #################################################################################

if S.pot_corr

	% --- RADIAL RANGE OF CORRECTION -------------------------------------------------------------------------------------------------
	sgd_edges = sort((S.br:-0.05*S.rp:S.br - potdepth)'); % in two hundreths of a radius
	sgd_bins = numel(sgd_edges) - 1;
	sgd_centers = sgd_edges(1:end-1) + diff(sgd_edges)/2;
	sgd_vols=(4/3)*pi*(sgd_edges(2:end)).^3 - (4/3)*pi*(sgd_edges(1:end-1)).^3;
	% --------------------------------------------------------------------------------------------------------------------------------
	
	% --- PDF INITIALIZATION pdf ---------------------------------------------------------------------------------------
	maxdist=2*S.br;
    if 10*S.rp>(maxdist-2*S.rc)
        pdf.edges=sort((maxdist:-0.05*S.rp:0)'); % from 0 to box diameter
    else
        OS=sort((maxdist:-0.05*S.rp:(maxdist-2*S.rc))');
        IS=(0:0.05*S.rp:10*S.rp)';
        MS=linspace(max(IS),min(OS),ceil((min(OS)-max(IS))/S.rp))';
        pdf.edges=unique(sort([OS;MS;IS]));
    end
	clear OS IS MS
	pdf.bins = numel(pdf.edges) - 1;
	pdf.centers = pdf.edges(1:end-1) + diff(pdf.edges)/2;
	pdf.vols=(4/3)*pi*(pdf.edges(2:end)).^3 - (4/3)*pi*(pdf.edges(1:end-1)).^3;
	pdf.ndens0 = (S.N / S.bv);
    pdf.r_norm = pdf.centers / S.br; % Geometric Form Factor - Normalize distance by Box Radius (S.br)
    pdf.geom_factor = 1 - (3/4)*pdf.r_norm + (1/16)*pdf.r_norm.^3; % The Finite Volume Correction Polynomial
    pdf.geom_factor = max(0, pdf.geom_factor); % Clamp negative values to 0
    pdf.ndens0 = (S.N / S.bv);
    % Corrected Denominator (Halved for Unique Pairs)
    pdf.denom = 0.5 * (pdf.ndens0 * pdf.geom_factor * S.N) .* pdf.vols;
	% ----------------------------------------------------------------------------------------------------------------
	
	% --- STATISTICAL DETERMINATION OF MINIMUM NUMBER OF STEPS IN BATCH sdmns--------
	% In dilute system the relaxation time could be very small leading to very small batches which could
	% mean that many sgd bins remain empty causing problems. So we here calculate how many steps we do
	% need to have a minimum of counts in each bin	
	sdmns.min_counts_per_bin = 20; 	
	sdmns.prob_hit = sgd_vols(1) / S.bv; % Probability of landing in smallest bin (assuming uniform density)	
	sdmns.required_total_samples = sdmns.min_counts_per_bin / sdmns.prob_hit; % Required total samples = min_counts / prob_hit	
	sdmns.steps_for_stats = ceil(sdmns.required_total_samples / S.N); % Steps needed = Samples / Number of Particles
	base_batch_size = max(steps_for_decorrelation, sdmns.steps_for_stats);
	fprintf('Batch Sizing:\n  Physics Decorrelation: %d steps\n Statistical Floor: %d steps\n -> Selected Batch:%d steps\n', ...
    steps_for_decorrelation, sdmns.steps_for_stats, base_batch_size);
	% -------------------------------------------------------------------------------------------------------------------

	% --- TAPER ON CORRECTION  -------------------------------------------------------------------------------------------------------
	% force correction to taper to zero at the inner edgee of the correction radial range to prevent impulsive forces.
	r_inner_edge = min(sgd_edges); % inner edge of the correction radial range
	taper_mask = zeros(sgd_bins, 1); % initialize the taper mask
	% loop creating a linear taper starting from r_inner_edge at 0 and capping at 1 at r_inner_edge+taper_width
	for itaper = 1:sgd_bins
		taper_mask(itaper) = min(1, max(0, (sgd_centers(itaper) - r_inner_edge) / taper_width));
	end
	% --------------------------------------------------------------------------------------------------------------------------------

	% --- INITIALIZE KEY VECTORS -----------------------------------------------------------------------------------------
	sgd_correction = zeros(sgd_bins, 1);
	batch_sum_drift = zeros(sgd_bins,1);
	batch_sum_drift_sq = zeros(sgd_bins,1);
	batch_counts = zeros(sgd_bins,1);
	pdf.pre_counts = zeros(numel(pdf.edges)-1,1);
	% --------------------------------------------------------------------------------------------------------------------

	% --- INITIALIZE INTERPOLANT OF CORRECTION ---------------------------------------------------------------------------------------
	F_corr_interp = griddedInterpolant(sgd_centers, sgd_correction, 'linear', 'nearest');
	% --------------------------------------------------------------------------------------------------------------------------------

	% --- STARTING VALUES OF ANNEALING VARIABLES  ------------------------------------------------------------------------
	steps_in_batch = 0; % step counter in each batch
	current_batch_mult = 1;
	sgd_batch_size = base_batch_size * current_batch_mult; % size of the batch
	sgd_base_gain = sgd_base_gain_default;
	sgd_gain = sgd_base_gain;
	rms_pass_counter = 0;
	batches_in_stage = 0; 
	stage_index = 1;
	% --------------------------------------------------------------------------------------------------------------------
	
	% --- FLAGS ----------------------------------------------------------------------------------------------------------------------
	is_frozen_production = false;
	% --------------------------------------------------------------------------------------------------------------------------------
	
	% --- UTILITIES -------------------------------------------------------------------------------------------------------
	reverseStr = '';
	% ---------------------------------------------------------------------------------------------------------------------
	
	% --- THERMALIZATION CONDITIONS --------------------------------------------------------------------------------------------------
	therm_pdf_passes = 0;
	required_therm_passes = 100;
	prev_pdf_rms = inf;
	thermflag = 0;
	pdf.therm_mask = pdf.centers > 2*(S.br - potdepth) & pdf.centers < 2*S.br-2*S.rp;
	pdf.mask = pdf.centers > 2*(S.br - potdepth) & pdf.centers < 2*S.br;
    therm_block_size = max(ceil(relaxsteps), 1000);
	% --------------------------------------------------------------------------------------------------------------------------------

	% --- INITIALIZE DIAGNOSTIC VALUES ------------------------------------------------------------------------------------
	pdf.metric = 0;
	% ---------------------------------------------------------------------------------------------------------------------

	% --- INITIALIZE HISTORY ---------------------------------------------------------------------------------------------------------
	history = struct('steps',[],'pdf_dev',[],'pdf_smooth',[],'target',[],'max_corr',[],'gain',[],...
		'batch_size',[],'fraction_updated',[],'median_snr',[]);
	% --------------------------------------------------------------------------------------------------------------------------------
	
	% --- INITIALIZE PLOTTING AND MONITORING ------------------------------------------------------------------------------
	if graphing 
		% initialize number density mapping
		ndens.edges = sort((S.br:-0.05*S.rp:0)');
		ndens.centers = ndens.edges(1:end-1) + diff(ndens.edges)/2;
		ndens.counts = zeros(numel(ndens.centers),1);
		ndens.vols = (4/3)*pi*(ndens.edges(2:end).^3 - ndens.edges(1:end-1).^3);
		ndens.ndens0 = (S.N / S.bv);
		
		% Plot Setup
        f_diag = figure('Color', 'k', 'Name', 'Kinetic SCF Diagnostics');
        ax1 = subplot(2,2,1); ax2 = subplot(2,2,2); 
        ax3 = subplot(2,2,3); ax4 = subplot(2,2,4);
	end
	% -----------------------------------------------------------------------------------------------------------------------
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% -------------------- 7. MAIN LOOP -----------------------------------------------
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if S.pot_corr
    fprintf('Starting SGD_PDF V2 (Annealing). Base Batch: %d. Gain: %.2f. Cap: %.2e\n', sgd_batch_size, sgd_base_gain, sgd_cap);
else
    fprintf('Thermalizing structure')
end

% ---- CALCULATION OF DISPLACEMENT LIBRARIES ----
DISP=build_noise_library(S.stdx,1e6);
qd=1;
% -----------------------------------------------

qs = 0; % step counter

while true
	qs = qs + 1; % update counter
	
	% --- DEFINE STARTING CONFIGURATION ----------------------------
	% norms of real particles
    prho = vecnorm(p, 2, 2);
	% versors of real particles
    pvers = p ./ (prho + eps);
	% mask of ghost generators
    idxgp = prho > (S.br - S.rc);
	% accumulate PDF statistics
		pairdists = pdist(p); % get distances
		[hc_pdf, ~] = histcounts(pairdists, pdf.edges); % bin them
		if numel(hc_pdf) == numel(pdf.pre_counts) % store them
			pdf.pre_counts = pdf.pre_counts + hc_pdf';
		end
	% --------------------------------------------------------------
    
    % --- PDF-BASED THERMALIZATION CHECK -------------------------------------------------------
    if thermflag == 0 && qs > therm_block_size
        
        % cumulative normalization (using total steps 'qs')
        pdf.curr_g = (pdf.pre_counts / qs) ./ pdf.denom;
        % calculate RMS on the masked PDF
        pdf.therm_residuals = pdf.curr_g(pdf.therm_mask) - 1;
        pdf.rms = sqrt(mean(pdf.therm_residuals.^2));
        % calculate noise floor
        expected_counts = pdf.denom(pdf.therm_mask) * qs;
        expected_counts(expected_counts == 0) = inf;
        sigma_pdf = sqrt(mean(1 ./ expected_counts));
        
		% --- THE CRITERIA ---
        % 1. Drift Check:
        % Since we are averaging cumulatively, 'pdf.rms' will change VERY slowly.
        % We demand it effectively stops changing.
        drift_ok = abs(pdf.rms - prev_pdf_rms) < 3*(sigma_pdf/base_batch_size); 
        % 2. Sanity Check:
        % Just ensure we aren't stuck in a "Zombie" state (RMS > 0.5)
        rms_ok = pdf.rms < 0.2; 
        
        % Print outcome for checking
        if mod(qs, 10) == 0
             fprintf('Therm Check (Step %d): RMS=%.4f | Drift=%.1e (Tol %.1e) | Stable? %d\n', ...
                qs, pdf.rms, abs(pdf.rms - prev_pdf_rms), 3*(sigma_pdf/base_batch_size), drift_ok);
        end
		% Checking criteria and accumulate or reset passes
        if drift_ok && rms_ok
            therm_pdf_passes = therm_pdf_passes + 1;
        else
            therm_pdf_passes = 0;
        end
        % Store last step's pdf rms for comparison 
        prev_pdf_rms = pdf.rms;
		% Completing thermalization     
        if therm_pdf_passes >= required_therm_passes
			% wave the flag
            disp('--- Thermalization Complete (Cumulative Stability) ---');
			% reset counters
            thermflag = 1; qs = 0;
			% reset accumulators
            pdf.pre_counts(:) = 0; ndens.counts(:) = 0;
            % save thermalized configuration and return to caller if no correction is needed
            if ~S.pot_corr
                if enable_io, 
					save([data_folder,'\',filestartingconfiguration], 'p', 'pgp', 'S'); 
				end
                return
            end
        end
    end
	% ----------------------------------------------------------------------------------------------

	% --- POTENTIALS AND DISPLACEMENTS ---------------------------------------------
	% collect brownian displacements
	v_rand = DISP(qd:qd+S.N-1, :);
	v_rand_gp = DISP(qd+S.N:qd+2*S.N-1, :);
	qd=qd+2*S.N;
	if qd+2*S.N>1e6
		DISP=DISP(randperm(1e6),:);
		qd=1;
	end
	
	% collect potential displacements
    if S.potential ~= 0 
		% collect potential displacements
        ptemp = [p; pgp(idxgp,:)];
        all_potdisps = potential_displacements_v13(ptemp, S, H, H_interpolant, 1);
        potdisps = all_potdisps(1:S.N, :);
        potdispsgp = all_potdisps(S.N+1:end, :);
		% add brownian displacements        
        base_disp = v_rand + potdisps;
		base_disp_gp = v_rand_gp;
		base_disp_gp(idxgp,:)=base_disp_gp(idxgp,:)+potdispsgp;
    else
        base_disp = v_rand;
		base_disp_gp = v_rand_gp;
    end
	% ------------------------------------------------------------------------------

	% ############################################################################
    % --- KINETIC SCF ENGINE (REPLACES ML ENGINE) --------------------------------
    % ############################################################################
    if thermflag == 1
        % 1. MEASURE & ACCUMULATE (Every Step)
        dr_step = sum(base_disp .* pvers, 2); 
        bin_idx = discretize(prho, sgd_edges);
        valid = bin_idx > 0 & bin_idx <= sgd_bins;
        
        if any(valid)
            idx = bin_idx(valid); 
            val = dr_step(valid);
            batch_sum_drift = batch_sum_drift + accumarray(idx, val, [sgd_bins, 1]);
            batch_sum_drift_sq = batch_sum_drift_sq + accumarray(idx, val.^2, [sgd_bins, 1]);
            batch_counts = batch_counts + accumarray(idx, 1, [sgd_bins, 1]);
        end

        % Accumulate Density & PDF diagnostics
        [hc_dens, ~] = histcounts(prho, ndens.edges);
        ndens.counts = ndens.counts + hc_dens';
        
        pairdists = pdist(p); 
        [hc_pdf, ~] = histcounts(pairdists, pdf.edges);
        pdf.pre_counts = pdf.pre_counts + hc_pdf';
        
        steps_in_batch = steps_in_batch + 1;

        % 2. CORRECT & PLOT (End of Batch)
        if steps_in_batch >= sgd_batch_size
            % A. Calculate Mean Drift & Certainty
            mu = batch_sum_drift ./ (batch_counts + eps);
            var_bin = (batch_sum_drift_sq - batch_counts .* mu.^2) ./ max(1, batch_counts - 1);
            sem = sqrt(max(0, var_bin)) ./ sqrt(batch_counts + eps);
            
            % THE 5-SIGMA RULE: Subtract mu only if statistically certain
            update = mu .* (abs(mu) > (5.0 * sem)); 
            sgd_correction = sgd_correction - (update .* taper_mask);
            
            % Smooth slightly to prevent force discontinuities
            sgd_correction = smoothdata(sgd_correction, 'movmean', 5);
            F_corr_interp.Values = sgd_correction;

            % B. DIAGNOSTIC PLOTS
            if graphing
                % Plot 1: Number Density (Is the dip gone?)
                rho_prof = 100 * ((ndens.counts / steps_in_batch ./ ndens.vols) ./ ndens.ndens0);
                plot(ax1, ndens.centers/S.rp, rho_prof, 'w', 'LineWidth', 2);
                hold(ax1, 'on'); yline(ax1, 100, '--g'); hold(ax1, 'off');
                title(ax1, 'Relative Number Density [%]', 'Color', 'w'); ylim(ax1, [80 120]);

                % Plot 2: PDF g(r) (Structural Integrity)
                curr_g = (pdf.pre_counts / steps_in_batch) ./ pdf.denom;
                plot(ax2, pdf.centers/S.rp, curr_g, 'y', 'LineWidth', 2);
                title(ax2, 'Pair Distribution g(r)', 'Color', 'w'); ylim(ax2, [0.5 1.5]);

                % Plot 3: Correction Potential (The "Bias")
                plot(ax3, sgd_centers/S.rp, sgd_correction, 'c', 'LineWidth', 2);
                title(ax3, 'Applied Radial Bias', 'Color', 'w');

                % Plot 4: Signal-to-Noise (Drift Certainty)
                bar(ax4, sgd_centers/S.rp, abs(mu)./sem); 
                hold(ax4, 'on'); yline(ax4, 5, '-r', '5-Sigma'); hold(ax4, 'off');
                title(ax4, 'Drift Confidence (\mu/\sigma_m)', 'Color', 'w');
                
                drawnow;
            end

            % Console Output
            fprintf('Batch Finished | Bins Adjusted: %d | Max Bias: %.2e\n', ...
                sum(abs(update)>0), max(abs(sgd_correction)));

            % C. RESET ACCUMULATORS
            batch_sum_drift(:)=0; batch_sum_drift_sq(:)=0; batch_counts(:)=0;
            ndens.counts(:)=0; pdf.pre_counts(:)=0; steps_in_batch=0;
        end
    end

    % --- UPDATE REAL PARTICLE POSITIONS --------------------
	% get correction from interpolant and the norms 
    dr_corr_mag = F_corr_interp(prho);
	% zero out correction outside of correction depth
    core_mask = prho < (S.br - potdepth); 
	dr_corr_mag(core_mask) = 0;
	% add corrections to the real particle displacements
    total_disp = base_disp + (dr_corr_mag .* pvers);
	% get final real particle positions + norms
    p2 = p + total_disp;
	p2rho=vecnorm(p2, 2, 2);
	% ------------------------------------------------------

    % --- GHOST PROTOCOL -----------------------------------------------
	% norms of all ghosts
    pgp_norm = vecnorm(pgp, 2, 2) + eps; 
	% versors of all ghosts
    pgp_dir  = pgp ./ pgp_norm;
	% extract radial component of displacement
    v_rad_comp = sum(base_disp_gp .* pgp_dir, 2);
	% extract tangential component of displacement
    v_tan_gp   = base_disp_gp - (v_rad_comp .* pgp_dir);
	% move ghosts tangentially
    pgp_temp = pgp + v_tan_gp;
	% recalculate versors
    pgp_next_dir  = pgp_temp ./ (vecnorm(pgp_temp, 2, 2) + eps);
	% calculate final position as tethered to updated real positions
    pgp2  = pgp_next_dir .* (2*S.br - p2rho);
	% -----------------------------------------------------------------

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
	
	% --- PROMOTION AND DEMOTION --------------------------------------
    p2rho = vecnorm(p2,2,2); 
	pgp2rho = vecnorm(pgp2,2,2);
    idx_swap = p2rho > pgp2rho;
    p(idx_swap,:) = pgp2(idx_swap,:); 
	pgp(idx_swap,:) = p2(idx_swap,:);
    p(~idx_swap,:) = p2(~idx_swap,:); 
	pgp(~idx_swap,:) = pgp2(~idx_swap,:);
	% -----------------------------------------------------------------

    if qs > 5e7, fprintf('Internal safety limit (5e7) reached.\n'); break; end
end

ASYMCORR.correction = [sgd_centers, sgd_correction];
ASYMCORR.history = history; ASYMCORR.sgd_edges = sgd_edges; ASYMCORR.S=S; ASYMCORR.opts=opts;
if enable_io
    save([data_folder,'\',filenamecorrection], 'ASYMCORR', 'sgd_edges');
    save([data_folder,'\',filestartingconfiguration], 'p', 'pgp', 'S');
end
disp('SGD V10 Complete.');
end