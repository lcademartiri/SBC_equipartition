function [p,pgp,ASYMCORR] = sgd_pdf_metric_sobochem(S,H,H_interpolant,opts,data_folder)
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
if isfield(opts,'sgd_smooth_win'),       sgd_smooth_win = opts.sgd_smooth_win; else sgd_smooth_win = 3; end
if isfield(opts,'sgd_cap'),              sgd_cap = opts.sgd_cap; else sgd_cap = 0.02 * S.rp; end  % should be set based on stdx . right now it is redundant
if isfield(opts,'clip_frac'),            clip_frac = opts.clip_frac; else clip_frac = 0.3; end 
if isfield(opts,'abs_cap_frac'),         abs_cap_frac = opts.abs_cap_frac; else abs_cap_frac = 0.005; end
if isfield(opts,'flatness_lambda'),      flatness_lambda = opts.flatness_lambda; else flatness_lambda = 200; end % conservative default

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
	base_batch_size = max([steps_for_decorrelation, sdmns.steps_for_stats,10000]);
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
    sgd_momentum = zeros(sgd_bins, 1); % Stores the "velocity" of the correction
    momentum_beta = 0.98;              % Damping factor (0.8 to 0.95 is ideal)
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
		
		% initialize plots and formatting
		f_fig = figure('Units','normalized','Position',[0.05 0.05 0.85 0.85]);
		set(gcf,'Color','k');
		ax_dens = subplot(3,2,1); ax_pdf  = subplot(3,2,2);
		ax_conv = subplot(3,2,3); ax_ctrl = subplot(3,2,4);
		ax_diag = subplot(3,2,5); ax_snr  = subplot(3,2,6);
		axs = [ax_dens, ax_pdf, ax_conv, ax_ctrl, ax_diag, ax_snr];
		for a = axs
			set(a,'Color','k','XColor','w','YColor','w','LineWidth',3);
			set(get(a,'Title'),'FontWeight','bold','Color','w');
			set(get(a,'XLabel'),'FontWeight','bold','Color','w');
			set(get(a,'YLabel'),'FontWeight','bold','Color','w');
			set(a,'FontWeight','bold','FontSize',10);
		end
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
    check_freq = 10000; 
    
    if thermflag == 0 && mod(qs, check_freq) == 0
        thermflag=1;
        
        % % 1. Calculate g(r) for ONLY the current block
        % % We don't want the long history, we want to know: "Are we stable NOW?"
        % curr_block_counts = pdf.pre_counts; % pre_counts should only hold data for this block
        % pdf.curr_g = (curr_block_counts / check_freq) ./ pdf.denom;
        % 
        % % 2. Calculate current RMS
        % pdf.therm_residuals = pdf.curr_g(pdf.therm_mask) - 1;
        % pdf.rms = sqrt(mean(pdf.therm_residuals.^2));
        % 
        % % 3. Calculate Noise Floor for a single block
        % expected_counts = pdf.denom(pdf.therm_mask) * check_freq;
        % sigma_block = sqrt(mean(1 ./ (expected_counts + eps)));
        % 
        % % 4. Stability Criterion: Is the change between blocks less than the noise?
        % % We use a 2-sigma threshold (95% confidence) to allow for Brownian noise
        % tolerance = 2.0 * sigma_block; 
        % 
        % if qs > check_freq % Only compare if we have a previous block
        %     drift = abs(pdf.rms - prev_pdf_rms);
        %     drift_ok = drift < tolerance;
        % else
        %     drift = 0;
        %     drift_ok = false; 
        % end
        % 
        % rms_ok = pdf.rms < 0.3; % Basic sanity check
        % 
        % fprintf('Therm Block (Step %d): RMS=%.4f | Drift=%.1e (Tol %.1e) | Passes: %d/%d\n', ...
        %     qs, pdf.rms, drift, tolerance, therm_pdf_passes, required_therm_passes);
        % 
        % if drift_ok && rms_ok
        %     therm_pdf_passes = therm_pdf_passes + 1;
        % else
        %     therm_pdf_passes = 0; % Reset on any jump larger than noise
        % end
        % 
        % % --- IMPORTANT: Reset accumulators for the NEXT block ---
        % prev_pdf_rms = pdf.rms;
        % pdf.pre_counts(:) = 0; 
        % 
        % % 5. Exit
        % if therm_pdf_passes >= required_therm_passes
        %     disp('--- Thermalization Complete (Structural Convergence) ---');
        %     thermflag = 1; qs = 0;
        %     ndens.counts(:) = 0; % Clear for SGD phase
        % end
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
    % --- MACHINE LEARNING ENGINE ------------------------------------------------
	% ############################################################################
	% 
	% LOGIC: 
	% 1. accumulate binwise radial disps, squared radial disps and bin occupancies for a whole batch of timesteps
	% 2. at the end of batch trigger analysis
	
    if thermflag == 1
		% extract radial displacement of all real particles by dot product of brownian+potential disp and the versor at the beginning of step 
        dr_raw = sum(base_disp .* pvers, 2);
		% bin each real particle to one of the sgd bins
        bin_idx = discretize(prho, sgd_edges);
		
		% --- BINWISE ACCUMULATION OF STRUCTURE AND DISPLACEMENT DATA ------------------------
        valid_mask = bin_idx > 0 & bin_idx <= sgd_bins;
		% accumulation of radial displacements, radial SQUARED displacements, and bin occupancy
        if any(valid_mask)
            idxs = bin_idx(valid_mask); % get the valid bin numbers of all reals
            values = dr_raw(valid_mask); % get the valid radial displacements of all reals
            new_sums = accumarray(idxs, values, [sgd_bins, 1]); % accumulate the radial displacements binwise
            new_sums_sq = accumarray(idxs, values.^2, [sgd_bins, 1]); % accumulate the SQUARED radial displacements binwise
            new_counts = accumarray(idxs, 1, [sgd_bins, 1]); % accumulate the occupancy binwise
            batch_sum_drift = batch_sum_drift + new_sums; % accumulate the binwise radial displacements in batch-wise accumulators
            batch_sum_drift_sq = batch_sum_drift_sq + new_sums_sq; % accumulate the binwise SQUARED radial displacements in batch-wise accumulators
            batch_counts = batch_counts + new_counts; % accumulate the binwise occupancy in batch-wise accumulators
        end
		% accumulation of number density data - cannot use the same edges as pdf as this is much finer resolved
        [hc, ~] = histcounts(prho, ndens.edges);
        ndens.counts = ndens.counts + hc';
		% ------------------------------------------------------------------------------------

        steps_in_batch = steps_in_batch + 1;

        % --- BATCH PROCESSING TRIGGER ---
        if steps_in_batch >= sgd_batch_size
            
            % A. Initialize Stats Processing
				% mask of populated bins
				has_data = batch_counts > 0; % mask of populated bins
				idxs = find(has_data); % indices of populated bins
				% initialization of batch vectors
				bin_mean = zeros(sgd_bins,1);
				snr = zeros(sgd_bins,1);
				n_i = batch_counts;
			% A. end
            
            % B. Calculate Mean Displacement and SNR for Each Bin
				for ii = idxs' % loop over every populated bin
					ni = n_i(ii); % get the total occupancy count of that bin
					mu = batch_sum_drift(ii) / ni; % get the mean radial displacement in that bin
					ss = batch_sum_drift_sq(ii); % get the sum of SQUARED radial displacement in that bin
					
					% Robust SNR Calculation (Prevent Infinite SNR)
					if ni > 1 % if occupancy in the bin is more than one
						v = max(0, (ss - ni * mu^2) / (ni - 1)); % variance of the radial displacement in this bin
						se = sqrt(v) / sqrt(ni);  % standard error of the mean radial displacement in that bin
						s = abs(mu) / (se + eps); % calculate the signal to noise ratio in that bin 
					else % assign zero SNR to empty bins
						s = 0; 
					end
					bin_mean(ii) = mu;
					snr(ii) = s;
				end
			% B. end
            
            % C. Calculate the Composite Sobolev Metric
                % --- 1. PDF COMPONENT (Local Structure) ---
				% This ensures the local packing (hydration shells) remains correct
				pdf.curr_g = (pdf.pre_counts / max(1,steps_in_batch)) ./ pdf.denom;
				pdf.residuals = pdf.curr_g(pdf.mask) - 1;
				pdf.weights = pdf.denom(pdf.mask);
				
                % Standard RMS for PDF (we WANT local peaks here, so no Sobolev)
				metric_pdf = sqrt(sum(pdf.weights .* (pdf.residuals.^2)) / sum(pdf.weights));

                % --- 2. DENSITY SOBOLEV COMPONENT (Global Homogeneity) ---
                % This targets the standing waves
                
                % Calculate normalized density profile (1.0 = Ideal)
                rho_profile = ((ndens.counts / max(1,steps_in_batch)) ./ ndens.vols) ./ ndens.ndens0;
                
                % Mask for the bulk (avoiding the wall layer where gradients are physical)
                % We look at 0 to 90% of radius
                mask_dens = ndens.centers > (S.br - potdepth); 
                rho_bulk = rho_profile(mask_dens);
                
                if ~isempty(rho_bulk)
                    % A. The Amplitude Term (L2 Norm) - "Is the density correct?"
                    err_ampl = rho_bulk - 1.0;
                    score_ampl = sqrt(mean(err_ampl.^2));
                    
                    % B. The Oscillation Term (H1 Semi-norm) - "Is the density flat?"
                    % Calculate gradient between adjacent bins
                    grad_rho = gradient(rho_bulk); 
                    score_grad = sqrt(mean(grad_rho.^2));
                    
                    % Sobolev Metric: Error + Alpha * Gradient
                    % Alpha=5.0 heavily penalizes wiggles.
                    metric_sobolev = score_ampl + flatness_lambda * score_grad;
                else
                    metric_sobolev = 0;
                end

                % --- 3. COMBINE THEM ---
                % We mix them 50/50. 
                % Note: metric_sobolev is usually larger than metric_pdf, so this prioritizes density.
				pdf.raw_metric = 0.5 * metric_pdf + 0.5 * metric_sobolev;

                % --- 4. SMOOTHING (Existing Logic) ---
				if pdf.metric == 0, 
					pdf.metric = pdf.raw_metric;
				else 
					pdf.metric = (1 - metric_smoothing_param) * pdf.metric + metric_smoothing_param * pdf.raw_metric;
				end
				
				if pdf.metric > 0
					min_stage_rms = min(min_stage_rms, pdf.metric);
				end
			% C. end
            
            % D. Calculate the Dynamic Noise Floor
				% counts one would expect from an ideal gas in the steps made 
				expected_counts = pdf.denom(pdf.mask) * steps_in_batch;
				expected_counts(expected_counts==0) = inf;
				% Poisson noise associated with the expected count
				bin_noise_sigma = 1 ./ sqrt(expected_counts);
				% estimation of the noise floor using the noise prefactor as a multiplier
				current_noise_floor = noise_prefactor * sqrt(mean(bin_noise_sigma.^2));
			% D. end
            
            % E. Gain Calculation (Quadratic Governor)
				% measure the current maximum correction
				current_max_corr = max(abs(sgd_correction));
				% measure distance of the maximum correction from the established cap
				cap_proximity = min(1, current_max_corr / sgd_cap);
				% calculate the governor factor that would limit gain
				governor_factor = max(0.01, 1 - cap_proximity^2);
				% establish real gain by multiplying the base gain by the governor factor
				sgd_gain = sgd_base_gain * governor_factor;
			% E. end
            
            % F. Record History & Plot BEFORE Reset
				history.steps(end+1) = qs;
				history.pdf_dev(end+1) = pdf.raw_metric;
				history.pdf_smooth(end+1) = pdf.metric;
                history.target(end+1) = rms_tolerance * current_noise_floor;
				history.max_corr(end+1) = current_max_corr;
				history.gain(end+1) = sgd_gain;
				history.batch_size(end+1) = sgd_batch_size;
				history.fraction_updated(end+1) = numel(idxs) / max(1, sgd_bins);
				if any(has_data)
					history.median_snr(end+1) = median(snr(has_data));
				else
					history.median_snr(end+1) = 0;
				end
			% F. end           
            
            % G. Correction Update (Density-Targeted with Momentum Damping)
                if ~is_frozen_production 
                    
                    % 1. Calculate Normalized Density Profile (1.0 = Ideal Bulk)
                    rho_profile = ((ndens.counts / max(1,steps_in_batch)) ./ ndens.vols) ./ ndens.ndens0;
                    rho_err = rho_profile - 1.0; 
                    
                    % 2. Calculate Density Gradient (Sobolev term to damp waves)
                    grad_rho = gradient(rho_profile);
                    
                    % 3. Map Density Error and Gradient onto Correction Bins (sgd_centers)
                    err_map = interp1(ndens.centers, rho_err, sgd_centers, 'linear', 'extrap');
                    grad_map = interp1(ndens.centers, grad_rho, sgd_centers, 'linear', 'extrap');
                    
                    % 4. Calculate Raw Proposed Change (delta)
                    % Logic: If rho < 1 (depletion), err_map is negative. 
                    % We use -sgd_gain to turn that into a POSITIVE radial pull toward the boundary.
                    % flatness_lambda * grad_map penalizes the "wiggles" to stop divergence.
                    d_raw = -sgd_gain * (err_map + flatness_lambda * grad_map);
                    
                    % 5. Apply the Taper (Zero at inner edge, Max at boundary)
                    d_tapered = d_raw .* taper_mask;
                    
                    % 6. Apply Velocity Memory (Momentum)
                    % This low-pass filters the update to prevent structural blow-up.
                    sgd_momentum = (momentum_beta * sgd_momentum) + ((1 - momentum_beta) * d_tapered);
                    
                    % 7. Apply Momentum to the Correction
                    sgd_correction = sgd_correction + sgd_momentum;
                    
                    % 8. Apply the Total Global Cap (using the increased opts.sgd_cap)
                    % We use a soft-cap (tanh) for smoother transitions near the limit
                    % but a hard clip is also fine if opts.sgd_cap is high enough.
                    sgd_correction = max(min(sgd_correction, opts.sgd_cap), -opts.sgd_cap);
                    
                    % 9. Final Smoothing and Interpolant Update
                    sgd_correction = smoothdata(sgd_correction, 'movmean', sgd_smooth_win);
                    F_corr_interp.Values = sgd_correction;
                end
            % G. end
            
            % H. Adaptive Annealing Logic - at each stage of learning change the grace limit 
				batches_in_stage = batches_in_stage + 1;
				% Select the appropriate grace period
				if stage_index == 1
					current_grace_limit = stage1_grace_batches; % Use the long warmup for construction
				else
					current_grace_limit = stage_grace_batches;  % Use short warmup for refinement
				end
			% H. end
            
            % I. Grace Period 
				% this check ensures that we start considering the comparison between the pdf metric and the noise floor/tolerance only after a certain grace period.
				% If I have conducted more batches than the grace limit... 
				if batches_in_stage > current_grace_limit
					% ... and if current pdf metric is within tolerance of the noise floor accumulate a pass counter...
					if pdf.metric < (rms_tolerance * current_noise_floor)
						rms_pass_counter = rms_pass_counter + 1;
					% ... otherwise reset it
					else
						rms_pass_counter = 0;
					end
				% ... and if I have NOT conducted more batches than the grace limit keep the rms pass counter null
				else
					 rms_pass_counter = 0; 
				end
			% I. end
            
            % L. Adaptive Learning Trigger
				learning_triggered = false;
				
				% Define Timeout: Give Stage 1 extra time (4x) to climb the hill
				if stage_index == 1
					current_timeout = max_stage_batches * 4; % e.g. 60 batches
				else
					current_timeout = max_stage_batches;     % e.g. 15 batches
				end
				
				% Check Timeout
				% this checks if we are stuck because the system is not able to further improve the metric probably because the noise floor is estimated incorrectly
				% at which point the current metric teaches us what is the noise floor. We basically assume to have reached the noise floor.
				if batches_in_stage >= current_timeout			
					theoretical_sigma = sqrt(mean(bin_noise_sigma.^2)); % RMS of the bin noises to establish the global noise floor consistent with how the metric is measured
					% If actual noise > predicted, update our physics model to match reality
					noise_prefactor = max(noise_prefactor, min_stage_rms / theoretical_sigma);
					current_noise_floor = noise_prefactor * theoretical_sigma;
					fprintf('(!) ADAPTIVE: Target unreachable. Raising noise floor to %.4f\n', current_noise_floor);
					learning_triggered = true; % learning flag that triggers transition to new stage
				end
			% L. end
            
			% PHASE TRANSITION based on either beating the noise floor or getting to timeout and triggering learning
            transition_triggered = (rms_pass_counter >= required_passes) || learning_triggered;

            % --- B. PLOT (Now that history exists) ---
             if graphing
                set(0, 'CurrentFigure', f_fig);
                
                % 1. Density (Top Left) - NOW SHOWS SOBOLEV REGION
                curr_ndens_norm = 100 * (((ndens.counts / max(1,steps_in_batch))./ndens.vols) ./ ndens.ndens0);
                subplot(ax_dens); 
                plot(ndens.centers, curr_ndens_norm, 'w', 'LineWidth', 2);
                xline(S.br, '-r'); 
                
                % VISUALIZE THE SOBOLEV CUTOFF (The "Bulk" you are optimizing)
                xline(S.br - potdepth, '--m', 'LineWidth', 1); 
                
                ylim([80 120]); xlim([0 S.br]);
                title('Density [%] (Magenta: Sobolev Limit)','Color','w'); 
                set(gca,'Color','k','XColor','w','YColor','w','LineWidth',2);

                % 2. PDF (Top Right) -- RESTORED
                % Calculate current g(r)
                curr_g = (pdf.pre_counts / max(1,steps_in_batch)) ./ pdf.denom;
                subplot(ax_pdf);
                % Only plot valid range to avoid weird scaling
                plot(pdf.centers, curr_g, 'Color', [1 1 0], 'LineWidth', 2); 
                yline(1, '--w');
                xlim([0 2.1*S.br]); ylim([0.5 1.5]);
                title('Pair Distribution Function g(r)','Color','w');
                set(gca,'Color','k','XColor','w','YColor','w','LineWidth',2);

                % 2. Convergence Metrics (Top Right) - NOW SHOWS TARGET
                subplot(ax_conv);
                yyaxis left; ax=gca; ax.YColor=[1 1 0];
                
                % Plot the Metric (Yellow)
                plot(history.steps, history.pdf_smooth, '.-', 'Color',[1 1 0], 'LineWidth', 2);
                hold on;
                
                % Plot the TARGET (Red Dashed) - This shows the Adaptive Learning!
                plot(history.steps, history.target, '--r', 'LineWidth', 1.5);
                hold off;
                
                ylabel('Metric vs Target');

                yyaxis right; ax=gca; ax.YColor=[0 1 1];
                plot(history.steps, history.max_corr, '.-', 'Color',[0 1 1], 'LineWidth', 1); 
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
                    qs, pdf.metric, rms_tolerance*current_noise_floor, rms_pass_counter, required_passes, sgd_gain);
                fprintf([reverseStr, msg]);
                reverseStr = repmat('\b', 1, length(msg));
            end
            
			% M. Handling Phase Transition 
			% this bit takes over when the stage is finished as signaled by the flag transition_triggered and
			% looks ahead into what the next stage would look like in terms of numbers of steps
			% if the stage is too long, it stops learning and proceed to a production run of 1e6 steps with a frozen potential
			% otherwise it updates the characteristics of the stage and resets the counters.
			% M. Handling Phase Transition 
            if transition_triggered && ~is_frozen_production
                fprintf('\n=== STAGE %d COMPLETE (RMS %.4f) ===\n', stage_index, pdf.metric);

                % 1. ANCHORING: Calculate the true "Noise Factor" of the current stage
                % This captures how much 'dirtier' your system is compared to an Ideal Gas
                theoretical_sigma = sqrt(mean(bin_noise_sigma.^2));
                observed_prefactor = min_stage_rms / theoretical_sigma;
                
                % Update the global prefactor. 
                % If this number keeps rising stage-after-stage, it proves you have systematic waves.
                noise_prefactor = max(0.5, observed_prefactor); 
                fprintf('    System "Dirtiness" (Prefactor): %.2f x Ideal Gas\n', noise_prefactor);

                % 2. TRANSITION
                next_batch_mult = current_batch_mult * 4;
                next_batch_size = base_batch_size * next_batch_mult;

                if next_batch_size > max_batch_size_limit
                    is_frozen_production = true;
                    sgd_base_gain = 0; sgd_gain = 0; sgd_batch_size = 1e6;
                    fprintf('*** ANNEALING COMPLETE. ENTERING FROZEN PRODUCTION ***\n');
                else
                    current_batch_mult = next_batch_mult;
                    sgd_batch_size = next_batch_size;
                    sgd_base_gain = sgd_base_gain * 0.5;
                    
                    % 3. CALCULATE NEXT TARGET
                    % Theoretical noise drops by half (sqrt(4)=2)
                    next_theoretical_sigma = theoretical_sigma * 0.5;
                    
                    % The new target scales the prefactor by the NEW theoretical noise
                    next_target = rms_tolerance * noise_prefactor * next_theoretical_sigma;
                    
                    fprintf('>>> ADVANCING to Stage %d.\n', stage_index + 1);
                    fprintf('    Batch Size: %d -> %d\n', steps_in_batch, sgd_batch_size);
                    fprintf('    Target:     %.4f -> %.4f ', rms_tolerance*current_noise_floor, next_target);
                    
                    % DIAGNOSTIC: Tell the user if we are assuming Noise or Waves
                    if abs(next_target - (pdf.metric/2)) < 1e-4
                        fprintf('(Scaling as Pure Noise)\n');
                    else
                        fprintf('(Dominated by Systematic Waves)\n');
                    end

                    stage_index = stage_index + 1;
                    rms_pass_counter = 0;
                    batches_in_stage = 0; 
                    min_stage_rms = inf; 
                end
			end
            
            % G. Reset Accumulators
            batch_sum_drift(:) = 0; batch_sum_drift_sq(:) = 0; batch_counts(:) = 0;
            steps_in_batch = 0;
            ndens.counts(:) = 0; 
			pdf.pre_counts(:) = 0; 
            if transition_triggered, pdf.metric = 0; end
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