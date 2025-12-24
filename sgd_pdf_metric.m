function [p,pgp,ASYMCORR] = sgd_pdf_metric(S,PDF,H,H_interpolant,opts,data_folder)
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
noise_prefactor = 2;
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
filenamecorrection = sprintf(['ASYMCORR2eps_',seriesname,'_%s_%.0e_%.0e_%.0f_%.1f_%.1e.mat'],...
    potname,S.rp,S.phi,S.N,S.pot_epsilon/S.kbT,S.pot_sigma);
filestartingconfiguration = sprintf(['START2eps_SBC_%s_%.0e_%.0e_%.0f_%.1f_%.1e.mat'],...
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
	history = struct('steps',[],'pdf_dev',[],'pdf_smooth',[],'max_corr',[],'gain',[],...
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
        drift_ok = abs(pdf.rms - prev_pdf_rms) < 0.001 * sigma_pdf; 
        % 2. Sanity Check:
        % Just ensure we aren't stuck in a "Zombie" state (RMS > 0.5)
        rms_ok = pdf.rms < 0.2; 
        
        % Print outcome for checking
        if mod(qs, 10) == 0
             fprintf('Therm Check (Step %d): RMS=%.4f | Drift=%.1e (Tol %.1e) | Stable? %d\n', ...
                qs, pdf.rms, abs(pdf.rms - prev_pdf_rms), 0.1*sigma_pdf, drift_ok);
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
            
            % C. Calculate the PDF Metric
				% calculate the complete g(r)
				pdf.curr_g = (pdf.pre_counts / max(1,steps_in_batch)) ./ pdf.denom;
				% calculate the residuals from 1 of the masked g(r)
				pdf.residuals = pdf.curr_g(pdf.mask) - 1;
				% calculate the weights, i.e., the masked pdf denominator
				pdf.weights = pdf.denom(pdf.mask);
				% calculate the weighted RMS score (the smaller is the weight,i.e. the count, the smaller is its influence in the score)
				pdf.raw_metric = sqrt(sum(pdf.weights .* (pdf.residuals.^2)) / sum(pdf.weights));
				% smoothen the weighted RMS score by considering the previously accumulated score pdf.metric
				if pdf.metric == 0, 
					pdf.metric = pdf.raw_metric;
				else 
					pdf.metric = (1 - metric_smoothing_param) * pdf.metric + metric_smoothing_param * pdf.raw_metric;
				end
				% store the pdf.metric if this is the smallest ever recorded
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
            
            % G. Correction Update (Only if not frozen)
				bins_to_update = false(sgd_bins,1); % initialize false vector on bins to update
				if ~is_frozen_production % proceed unless frozen
					delta = zeros(sgd_bins,1); % initialize vector
					
					% loop over all bins to effectuate the correction based on two safety nets: bayesian shrinkage and snr-weighing
					for ii = 1:sgd_bins
						% apply bayesian shrinkage based on the n_min threshold to the bins to reduce influence of sparse bins
						% the output is mu_eff which is the mean radial displacement of the bin and s_eff which is the snr.
						if n_i(ii) < n_min
							lambda = 100;
							mu_eff = (n_i(ii)/(n_i(ii)+lambda)) * bin_mean(ii);
							s_eff = snr(ii) * (n_i(ii)/(n_i(ii)+lambda));
						else
							mu_eff = bin_mean(ii);
							s_eff = snr(ii);
						end
						
						% moderate the gain based on the snr. Either a soft correction, or a sharp yes/no threshold depending on snr_target opt
						if use_soft_snr
							per_bin_factor = min(1, s_eff / snr_target);
						else
							per_bin_factor = double(s_eff >= snr_target); 
						end
						
						% calculate the actual correction by inverting the drift*gain and applying it to the bin (delta). And taper it.
						if per_bin_factor > 0
							eta_i = sgd_gain * per_bin_factor; 
							% Apply Taper: Forces delta to 0 at inner boundary
							delta(ii) = -eta_i * mu_eff * taper_mask(ii);
							bins_to_update(ii) = true;
						end
					end
					
					% calculate the list of the maximum current corrections in each bin, capping the minima to the min_clip_floor
					max_current_corr = max(max(abs(sgd_correction)), min_clip_floor);
					% calculate the maximum allowed change to the correction in each bin as set by the clip_frac opts
					max_delta_rel = clip_frac * max_current_corr;
					
					% loop over the updated bins to cap the corrections based on maximum relative change to the prior value and the absolute value
					bins_updated = find(bins_to_update);
					for ii = bins_updated'
						d = delta(ii);
						d = sign(d) * min(abs(d), max_delta_rel);
						d = sign(d) * min(abs(d), abs_cap);
						sgd_correction(ii) = sgd_correction(ii) + d;
					end
					
					% smoothen the potential and update the interpolator
					sgd_correction = max(min(sgd_correction, sgd_cap), -sgd_cap);
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
                
                % 1. Density (Top Left)
                curr_ndens_norm = 100 * (((ndens.counts / max(1,steps_in_batch))./ndens.vols) ./ ndens.ndens0);
                subplot(ax_dens); 
                plot(ndens.centers, curr_ndens_norm, 'w', 'LineWidth', 2);
                xline(S.br, '-r'); ylim([80 120]); xlim([0 S.br]);
                title('Density [%]','Color','w'); 
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

                % 2. Convergence Metrics (Top Right)
                subplot(ax_conv);
                yyaxis left; ax=gca; ax.YColor=[1 1 0];
                % '.-' ensures points are visible even if it's the very first batch
                plot(history.steps, history.pdf_smooth, '.-', 'Color',[1 1 0], 'LineWidth', 2); 
                ylabel('RMS');
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
                    qs, pdf.metric, rms_tolerance*current_noise_floor, rms_pass_counter, required_passes, sgd_gain);
                fprintf([reverseStr, msg]);
                reverseStr = repmat('\b', 1, length(msg));
            end
            
			% M. Handling Phase Transition 
			% this bit takes over when the stage is finished as signaled by the flag transition_triggered and
			% looks ahead into what the next stage would look like in terms of numbers of steps
			% if the stage is too long, it stops learning and proceed to a production run of 1e6 steps with a frozen potential
			% otherwise it updates the characteristics of the stage and resets the counters.
			if transition_triggered && ~is_frozen_production
                fprintf('\n=== STAGE %d COMPLETE (RMS %.4f vs Target %.4f) ===\n', ...
                    stage_index, pdf.metric, rms_tolerance*current_noise_floor);

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
                    
                    % Update theoretical floor
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
                % 1. CALCULATE FINAL VALIDATION STATISTICS
                % We use the massive amount of data collected in this frozen run
                validation_g = (pdf.pre_counts / max(1,steps_in_batch)) ./ pdf.denom;
                
                % 2. STORE IN THE OUTPUT STRUCT
                % This saves the "Proof" that the potential works
                ASYMCORR.validation.r = pdf.centers;
                ASYMCORR.validation.gr = validation_g;
                ASYMCORR.validation.counts = pdf.pre_counts;
                ASYMCORR.validation.steps = steps_in_batch;
                ASYMCORR.validation.rms_error = pdf.metric; % The final RMS score

                % 2. VALIDATION DENSITY (Normalized to %)
                % Calculate percentage deviation from ideal density
                final_dens_norm = 100 * (((ndens.counts / max(1,steps_in_batch))./ndens.vols) ./ ndens.ndens0);
                
                ASYMCORR.validation.dens_r = ndens.centers;
                ASYMCORR.validation.dens_profile = final_dens_norm;
                ASYMCORR.validation.dens_counts = ndens.counts;
                
                % 3. METADATA
                ASYMCORR.validation.steps = steps_in_batch;
                break;
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