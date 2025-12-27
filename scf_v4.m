function [p,pgp,ASYMCORR] = scf_v4(S,H,H_interpolant,opts,data_folder)

% modal SCF using covariance radial dispplacement/|displacement| - 2R tether

enable_io=true;
debugging=false;
graphing=true;
corr_smooth_window=15;
pdfdens_int=10;
gainscale=[0,2,4,10];
gain_Interpolant=griddedInterpolant([0,1,2,3], gainscale, 'pchip', 'nearest');
check_interval = 1000;
check_interval_growth_factor=1.1;
check_interval_maxsteps=2e5;
totalbatches=floor(log(check_interval_maxsteps/check_interval)/log(check_interval_growth_factor));
totalsteps=sum(check_interval.*check_interval_growth_factor.^(1:totalbatches)')

% -------------------- 2. Derived Physical Params -----------------------
gCS = (1 - S.phi/2) / (1 - S.phi)^3; % Carnahan-Starling Contact Value g(sigma) - probability of finding two particles touching each other compared to an ideal gas
diffE = S.esdiff * S.alpha / gCS; % Enskog Effective Diffusivity - how fast a particle diffuses inside a dense crowd - corrected by alpha
tau_alpha = (S.rp^2) / (6 * diffE); % Structural relaxation time - characteristic time it takes for a particle to diffuse a distance equal to its own radius
relaxsteps = ceil(tau_alpha / S.timestep); % Structural relaxation STEPS

% Depth of the correction 
taper_width = 1.0 * S.rp; % width of the taper in radius units
potdepth = S.rc+taper_width;
if 2*S.br - 2*potdepth < 0, potdepth = S.br; end % if correction depth is larger than the domain radius, just use the domain radius 

% #################################################################################
% -------------------- 2. FILENAMES -------------------------
% #################################################################################
% name the potentials
if S.potential==1, potname='lj'; elseif S.potential==2, potname='wca'; else potname='hs'; end

% name the series with the specific name passed on from opts
if isfield(opts, 'series_name')
    seriesname = opts.series_name; % Uses the Unique ID from barebones (e.g., Rep1, Rep2)
else
    seriesname = 'scf.anneal';
end

% define specific names of the correction and the starting config
filenamecorrection = sprintf(['ASYMCORR_',seriesname,'_%s_%.0e_%.0e_%.0f_%.1f_%.1e.mat'],...
    potname,S.rp,S.phi,S.N,S.pot_epsilon/S.kbT,S.pot_sigma);
filestartingconfiguration = sprintf(['START_SBC_',seriesname,'_%s_%.0e_%.0e_%.0f_%.1f_%.1e.mat'],...
    potname,S.rp,S.phi,S.N,S.pot_epsilon/S.kbT,S.pot_sigma);

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
% -------------------- 6. SCF STATE INITALIZATION ---------------------------------
% #################################################################################

if S.pot_corr

    % --- Leaky Accumulator Params ---
       % How often to check for the 3-Sigma trigger
	dr_step=zeros(S.N,check_interval);
	base_step=zeros(S.N,check_interval);
	bin_idx=zeros(S.N,check_interval);
    sigmadrift=3;
    gaindrift=0.1;
    steps_since_plot=0;

	% --- RADIAL RANGE OF CORRECTION -------------------------------------------------------------------------------------------------
	scf.edges = sort([0;(S.br:-0.05*S.rp:0)']); % in two hundreths of a radius
	scf.bins = numel(scf.edges) - 1;
	scf.centers = scf.edges(1:end-1) + diff(scf.edges)/2;
	scf.vols=(4/3)*pi*(scf.edges(2:end)).^3 - (4/3)*pi*(scf.edges(1:end-1)).^3;
	scf.correction = zeros(scf.bins, 1);
	scf.mask=scf.centers>(S.br-potdepth);
	scf.modes.leftovers=mod(scf.bins,(1:scf.bins)');
	scf.modes.bins=find(scf.modes.leftovers<(0.1*scf.bins));
	for im=1:numel(scf.modes.bins)
		temp=scf.centers(scf.modes.leftovers(scf.modes.bins(im))+1:end,:);
		templ=scf.centers(1:scf.modes.leftovers(scf.modes.bins(im)));
		temp=reshape(temp',[],floor(scf.bins/scf.modes.bins(im)))';
		scf.modes.centers{im,1}=mean(temp,2);
		scf.modes.centers{im,1}=[mean(templ);scf.modes.centers{im,1}];
		scf.modes.centers{im,1}=scf.modes.centers{im,1}(~isnan(scf.modes.centers{im,1}),:);
	end
	
	% --------------------------------------------------------------------------------------------------------------------------------
	
	
	% --- PDF INITIALIZATION pdf ---------------------------------------------------------------------------------------
	pdf.edges = sort([0;(2*S.br:-0.05*S.rp:0)']); % in 5 hundreths of a radius
	pdf.bins = numel(pdf.edges) - 1;
	pdf.centers = pdf.edges(1:end-1) + diff(pdf.edges)/2;
	pdf.vols=(4/3)*pi*(pdf.edges(2:end)).^3 - (4/3)*pi*(pdf.edges(1:end-1)).^3;
	pdf.r_norm = pdf.centers / S.br; % Geometric Form Factor - Normalize distance by Box Radius (S.br)
    pdf.geom_factor = 1 - (3/4)*pdf.r_norm + (1/16)*pdf.r_norm.^3; % The Finite Volume Correction Polynomial
    pdf.geom_factor = max(0, pdf.geom_factor); % Clamp negative values to 0
    ndens.ndens0 = (S.N / S.bv);
    pdf.denom = 0.5 * (ndens.ndens0 * pdf.geom_factor * S.N) .* pdf.vols;
	pdf.counts = zeros(size(pdf.centers,1), 1);
	ndens.counts = zeros(size(scf.centers,1), 1);
	% ----------------------------------------------------------------------------------------------------------------
	
	% --- TAPER ON CORRECTION  -------------------------------------------------------------------------------------------------------
	% force correction to taper to zero at the inner edgee of the correction radial range to prevent impulsive forces.
	r_inner_edge = (S.br-potdepth); % inner edge of the correction radial range
	taper_mask = zeros(scf.bins, 1); % initialize the taper mask
	% loop creating a linear taper starting from r_inner_edge at 0 and capping at 1 at r_inner_edge+taper_width
	for itaper = 1:scf.bins
		taper_mask(itaper) = min(1, max(0, (scf.centers(itaper) - r_inner_edge) / taper_width));
	end
	% --------------------------------------------------------------------------------------------------------------------------------

	% --- INITIALIZE INTERPOLANT OF CORRECTION ---------------------------------------------------------------------------------------
	F_corr_interp = griddedInterpolant(scf.centers, scf.correction, 'linear', 'nearest');
	% --------------------------------------------------------------------------------------------------------------------------------

	% --- THERMALIZATION CONDITIONS --------------------------------------------------------------------------------------------------
	thermflag = 0;
	therm_block_size = max(ceil(relaxsteps), 1000);
	% --------------------------------------------------------------------------------------------------------------------------------

	% --- INITIALIZE PLOTTING AND MONITORING ------------------------------------------------------------------------------
	if graphing 		
		% Plot Setup
        f_diag = figure('Color', 'k', 'Name', 'Kinetic SCF Diagnostics');
        ax1 = subplot(2,2,1); ax2 = subplot(2,2,2); 
        ax3 = subplot(2,2,3); ax4 = subplot(2,2,4);
		
		% --- Add to the existing S.pot_corr initialization block ---
		% History buffer for heatmap (Radius x Batch)
		% We'll start with a fixed size and grow it if needed
		correction_history = zeros(scf.bins, 500); 
		batch_count = 0;

		f_modal = figure('Color', 'k', 'Name', 'Modal Spectrum & Stability');
		ax_spec = subplot(2,2,1); title(ax_spec, 'Modal Update Spectrum', 'Color', 'w');
		ax_res  = subplot(2,2,2); title(ax_res, 'Residual Flux vs. Correction', 'Color', 'w');
		ax_heat = subplot(2,1,2); title(ax_heat, 'Correction Evolution (Heatmap)', 'Color', 'w');
		end
	% -----------------------------------------------------------------------------------------------------------------------
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% -------------------- 7. MAIN LOOP -----------------------------------------------
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if S.pot_corr
    fprintf('Starting SCF V3 (Annealing)');
else
    fprintf('Thermalizing structure')
end

% ---- CALCULATION OF DISPLACEMENT LIBRARIES ----
DISP=build_noise_library(S.stdx,1e6);
qd=1;
% -----------------------------------------------

qs = 0; % step counter
exitflag=false;
% Replace the cell initialization with this:
modal_lines = struct('r', [], 'u', [], 'k', []);

while exitflag==false
	qs = qs + 1; % update counter
	
	% --- DEFINE STARTING CONFIGURATION ----------------------------
	% norms of real particles
    prho = vecnorm(p, 2, 2);
	% versors of real particles
    pvers = p ./ (prho + eps);
	% mask of ghost generators
    idxgp = prho > (S.br - S.rc);
	
	% --------------------------------------------------------------
    
    % --- PDF-BASED THERMALIZATION CHECK -------------------------------------------------------
    if thermflag == 0 && qs > therm_block_size
        thermflag=1;
		disp('--- Thermalization Complete (Structural Convergence) ---');
		qs = 1; % We reset this to start the SGD counter
		if S.pot_corr==0
			ASYMCORR=0;
			return
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
    % --- KINETIC SCF ENGINE -----------------------------------------------------
    % ############################################################################
    if thermflag == 1
        % 1. CONTINUOUS ACCUMULATION OF DISPLACEMENT INFORMATION
		dr_corr_mag = F_corr_interp(prho);
		base_step(:,mod(qs-1,check_interval)+1) = vecnorm(base_disp,2,2)+dr_corr_mag; %  displacement norm
        dr_step(:,mod(qs-1,check_interval)+1) = sum(base_disp .* pvers, 2)+dr_corr_mag; % radial displacement
        bin_idx(:,mod(qs-1,check_interval)+1) = discretize(prho, scf.edges); % bins of radial displacements
        
        % 2. CONTINUOUS ACCUMULATION OF NUMBER DENSITY and DISTANCES FOR PDF
		if mod(qs,pdfdens_int)==0
			% number density accumulation
			[hc_dens, ~] = histcounts(prho, scf.edges);
			ndens.counts = ndens.counts + hc_dens';
			% number density accumulation
			pairdists = pdist(p); % get distances
			[hc_pdf, ~] = histcounts(pairdists, pdf.edges); % bin them
			if numel(hc_pdf) == numel(pdf.counts) % store them
				pdf.counts = pdf.counts + hc_pdf';
			end
		end

        % 3. EVALUATE & CORRECT (Every batch_size)
        if mod(qs, check_interval) == 0
			comp = accumarray(bin_idx(:), dr_step(:), [numel(scf.centers) 1], @(x) {x});
			comp_base = accumarray(bin_idx(:), base_step(:), [numel(scf.centers) 1], @(x) {x});
            update_full = zeros(scf.bins,1);
			weight_full = zeros(scf.bins,1);
            qcenters=0;
			for ib=scf.modes.bins'
                qcenters=qcenters+1;
                scftempcenters=scf.modes.centers{qcenters,1};
				if ib==1
					compb=comp;
					compb_base=comp_base;
                else
                    if scf.modes.leftovers(ib)~=0
					    compl=comp(1:scf.modes.leftovers(ib),:);
					    compl_base=comp_base(1:scf.modes.leftovers(ib),:);
                        compl = cellfun(@(x) vertcat(compl{:, x}), num2cell(1:size(compl,2)), 'UniformOutput', false)';
					    compl_base = cellfun(@(x) vertcat(compl_base{:, x}), num2cell(1:size(compl_base,2)), 'UniformOutput', false)';
                    else
                        compl=[];
                        compl_base=[];
                    end
					compb=comp(scf.modes.leftovers(ib)+1:end,:);
					compb_base=comp_base(scf.modes.leftovers(ib)+1:end,:);
					compb=reshape(compb',[],floor(scf.bins/ib));
					compb_base=reshape(compb_base',[],floor(scf.bins/ib));
					compb = cellfun(@(x) vertcat(compb{:, x}), num2cell(1:size(compb,2)), 'UniformOutput', false)';
					compb_base = cellfun(@(x) vertcat(compb_base{:, x}), num2cell(1:size(compb_base,2)), 'UniformOutput', false)';					
					compb=vertcat(compl,compb);
					compb_base=vertcat(compl_base,compb_base);
				end
				nocomp=size(compb,1);
				counts=[];
				mu_raw=[];
				radle=[];
				mean_abs=[];
				drift=[];
				var_r=[];
				snr=[];
				for i0=1:nocomp					
					counts(i0,1)=numel(compb{i0,1});
					mu_raw(i0,1)=mean(compb{i0,1});
					radle(i0,1)=mean(compb{i0,1}.*compb_base{i0,1});
					mean_abs(i0,1) = mean(compb_base{i0,1});

					if counts(i0,1)>100
						dr_edges=linspace(-max(abs(compb{i0,1})),max(abs(compb{i0,1})),100)';
						dr_centers=dr_edges(1:end-1,1)+0.5*diff(dr_edges(:,1));
						try
							[hctemp,~]=histcounts(compb{i0,1},dr_edges);
							[drifttemp,~,~, ~]=createPVFit(dr_centers, hctemp',0.999);
							% 1. FIX: Noise is the standard error
							snr(i0,1)=0.5*(drifttemp(4,3)-drifttemp(4,2));
							drift(i0,1) = drifttemp(4,1) ;
						catch
							snr(i0,1)=inf;
							drift(i0,1)=0;
						end
					else
						snr(i0,1)=inf;
						drift(i0,1)=0;
					end
				end
				snr = drift ./ snr;
				snr(isnan(snr),:)=0;
				% 3. Calculate the modally resolved flux
				radle_corr = radle - mu_raw .* mean_abs;
                if size(drift,1)~=size(scftempcenters,1)
                    flag
                end

				flux = drift + (radle_corr ./ scftempcenters);
				
				% 2. Calculate Commensurate Gain Multiplier
				gain_multiplier = gain_Interpolant(abs(snr));
				update_mag = (1e-2*gain_multiplier) .* flux;
                update_mag(isnan(update_mag))=0;
				
				% --- MAP MODAL UPDATE BACK TO FULL RESOLUTION --------------------

				% interpolate modal update onto full grid
				if numel(scftempcenters) > 1 && numel(update_mag) > 1 
                    u_interp = interp1(scftempcenters, update_mag, scf.centers, 'linear', 'extrap');
                else
                    u_interp = update_mag(1) * ones(size(scf.centers));
                end
				
				if graphing
					% Store the interpolated update for this mode to plot later
					modal_lines(ib).r = scf.centers;
					modal_lines(ib).u = u_interp;
					modal_lines(ib).k = ib;
					if ib == 1
						high_res_flux = flux; % This is the raw evidence the SCF is trying to fix
					end
				end
				
				w_mode = (1/ib) * ones(size(u_interp)); % 'ib' is the aggregation factor
				update_full = update_full + w_mode .* u_interp;
				weight_full = weight_full + w_mode;
			end
			
			% --- COMBINE MODES AND APPLY CORRECTION --------------------------

			valid = weight_full > 0;
			update_combined = zeros(size(update_full));
			update_combined(valid) = update_full(valid) ./ weight_full(valid);

			batchcorr = -(update_combined .* taper_mask);
			batchcorr = smoothdata(batchcorr, 'sgolay', corr_smooth_window);

			scf.correction = scf.correction + batchcorr;
			F_corr_interp.Values = scf.correction;

            
            % C. DIAGNOSTICS (Plot every batch or so)
            if graphing
				% DENSITY PLOT
				ndens.meancounts=(ndens.counts./(check_interval/pdfdens_int));
				ndens.meandens=ndens.meancounts./scf.vols;
				ndens.reldens=ndens.meandens./ndens.ndens0;
                plot(ax1, scf.centers/S.rp, ndens.reldens, 'w', 'LineWidth', 2);
                hold(ax1, 'on'); yline(ax1, 1, '--g'); hold(ax1, 'off');
				xlim(ax1,[S.br/(3*S.rp) S.br/(S.rp)]) 
				% PDF PLOT
				pdf.curr_g = (pdf.counts / (check_interval/pdfdens_int)) ./ pdf.denom;
                plot(ax2, pdf.centers/S.rp, pdf.curr_g, 'c', 'LineWidth', 2);
				hold(ax2, 'on'), yline(ax2, 1, '--r'), hold(ax2, 'off')
				% CORRECTION PLOT
                plot(ax3, scf.centers/S.rp, scf.correction, 'c', 'LineWidth', 2);
				xlim(ax3,[(S.br-potdepth)/(S.rp) S.br/(S.rp)])
                % SNR PLOT
                bar(ax4, scf.centers/S.rp, snr);
				hold(ax4, 'on');				
                yline(ax4, sigmadrift, '-r');
				yline(ax4, -sigmadrift, '-r');
				hold(ax4, 'off');
                drawnow;
                
                % Clear density diagnostic (purely for plotting clarity)
                ndens.counts(:) = 0;	
                pdf.counts(:) = 0;
            end
			if graphing
				batch_count = batch_count + 1;
				correction_history(:, batch_count) = scf.correction;

				% --- Plot 1: Modal Spectrum (Upper Left) ---
				cla(ax_spec); hold(ax_spec, 'on');
				colors = jet(scf.bins);
				for ib = 1:numel(scf.modes.bins)
                    ibi=scf.modes.bins(ib);
					plot(ax_spec, modal_lines(ibi).r/S.rp, modal_lines(ibi).u, ...
						 'Color', colors(ibi,:), 'LineWidth', 1.5, ...
						 'DisplayName', ['k=' num2str(modal_lines(ibi).k)]);
				end
				grid(ax_spec, 'on'); set(ax_spec, 'XColor', 'w', 'YColor', 'w');
				ylabel(ax_spec, 'Update Mag');

				% --- Plot 2: Residual vs Correction (Upper Right) ---
				% We compare the raw evidence (Flux) with the cumulative potential built (Correction)
				cla(ax_res); hold(ax_res, 'on');
				% Normalize for visualization if orders of magnitude differ
				plot(ax_res, scf.centers/S.rp, high_res_flux, 'y', 'LineWidth', 2, 'DisplayName', 'Raw Flux');
				plot(ax_res, scf.centers/S.rp, scf.correction, 'c--', 'LineWidth', 1.5, 'DisplayName', 'Total Corr');
				legend(ax_res, 'TextColor', 'w', 'Location', 'best');
				grid(ax_res, 'on'); set(ax_res, 'XColor', 'w', 'YColor', 'w');

				% --- Plot 3: Stability Heatmap (Bottom) ---
				% X is batch number, Y is Radius
				imagesc(ax_heat, 1:batch_count, scf.centers/S.rp, correction_history(:, 1:batch_count));
				colormap(ax_heat, parula); colorbar(ax_heat, 'Color', 'w');
				xlabel(ax_heat, 'Batch Number'); ylabel(ax_heat, 'Radius (r/rp)');
				set(ax_heat, 'XColor', 'w', 'YColor', 'w');
				title(ax_heat, 'Correction History (Stability Check)', 'Color', 'w');

				drawnow;
			end
			check_interval=floor(check_interval_growth_factor*check_interval);
			if check_interval>check_interval_maxsteps
				check_interval=check_interval_maxsteps;
				exitflag=1;
			end
			base_step=zeros(S.N,check_interval);
			dr_step=zeros(S.N,check_interval);
			bin_idx=zeros(S.N,check_interval);
			qs=0;
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

    if qs > 1e6, fprintf('Internal safety limit (1e6) reached.\n'); break; end
end

ASYMCORR.correction = [scf.centers, scf.correction];
ASYMCORR.history = history; ASYMCORR.S=S;
if enable_io
    save([data_folder,'\',filenamecorrection], 'ASYMCORR', 'scf.edges');
    save([data_folder,'\',filestartingconfiguration], 'p', 'pgp', 'S');
end
disp('SGD V10 Complete.');
end